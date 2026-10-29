"""Hard-verified generation using a native diffusion runtime and optional guidance."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
import threading
import time
from typing import Callable

from ..contracts import Candidate
from ..generation import OllamaGenerator
from .contracts import DiffusionRequest, DiffusionResult
from .guidance import GuidanceConfig, SparseMapGuidance, _json_safe


def technical_prompt(examples) -> str:
    """Use the same example context as the scoring-only text baseline, without HTTP."""
    return (
        "Write one original, accurate technical or research paragraph of 60–100 words. "
        "Return only the paragraph, without a heading, bullets, citations or reasoning. "
        "Do not copy example sentences, invent references, or invent measurements.\n\n"
        + OllamaGenerator._prompt(examples)
    )


class DiffusionSearch:
    def __init__(self, runtime, reward=None,
                 config: GuidanceConfig | None = None) -> None:
        self.runtime = runtime
        self.reward = reward
        self.config = config or GuidanceConfig()

    def run(self, bundle, target, *, max_candidates: int = 1, seed: int = 42,
            on_candidate: Callable[[Candidate], None] | None = None,
            cancel_event: threading.Event | None = None,
            request: DiffusionRequest | None = None,
            guided: bool = True) -> DiffusionResult:
        """Return the best real paragraph, with every verified candidate in trace.

        Supply the identical request to guided and unguided calls for a paired
        comparison. No surrogate map loss is ever used to rank final candidates.
        """
        if not isinstance(max_candidates, int) or not 1 <= max_candidates <= 6:
            raise ValueError("max_candidates must be between 1 and 6")
        target = tuple(float(value) for value in target)
        if len(target) != 2 or not all(math.isfinite(value) for value in target):
            raise ValueError("Target must contain two finite map coordinates")
        if guided and self.reward is None and self.config.strength != 0:
            raise ValueError("Guided diffusion requires an encoder reward adapter")
        cancel_event = cancel_event if cancel_event is not None else threading.Event()
        started = time.perf_counter()
        if request is None:
            request = DiffusionRequest(prompt=technical_prompt(bundle.nearest(target, k=4)),
                                       seed=seed)
        else:
            # A supplied request owns its seed, which makes paired replay explicit.
            seed = request.seed
        candidates: list[Candidate] = []
        trajectories: list[dict] = []
        seen: set[str] = set()
        best_result: DiffusionResult | None = None
        best_candidate: Candidate | None = None
        error = None
        for index in range(max_candidates):
            if cancel_event.is_set():
                break
            candidate_request = replace(request, seed=seed + index)
            hook = SparseMapGuidance(self.reward, target, self.config) if guided else None
            trajectory = {"seed": candidate_request.seed,
                          "request": asdict(candidate_request)}
            trajectories.append(trajectory)
            try:
                generation_started = time.perf_counter()
                result = self.runtime.generate(candidate_request, hook=hook,
                                               cancel_event=cancel_event)
                trajectory["generation_seconds"] = time.perf_counter() - generation_started
                trajectory["runtime"] = _json_safe(result.trace)
                trajectory["guidance"] = hook.trace if hook else None
                if cancel_event.is_set():
                    trajectory["status"] = "cancelled"
                    break
                text = result.text.strip()
                key = " ".join(text.split()).casefold()
                if not key or key in seen:
                    trajectory["status"] = "empty" if not key else "duplicate"
                    continue
                seen.add(key)
                score_started = time.perf_counter()
                # Canonical text tokenization and encoder forward, never soft inputs.
                (x, y), distance = bundle.score(text, target)
                trajectory["hard_score_seconds"] = time.perf_counter() - score_started
                if not all(math.isfinite(float(value)) for value in (x, y, distance)) or distance < 0:
                    raise RuntimeError("Canonical encoder returned an invalid map score")
                candidate = Candidate(text=text, x=float(x), y=float(y),
                                      distance=float(distance),
                                      elapsed_seconds=time.perf_counter() - started,
                                      seed=candidate_request.seed)
                candidates.append(candidate)
                trajectory["status"] = "verified"
                trajectory["hard_candidate"] = candidate.payload()
                if best_candidate is None or candidate.distance < best_candidate.distance:
                    best_candidate, best_result = candidate, result
                if on_candidate is not None:
                    on_candidate(candidate)
            except Exception as exc:
                if cancel_event.is_set():
                    trajectory["status"] = "cancelled"
                    break
                trajectory["status"] = "error"
                trajectory["error"] = str(exc)
                error = str(exc)
                if not candidates:
                    raise
                break  # Retain already verified results if a later trajectory fails.
        status = "cancelled" if cancel_event.is_set() else "error" if error else "complete"
        if not candidates and status == "complete":
            raise RuntimeError("Diffusion produced no usable decoded text to verify")
        trace = {
            "strategy": "experimental-frozen-diffusion-map-guidance-v1" if guided
                        else "native-diffusion-unguided-v1",
            "status": status, "error": error, "map_id": bundle.map_id,
            "target": list(target), "runtime_identity": self.runtime.identity,
            "distance_metric": "euclidean-unwhitened-pca",
            "request": asdict(request), "requested_candidates": max_candidates,
            "elapsed_seconds": time.perf_counter() - started,
            "candidates": [candidate.payload() for candidate in candidates],
            "best_candidate": best_candidate.payload() if best_candidate else None,
            "trajectories": trajectories,
        }
        return DiffusionResult(text=best_candidate.text if best_candidate else "",
                               token_ids=list(best_result.token_ids) if best_result else [],
                               trace=_json_safe(trace))
