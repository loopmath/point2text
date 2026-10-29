"""Scoring-only search, independent of encoder and generator implementations."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

from .contracts import Candidate, TextGenerator


class BaselineSearch:
    def __init__(self, generator: TextGenerator) -> None:
        self.generator = generator

    def run(self, bundle, target, *, max_candidates: int, seed: int,
            on_candidate: Callable[[Candidate], None],
            cancel_event: threading.Event) -> list[Candidate]:
        """Return every unique, canonically verified candidate in generation order.

        ``max_candidates`` bounds generation attempts, including empty/duplicate
        outputs. Each attempt is independent; previous outputs are never fed back
        to the generator. The caller chooses the best result by distance.
        """
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if cancel_event.is_set():
            return []
        started = time.perf_counter()
        examples = bundle.nearest(target, k=getattr(getattr(self.generator, "prompt_policy", None), "neighbors", 4))
        candidates: list[Candidate] = []
        seen: set[str] = set()
        for attempt in range(max_candidates):
            if cancel_event.is_set():
                break
            candidate_seed = seed + attempt
            text = self.generator.generate(examples, seed=candidate_seed,
                                           cancel_event=cancel_event).strip()
            if cancel_event.is_set():
                break
            key = " ".join(text.split()).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            # The actual decoded text always passes through the canonical encoder.
            (x, y), distance = bundle.score(text, target)
            if not all(math.isfinite(float(value)) for value in (x, y, distance)):
                raise RuntimeError("The encoder returned a non-finite map score. "
                                   "Rebuild the map or check the encoder configuration.")
            if distance < 0:
                raise RuntimeError("The map returned an invalid negative distance.")
            if cancel_event.is_set():
                break
            candidate = Candidate(text=text, x=float(x), y=float(y),
                                  distance=float(distance),
                                  elapsed_seconds=time.perf_counter() - started,
                                  seed=candidate_seed)
            candidates.append(candidate)
            on_candidate(candidate)
        if not candidates and not cancel_event.is_set():
            raise RuntimeError("The generator produced no usable paragraph. "
                               "Try again or choose a different installed model.")
        return candidates
