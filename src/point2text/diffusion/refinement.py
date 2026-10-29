"""Budgeted canonical feedback around coarse diffusion and local discrete edits."""

from __future__ import annotations

from dataclasses import asdict, replace
import re
import threading
import time

from ..contracts import Candidate
from .contracts import DiffusionRequest, DiffusionResult
from .guidance import GuidanceConfig, SparseMapGuidance, _json_safe
from .refinement_contracts import RefinementConfig, TextProposal
from .search import technical_prompt
from .verification import CanonicalVerifier


class _DeadlineSignal:
    """Native stop signal; deadline expiry does not mutate the user's event."""

    def __init__(self, user_event, verifier):
        self.user_event = user_event
        self.verifier = verifier

    def is_set(self):
        return self.user_event.is_set() or self.verifier.budget_reason() == "time_budget"


class CanonicalRefinementSearch:
    def __init__(self, runtime, reward=None, guidance_config=None, edit_proposer=None,
                 config: RefinementConfig | None = None, *, quality_callback=None) -> None:
        self.runtime = runtime
        self.reward = reward
        self.guidance_config = guidance_config or GuidanceConfig()
        self.edit_proposer = edit_proposer
        self.config = config or RefinementConfig()
        self.quality_callback = quality_callback

    def run(self, bundle, target, *, max_candidates=1, seed=42, on_candidate=None,
            cancel_event=None, request=None, guided=True, initial_text=None) -> DiffusionResult:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 6:
            raise ValueError("max_candidates must be between 1 and 6")
        cancel_event = cancel_event if cancel_event is not None else threading.Event()
        verifier = CanonicalVerifier(bundle, target, self.config,
                                     quality_callback=self.quality_callback)
        native_stop = _DeadlineSignal(cancel_event, verifier)
        target = verifier.target
        started = time.monotonic()
        if request is None:
            prompt = technical_prompt(bundle.nearest(target, k=4)) if self.runtime is not None else ""
            request = DiffusionRequest(prompt=prompt, seed=seed)
            prompt_policy = "nearest-four-context-v1" if self.runtime is not None else "no_generator"
        else:
            seed = request.seed
            prompt_policy = "supplied_request"
        best = None
        beam = []
        history = []
        native_traces = []
        costs = {"proposer_seconds": 0.0, "native_seconds": 0.0,
                 "verification_seconds": 0.0, "repair_calls": 0}
        errors = []
        epsilon_confirmed = False
        current_seed = seed
        rounds = 0
        stall = 0
        stop_reason = None

        def halted():
            nonlocal stop_reason
            if cancel_event.is_set():
                stop_reason = "cancelled"
            elif epsilon_confirmed:
                stop_reason = "epsilon_reached"
            elif verifier.budget_reason():
                stop_reason = "budget_exhausted"
            return stop_reason is not None

        def consider(proposals, phase):
            nonlocal best, beam, epsilon_confirmed, stop_reason
            before = time.monotonic()
            verified = verifier.verify_batch(proposals)
            costs["verification_seconds"] += time.monotonic() - before
            for value in sorted(verified, key=lambda item: item.distance):
                if value.distance <= self.config.epsilon:
                    before = time.monotonic()
                    confirmed = verifier.confirm_epsilon(value)
                    costs["verification_seconds"] += time.monotonic() - before
                    if confirmed:
                        epsilon_confirmed = True
                        value = verifier.last_confirmed
                    else:
                        checks = verifier.confirmation.get("checks", [])
                        if checks:
                            worst = max(checks, key=lambda item: item["distance"])
                            value = replace(value, x=worst["x"], y=worst["y"],
                                            distance=worst["distance"],
                                            diagnostics={**value.diagnostics,
                                                         "epsilon_confirmation": dict(verifier.confirmation)})
                        elif verifier.budget_reason(confirmation=True):
                            stop_reason = "budget_exhausted"
                        if str(verifier.confirmation.get("reason", "")).startswith("confirmation_error"):
                            errors.append(verifier.confirmation["reason"])
                            stop_reason = "error"
                        elif verifier.confirmation.get("reason", "").startswith("quality_callback"):
                            continue
                beam.append(value)
                if best is None or value.distance < best.distance:
                    previous = best.distance if best is not None else None
                    best = value
                    item = {"phase": phase, "round": rounds, "seed": current_seed,
                            "previous_distance": previous, "distance": value.distance,
                            "x": value.x, "y": value.y, "text": value.text,
                            "elapsed_seconds": time.monotonic() - started,
                            "diagnostics": value.diagnostics}
                    history.append(item)
                    if on_candidate is not None:
                        on_candidate(Candidate(value.text, value.x, value.y, value.distance,
                                               item["elapsed_seconds"], current_seed))
                if epsilon_confirmed:
                    break
            beam = sorted({item.text: item for item in beam}.values(),
                          key=lambda item: item.distance)[:self.config.beam_width]

        def observer_for(phase):
            def observe(draft):
                if halted():
                    return True
                consider([TextProposal(draft.text, provenance={"phase": phase,
                                                               "native_step": draft.step})], phase)
                return halted()
            return observe

        def make_hook():
            if not guided or self.guidance_config.strength == 0:
                return None
            if self.reward is None:
                raise ValueError("Guided native refinement requires an encoder reward")
            return SparseMapGuidance(self.reward, target, self.guidance_config)

        def run_native(phase, native_request, *, text=None, span=None):
            hook = make_hook()
            before = time.monotonic()
            kwargs = dict(hook=hook, observer=observer_for(phase),
                          observe_every=self.config.observe_every, cancel_event=native_stop)
            try:
                if text is None:
                    result = self.runtime.generate(native_request, **kwargs)
                else:
                    costs["repair_calls"] += 1
                    result = self.runtime.refine_text(native_request, text,
                                                      edit_span=span, **kwargs)
                native_traces.append({"phase": phase, "request": asdict(native_request),
                                      "edit_span": span, "runtime": result.trace,
                                      "guidance": hook.trace if hook else None})
                if not cancel_event.is_set() and not epsilon_confirmed:
                    consider([TextProposal(result.text, edit_span=span,
                                           provenance={"phase": phase})], phase)
            except Exception as exc:
                native_traces.append({"phase": phase, "request": asdict(native_request),
                                      "edit_span": span, "error": str(exc),
                                      "guidance": hook.trace if hook else None})
                raise
            finally:
                costs["native_seconds"] += time.monotonic() - before

        try:
            if initial_text is not None and not halted():
                consider([TextProposal(initial_text, provenance={"phase": "initial_text"})],
                         "initial_text")
            elif self.runtime is not None:
                for index in range(max_candidates):
                    if halted():
                        break
                    current_seed = seed + index
                    run_native("coarse", replace(request, seed=current_seed))
            if not halted() and best is not None:
                proposer = self.edit_proposer
                if proposer is None:
                    try:
                        from .discrete import CanonicalEditProposer
                        proposer = CanonicalEditProposer(bundle, reward=self.reward)
                    except ImportError:
                        proposer = None
                for round_index in range(self.config.max_rounds):
                    if halted():
                        break
                    rounds = round_index + 1
                    old_distance = best.distance
                    proposals = []
                    if proposer is not None:
                        before = time.monotonic()
                        sources = list(beam) or [best]
                        remaining = self.config.proposals_per_round
                        for index, source in enumerate(sources):
                            if halted() or remaining <= 0:
                                break
                            allowance = max(1, remaining // (len(sources) - index))
                            proposed = list(proposer.propose(source.text, target,
                                                            max_proposals=allowance))[:allowance]
                            proposals.extend(proposed)
                            remaining -= len(proposed)
                        costs["proposer_seconds"] += time.monotonic() - before
                    if proposals and not halted():
                        consider(proposals, "discrete_edit")
                    meaningful = old_distance - best.distance > self.config.min_improvement
                    stalled_round = not meaningful
                    if (not halted() and self.runtime is not None
                            and hasattr(self.runtime, "refine_text")
                            and (stalled_round or rounds % self.config.repair_every == 0)):
                        span = self._repair_span(best.text, target, proposer, round_index)
                        if halted():
                            break
                        if span is not None:
                            current_seed = seed + max_candidates + round_index
                            repair_request = replace(request, seed=current_seed,
                                                     steps=self.config.repair_steps)
                            run_native("native_repair", repair_request, text=best.text, span=span)
                    meaningful = old_distance - best.distance > self.config.min_improvement
                    stall = 0 if meaningful else stall + 1
                    if stall >= self.config.stall_rounds:
                        stop_reason = "stalled"
                        break
                if stop_reason is None:
                    halted()
                if stop_reason is None:
                    stop_reason = "budget_exhausted"
            if stop_reason is None:
                halted()
            if stop_reason is None:
                stop_reason = "no_valid_text" if best is None else "stalled"
        except Exception as exc:
            errors.append(str(exc))
            stop_reason = "cancelled" if cancel_event.is_set() else "error"

        if cancel_event.is_set():
            stop_reason = "cancelled"
        elif epsilon_confirmed:
            stop_reason = "epsilon_reached"
        status = "cancelled" if cancel_event.is_set() else "error" if errors or best is None and stop_reason == "no_valid_text" else "complete"
        candidates = [Candidate(item["text"], item["x"], item["y"], item["distance"],
                                item["elapsed_seconds"], item["seed"]).payload() for item in history]
        best_payload = None
        if best is not None:
            best_payload = dict(candidates[-1])
        token_ids = []
        if best is not None and self.runtime is not None:
            try:
                token_ids = list(self.runtime.tokenizer.encode(best.text, add_special_tokens=False))
            except Exception:
                pass
        trace = {
            "strategy": "canonical-feedback-refinement-v1", "status": status,
            "stop_reason": stop_reason, "error": errors[0] if errors else None,
            "errors": errors, "epsilon": self.config.epsilon,
            "epsilon_confirmed": epsilon_confirmed,
            "epsilon_worst_distance": verifier.confirmation.get("worst_distance"),
            "epsilon_confirmation": dict(verifier.confirmation),
            "map_id": getattr(bundle, "map_id", None), "target": list(target),
            "distance_metric": "euclidean-unwhitened-pca",
            "config": asdict(self.config), "request": asdict(request),
            "prompt_policy": prompt_policy, "requested_candidates": max_candidates,
            "runtime_identity": self.runtime.identity if self.runtime is not None else None,
            "elapsed_seconds": time.monotonic() - started, "rounds": rounds,
            "stall_rounds": stall, "incumbent_improvements": len(history),
            "hard_queries": verifier.counts["hard_queries"],
            "accepted_proposals": verifier.counts["accepted"],
            "rejected_proposals": verifier.counts["rejected"],
            "budget_kind": verifier.budget_reason() or
                           ("round_limit" if stop_reason == "budget_exhausted" else None),
            "candidates": candidates, "best_candidate": best_payload,
            "history": history, "native_traces": native_traces, "costs": costs,
            "verification": verifier.diagnostics,
            "quality_validation_state": verifier.diagnostics["quality_validation_state"],
            "epsilon_is_attainability_claim": False,
        }
        return DiffusionResult(best.text if best else "", token_ids, _json_safe(trace))

    @staticmethod
    def _repair_span(text, target, proposer, round_index):
        words = list(re.finditer(r"\b\w+(?:[-'’]\w+)*\b", text))
        if not words:
            return None
        spans = []
        if proposer is not None and hasattr(proposer, "hotspots"):
            spans = list(proposer.hotspots(text, target, limit=4))
        valid = []
        for item in spans:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            start, end = item
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
                continue
            overlaps = [word for word in words if word.start() < end and word.end() > start]
            if overlaps:
                valid.append((overlaps[0].start(), overlaps[min(3, len(overlaps)-1)].end()))
        if valid:
            return valid[round_index % len(valid)]
        word = words[round_index % len(words)]
        return word.start(), word.end()
