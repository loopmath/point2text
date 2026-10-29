"""Budgeted canonical feedback around coarse diffusion and local discrete edits."""

from __future__ import annotations

from dataclasses import asdict, replace
from contextlib import contextmanager
import math
import re
import threading
import time

from ..contracts import Candidate
from .contracts import DiffusionRequest, DiffusionResult
from .guidance import GuidanceConfig, SparseMapGuidance, _json_safe
from .refinement_contracts import RefinementConfig, TextProposal
from .search import technical_prompt
from .verification import CanonicalVerifier
from .work import native_work_summary


def _proposal_policy(proposer):
    try:
        provider = getattr(proposer, "allowed_replacements", None)
        if provider is not None:
            return _json_safe(getattr(provider, "trace", None))
        provider = getattr(proposer, "provider", None)
        diagnostics = getattr(proposer, "diagnostics", None)
        if diagnostics is not None:
            return _json_safe({"bank": diagnostics,
                               "provider": getattr(provider, "trace", getattr(provider, "diagnostics", None))})
        return None
    except Exception as exc:
        return {"trace_error": str(exc)}


def _numeric_deltas(after, before):
    if not isinstance(after, dict):
        return {}
    before = before if isinstance(before, dict) else {}
    result = {}
    for key, value in after.items():
        if isinstance(value, dict):
            nested = _numeric_deltas(value, before.get(key))
            if nested:
                result[key] = nested
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            previous = before.get(key, 0)
            if isinstance(previous, (int, float)) and not isinstance(previous, bool):
                result[key] = value - previous
    return result


@contextmanager
def _proposal_stop_context(proposer, should_stop):
    """Temporarily compose supported stop callbacks with the active basin limit.

    Custom synchronous proposals without a cooperative callback finish their
    current call. No thread is killed and all original callbacks are restored.
    """
    pending, seen, changed = [proposer], set(), []
    try:
        while pending and len(seen) < 8:
            component = pending.pop()
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            for name in ("provider", "allowed_replacements"):
                nested = getattr(component, name, None)
                if nested is not None:
                    pending.append(nested)
            if not hasattr(component, "should_stop"):
                continue
            original = component.should_stop
            if original is not None and not callable(original):
                continue
            combined = lambda previous=original: should_stop() or (previous is not None and previous())
            try:
                component.should_stop = combined
            except (AttributeError, TypeError):
                continue
            changed.append((component, original))
        yield len(changed)
    finally:
        for component, original in reversed(changed):
            component.should_stop = original


class _DeadlineSignal:
    """Native stop signal; deadline expiry does not mutate the user's event."""

    def __init__(self, user_event, verifier, deadline=None):
        self.user_event = user_event
        self.verifier = verifier
        self.deadline = deadline

    def is_set(self):
        return (self.user_event.is_set() or self.verifier.budget_reason() == "time_budget"
                or self.deadline is not None and self.verifier.clock() >= self.deadline)


class CanonicalRefinementSearch:
    def __init__(self, runtime, reward=None, guidance_config=None, edit_proposer=None,
                 config: RefinementConfig | None = None, *, quality_callback=None,
                 late_quality=False, combination_planner=None, prompt_policy=None) -> None:
        self.runtime = runtime
        self.reward = reward
        self.guidance_config = guidance_config or GuidanceConfig()
        self.edit_proposer = edit_proposer
        self.config = config or RefinementConfig()
        self.quality_callback = quality_callback
        self.late_quality = late_quality
        self.combination_planner = combination_planner
        self.prompt_policy = prompt_policy

    def run(self, bundle, target, *, max_candidates=1, seed=42, on_candidate=None,
            cancel_event=None, request=None, guided=True, initial_text=None,
            on_draft=None) -> DiffusionResult:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 64:
            raise ValueError("max_candidates must be between 1 and 64")
        cancel_event = cancel_event if cancel_event is not None else threading.Event()
        verifier = CanonicalVerifier(bundle, target, self.config,
                                     quality_callback=self.quality_callback)
        native_stop = _DeadlineSignal(cancel_event, verifier)
        target = verifier.target
        started = time.monotonic()
        if request is None:
            prompt = technical_prompt(bundle.nearest(target, k=getattr(self.prompt_policy, "neighbors", 4)),
                                      self.prompt_policy) if self.runtime is not None else ""
            request = DiffusionRequest(prompt=prompt, seed=seed, max_tokens=192)
            prompt_policy = "nearest-four-context-v1" if self.runtime is not None else "no_generator"
        else:
            seed = request.seed
            prompt_policy = "supplied_request"
        best = None
        active_best = None
        beam = []
        history = []
        native_traces = []
        costs = {"proposer_seconds": 0.0, "native_seconds": 0.0,
                 "verification_seconds": 0.0, "quality_seconds": 0.0,
                 "combination_seconds": 0.0, "repair_calls": 0,
                 "cooperative_proposal_calls": 0, "noncooperative_proposal_calls": 0}
        errors = []
        epsilon_confirmed = False
        current_seed = seed
        rounds = 0
        stall = 0
        stop_reason = None
        can_generate = self.runtime is not None and callable(getattr(self.runtime, "generate", None))
        pending_initial_seeds = list(range(seed, seed + max_candidates)) if initial_text is None and can_generate else []
        initial_candidates_attempted = 0
        initial_text_pending = initial_text is not None
        restart_count = 0
        next_fresh_seed = seed + max_candidates
        basins = []
        restarts = []
        basin = None
        proposer = self.edit_proposer
        policy_before = _proposal_policy(proposer)

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
            nonlocal best, active_best, beam, epsilon_confirmed, stop_reason
            proposals = list(proposals)
            before = time.monotonic()
            verified = verifier.verify_batch(proposals, check_quality=not self.late_quality)
            costs["verification_seconds"] += time.monotonic() - before
            reactivated = set()
            if phase in ("coarse", "scheduled_coarse", "restart_coarse"):
                # Dedup protects the query budget, but must not prevent a fresh
                # seed from repairing an identical, already-scored paragraph.
                for proposal in proposals:
                    text = proposal.text if isinstance(proposal, TextProposal) else proposal
                    if not isinstance(text, str):
                        continue
                    text = text.strip()
                    cached = verifier.cache.get(" ".join(text.split()))
                    if (cached is None or cached.text != text or text in reactivated
                            or any(value.text == text for value in beam)
                            or any(value.text == text for value in verified)):
                        continue
                    if self.quality_callback is not None and not self.late_quality:
                        # verify_batch performed the current early quality check
                        # before reporting this as a duplicate. Never bypass a
                        # rejection with the earlier cached verdict.
                        if not verifier.quality_results.get(text, {}).get("accepted", False):
                            continue
                    verified.append(replace(cached, diagnostics={**cached.diagnostics,
                        "canonical_cache_reactivated": True, "reactivated_for_phase": phase,
                        "reactivation_quality_verdict": verifier.quality_results.get(text)
                        if self.quality_callback is not None and not self.late_quality else None}))
                    reactivated.add(text)
            for value in sorted(verified, key=lambda item: item.distance):
                if active_best is not None and phase not in ("coarse", "scheduled_coarse", "restart_coarse", "initial_text"):
                    source_first = next((char for char in active_best.text if char.isalpha()), "")
                    candidate_first = next((char for char in value.text if char.isalpha()), "")
                    if source_first.isupper() and candidate_first.islower():
                        verifier._reject("sentence_case_regression")
                        continue
                if self.late_quality and self.quality_callback is not None:
                    # Exact coordinates may rank all atomic edits, but only quality
                    # checked text can enter the beam or become the incumbent.
                    if len(beam) >= self.config.beam_width and value.distance >= max(item.distance for item in beam):
                        continue
                    if native_stop.is_set():
                        halted()
                        break
                    before = time.monotonic()
                    checked = verifier.verify_quality(value)
                    costs["quality_seconds"] += time.monotonic() - before
                    if checked is None or native_stop.is_set():
                        continue
                    value = checked
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
                if value.text in reactivated:
                    verifier.counts["basin_cache_reactivations"] += 1
                beam = sorted({item.text: item for item in beam}.values(),
                              key=lambda item: item.distance)[:self.config.beam_width]
                if active_best is None or value.distance < active_best.distance:
                    active_best = value
                    if basin is not None:
                        basin["active_improvements"].append({"phase": phase, "round": rounds,
                            "distance": value.distance, "elapsed_seconds": time.monotonic() - started})
                        basin["best_candidate"] = {"text": value.text, "x": value.x, "y": value.y,
                                                   "distance": value.distance, "seed": current_seed}
                if best is None or value.distance < best.distance:
                    previous = best.distance if best is not None else None
                    best = value
                    item = {"phase": phase, "round": rounds, "seed": current_seed,
                            "basin_id": basin["id"] if basin else None,
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
            return verified

        def observer_for(phase, operation_stop):
            def observe(draft):
                if on_draft is not None:
                    on_draft({"text": draft.text, "step": draft.step,
                              "num_steps": draft.num_steps, "phase": phase,
                              "basin_id": basin["id"] if basin else None,
                              "terminal": draft.terminal})
                if halted() or operation_stop.is_set():
                    return True
                if on_draft and draft.step != 0 and not draft.terminal and (draft.step + 1) % self.config.observe_every:
                    return False
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

        def run_native(phase, native_request, *, text=None, span=None, deadline=None):
            if text is not None and span is not None and self.config.repair_span_mode == "sentence":
                from .sentence_edits import protected_repair_prompt

                native_request = replace(native_request, prompt=protected_repair_prompt(
                    text, span, self.prompt_policy))
            hook = make_hook()
            before = time.monotonic()
            deadlines = [value for value in (native_stop.deadline, deadline) if value is not None]
            operation_stop = _DeadlineSignal(cancel_event, verifier, min(deadlines) if deadlines else None)
            kwargs = dict(hook=hook, observer=observer_for(phase, operation_stop),
                          observe_every=1 if on_draft else self.config.observe_every,
                          cancel_event=operation_stop)
            try:
                if text is None:
                    result = self.runtime.generate(native_request, **kwargs)
                else:
                    costs["repair_calls"] += 1
                    result = self.runtime.refine_text(native_request, text,
                                                      edit_span=span, **kwargs)
                native_traces.append({"phase": phase, "request": asdict(native_request),
                                      "basin_id": basin["id"] if basin else None,
                                      "operation_time_slice_expired": operation_stop.is_set() and not halted(),
                                      "edit_span": span, "runtime": result.trace,
                                      "guidance": hook.trace if hook else None})
                if not cancel_event.is_set() and not epsilon_confirmed:
                    consider([TextProposal(result.text, edit_span=span,
                                           provenance={"phase": phase})], phase)
            except Exception as exc:
                native_traces.append({"phase": phase, "request": asdict(native_request),
                                      "basin_id": basin["id"] if basin else None,
                                      "edit_span": span, "error": str(exc),
                                      "guidance": hook.trace if hook else None})
                raise
            finally:
                costs["native_seconds"] += time.monotonic() - before

        last_basin_reason = None
        try:
            while not halted():
                seeds = []
                if initial_text_pending:
                    kind, phase = "initial_text", "initial_text"
                    initial_text_pending = False
                elif pending_initial_seeds:
                    kind = "initial_candidates"
                    phase = "coarse" if not basins else "scheduled_coarse"
                    count = self.config.initial_candidates_before_refinement
                    seeds, pending_initial_seeds = pending_initial_seeds[:count], pending_initial_seeds[count:]
                elif can_generate and restart_count < self.config.max_restarts and rounds < self.config.max_rounds:
                    kind, phase = "restart", "restart_coarse"
                    restart_count += 1
                    seeds = list(range(next_fresh_seed, next_fresh_seed + self.config.restart_candidates))
                    next_fresh_seed += len(seeds)
                else:
                    break

                # Local exploration must not snap back to the global winner from
                # an earlier basin. All canonical/quality budgets remain shared.
                active_best, beam, stall = None, [], 0
                remaining_rounds = max(0, self.config.max_rounds - rounds)
                future_cohorts = math.ceil(len(pending_initial_seeds) / self.config.initial_candidates_before_refinement)
                future_restarts = max(0, self.config.max_restarts - restart_count) if can_generate and remaining_rounds else 0
                planned_basins = 1 + future_cohorts + future_restarts
                round_slice = max(1, remaining_rounds // planned_basins) if remaining_rounds else 0
                time_slice = None
                basin_deadline = None
                now = verifier.clock()
                if self.config.max_seconds is not None:
                    remaining_seconds = max(0.0, self.config.max_seconds - (now - verifier.started))
                    warmup_weight = 2 if not basins else 1
                    time_slice = remaining_seconds * warmup_weight / (planned_basins - 1 + warmup_weight)
                    basin_deadline = now + time_slice
                native_stop.deadline = basin_deadline
                generation_deadline = (now + time_slice * self.config.basin_generation_fraction
                                       if time_slice is not None else None)
                basin = {
                    "id": len(basins), "kind": kind, "phase": phase,
                    "trigger": last_basin_reason or ("initial_text" if kind == "initial_text" else "initial_generation"),
                    "restart_index": restart_count if kind == "restart" else None,
                    "requested_seeds": list(seeds), "attempted_seeds": [],
                    "started_seconds": time.monotonic() - started,
                    "time_slice_seconds": time_slice, "round_slice": round_slice,
                    "global_best_before": best.distance if best else None,
                    "active_improvements": [], "best_candidate": None,
                    "hard_queries_before": verifier.counts["hard_queries"],
                }
                basins.append(basin)
                if kind == "restart":
                    restarts.append(basin)
                if kind == "initial_text":
                    consider([TextProposal(initial_text, provenance={"phase": phase})], phase)
                else:
                    for index, candidate_seed in enumerate(seeds):
                        if halted() or generation_deadline is not None and verifier.clock() >= generation_deadline:
                            if kind == "initial_candidates":
                                pending_initial_seeds = seeds[index:] + pending_initial_seeds
                            break
                        current_seed = candidate_seed
                        basin["attempted_seeds"].append(current_seed)
                        if kind == "initial_candidates":
                            initial_candidates_attempted += 1
                        run_native(phase, replace(request, seed=current_seed), deadline=generation_deadline)

                def basin_expired():
                    return native_stop.deadline is not None and verifier.clock() >= native_stop.deadline

                basin_rounds = 0
                last_basin_reason = "no_new_verified_text" if active_best is None else None
                if not halted() and active_best is not None and not basin_expired() and round_slice:
                    if proposer is None:
                        try:
                            from .discrete import CanonicalEditProposer
                            proposer = CanonicalEditProposer(bundle, reward=self.reward)
                            policy_before = _proposal_policy(proposer)
                        except ImportError:
                            proposer = None
                while (not halted() and active_best is not None and not basin_expired()
                       and basin_rounds < round_slice and rounds < self.config.max_rounds):
                    round_index = basin_rounds
                    rounds += 1
                    basin_rounds += 1
                    old_distance = active_best.distance
                    proposals = []
                    source_groups = []
                    if proposer is not None:
                        before = time.monotonic()
                        sources = list(beam) or [active_best]
                        remaining = self.config.proposals_per_round
                        for index, source in enumerate(sources):
                            if halted() or basin_expired() or remaining <= 0:
                                break
                            allowance = max(1, remaining // (len(sources) - index))
                            with _proposal_stop_context(proposer, native_stop.is_set) as bindings:
                                key = "cooperative_proposal_calls" if bindings else "noncooperative_proposal_calls"
                                costs[key] += 1
                                proposed = list(proposer.propose(source.text, target,
                                                                max_proposals=allowance))[:allowance]
                            proposals.extend(proposed)
                            source_groups.append((source, proposed))
                            remaining -= len(proposed)
                        costs["proposer_seconds"] += time.monotonic() - before
                    if proposals and not halted() and not basin_expired():
                        measured = consider(proposals, "discrete_edit")
                        if self.combination_planner is not None and not halted() and not basin_expired():
                            before = time.monotonic()
                            combinations = []
                            for source, atomic in source_groups:
                                self.combination_planner.record(source, atomic, measured)
                                combinations.extend(self.combination_planner.propose(
                                    source, target,
                                    max_proposals=max(1, self.config.proposals_per_round // len(source_groups))))
                            costs["combination_seconds"] += time.monotonic() - before
                            if combinations and not halted() and not basin_expired():
                                consider(combinations, "measured_combination")
                    meaningful = old_distance - active_best.distance > self.config.min_improvement
                    stalled_round = not meaningful
                    if (not halted() and not basin_expired() and self.runtime is not None
                            and hasattr(self.runtime, "refine_text")
                            and (stalled_round or basin_rounds % self.config.repair_every == 0)):
                        with _proposal_stop_context(proposer, native_stop.is_set):
                            span = self._repair_span(active_best.text, target, proposer, round_index,
                                                     mode=self.config.repair_span_mode)
                        if halted() or basin_expired():
                            break
                        if span is not None:
                            current_seed = next_fresh_seed
                            next_fresh_seed += 1
                            repair_request = replace(request, seed=current_seed,
                                                     steps=self.config.repair_steps)
                            run_native("native_repair", repair_request, text=active_best.text, span=span)
                    meaningful = old_distance - active_best.distance > self.config.min_improvement
                    stall = 0 if meaningful else stall + 1
                    if stall >= self.config.stall_rounds:
                        last_basin_reason = "stalled"
                        break

                if halted():
                    last_basin_reason = stop_reason
                elif basin_expired():
                    last_basin_reason = "basin_time_slice"
                elif last_basin_reason != "stalled":
                    if rounds >= self.config.max_rounds:
                        last_basin_reason = "round_limit"
                    elif basin_rounds >= round_slice and active_best is not None:
                        last_basin_reason = "basin_round_slice"
                basin.update(ended_seconds=time.monotonic() - started, stop_reason=last_basin_reason,
                             rounds=basin_rounds, stall_rounds=stall,
                             global_best_after=best.distance if best else None,
                             hard_queries_after=verifier.counts["hard_queries"])
                # A fresh basin is considered only after this one is finished;
                # never clear global history or extend the shared verifier clock.
                native_stop.deadline = None
            if stop_reason is None:
                halted()
            if stop_reason is None:
                if best is None:
                    stop_reason = "no_valid_text"
                elif rounds >= self.config.max_rounds:
                    stop_reason = "budget_exhausted"
                elif can_generate and self.config.max_restarts and restart_count >= self.config.max_restarts:
                    stop_reason = "restarts_exhausted"
                else:
                    stop_reason = "stalled"
        except Exception as exc:
            errors.append(str(exc))
            stop_reason = "cancelled" if cancel_event.is_set() else "error"
            if basin is not None and "ended_seconds" not in basin:
                basin.update(ended_seconds=time.monotonic() - started, stop_reason=stop_reason,
                             error=str(exc), global_best_after=best.distance if best else None,
                             hard_queries_after=verifier.counts["hard_queries"])

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
        policy_after = _proposal_policy(proposer)
        costs["proposal_policy_numeric_deltas"] = _numeric_deltas(policy_after, policy_before)
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
            "prompt_policy_identity": getattr(self.prompt_policy, "identity", None),
            "runtime_identity": self.runtime.identity if self.runtime is not None else None,
            "elapsed_seconds": time.monotonic() - started, "rounds": rounds,
            "basins": basins, "restarts": restarts, "restart_count": restart_count,
            "initial_candidates_attempted": initial_candidates_attempted,
            "initial_candidates_pending": len(pending_initial_seeds),
            "initial_candidate_seed_schedule": list(range(seed, seed + max_candidates)) if initial_text is None and can_generate else [],
            "restart_candidates_attempted": sum(len(item["attempted_seeds"]) for item in restarts),
            "restart_scheduling_policy": "interleaved-initial-cohorts-adaptive-basin-slices-v1",
            "proposal_stop_policy": "temporarily-compose-proposer-and-provider-should_stop; restore-after-call",
            "deadline_limitations": "Running native/encoder operations and custom noncooperative callbacks may overrun a slice; no forced thread interruption",
            "stall_rounds": stall, "incumbent_improvements": len(history),
            "hard_queries": verifier.counts["hard_queries"],
            "accepted_proposals": verifier.counts["accepted"],
            "rejected_proposals": verifier.counts["rejected"],
            "budget_kind": verifier.budget_reason() or
                           ("round_limit" if stop_reason == "budget_exhausted" else None),
            "candidates": candidates, "best_candidate": best_payload,
            "history": history, "native_traces": native_traces, "costs": costs,
            "executed_work": native_work_summary(native_traces),
            "proposal_policy": policy_after, "proposal_policy_before": policy_before,
            "measured_combinations": self.combination_planner.cache_info() if self.combination_planner else None,
            "quality_policy": _json_safe(getattr(self.quality_callback, "trace", None)),
            "late_quality": self.late_quality,
            "verification": verifier.diagnostics,
            "quality_validation_state": verifier.diagnostics["quality_validation_state"],
            "epsilon_is_attainability_claim": False,
        }
        return DiffusionResult(best.text if best else "", token_ids, _json_safe(trace))

    @staticmethod
    def _repair_span(text, target, proposer, round_index, mode="words"):
        words = list(re.finditer(r"\b\w+(?:[-'’]\w+)*\b", text))
        if not words:
            return None
        spans = []
        if proposer is not None and hasattr(proposer, "hotspots"):
            spans = list(proposer.hotspots(text, target, limit=4))
        if mode == "sentence":
            from .sentence_edits import sentence_spans

            sentences = list(sentence_spans(text))
            preferred = []
            for item in spans:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    continue
                start, end = item
                if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
                    continue
                for sentence in sentences:
                    if sentence[0] < end and sentence[1] > start and sentence not in preferred:
                        preferred.append(sentence)
            ordered = preferred + [sentence for sentence in sentences if sentence not in preferred]
            return ordered[round_index % len(ordered)] if ordered else None
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
