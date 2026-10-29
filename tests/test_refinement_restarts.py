"""Restart scheduling and quality invariants, with no model or GPU calls."""

from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from point2text.contracts import CorpusRecord
from point2text.diffusion.contracts import DiffusionDraft, DiffusionRequest, DiffusionResult
from point2text.diffusion.refinement import CanonicalRefinementSearch
from point2text.diffusion.refinement_contracts import RefinementConfig, TextProposal
from point2text.diffusion.verification import CanonicalVerifier


class Bundle:
    map_id = "restart-fixture-map"

    def __init__(self):
        self.encoder = self
        self.calls = []
        self.model = SimpleNamespace(tokenizer=lambda texts, **kwargs: {
            "input_ids": [[ord(char) for char in text] for text in texts]})

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([[float(text.split()[-1].rstrip(".!?")), 0.0] for text in texts])

    def project(self, values):
        return values

    def nearest(self, target, k):
        return [CorpusRecord("1", "fixture", "Context", "Independent context paragraph.")]


def settings(**kwargs):
    return replace(RefinementConfig(min_words=2, max_words=12, max_seconds=None,
                                   beam_width=1, stall_rounds=1, max_rounds=20,
                                   proposals_per_round=4, restart_candidates=1,
                                   initial_candidates_before_refinement=1), **kwargs)


class NoEdits:
    def __init__(self):
        self.sources = []

    def propose(self, text, target, **kwargs):
        self.sources.append(text)
        return []

    def hotspots(self, text, target, **kwargs):
        return [(0, len(text))]


class Runtime:
    identity = "fake-native-restarts"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.seeds = []

    def generate(self, request, **kwargs):
        self.seeds.append(request.seed)
        return DiffusionResult(next(self.outputs), [], {"passes": 1})


def test_stall_restarts_fresh_native_generation_and_publishes_only_improvements():
    runtime = Runtime(["Initial basin 0.3.", "Fresh basin 0.1."])
    published, drafts = [], []
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_restarts=1)).run(Bundle(), (0, 0), guided=False,
                                            on_candidate=published.append, on_draft=drafts.append)
    assert result.text == "Fresh basin 0.1."
    assert [value.distance for value in published] == [0.3, 0.1]
    assert runtime.seeds == [42, 43]
    assert [row["phase"] for row in result.trace["native_traces"]] == ["coarse", "restart_coarse"]
    assert result.trace["restart_count"] == 1
    assert result.trace["restarts"][0]["trigger"] == "stalled"
    assert result.trace["restarts"][0]["attempted_seeds"] == [43]
    assert result.trace["stop_reason"] == "restarts_exhausted"
    assert not result.trace["epsilon_confirmed"]
    json.dumps(result.trace, allow_nan=False)


def test_worse_fresh_basin_refines_its_own_best_and_keeps_global_winner():
    class LocalEdits(NoEdits):
        def propose(self, text, target, **kwargs):
            self.sources.append(text)
            return {"Fresh basin 0.5.": [TextProposal("Fresh basin 0.4.")],
                    "Fresh basin 0.4.": [TextProposal("Fresh basin 0.3.")]}.get(text, [])

    class RepairRuntime(Runtime):
        def __init__(self):
            super().__init__(["Initial basin 0.1.", "Fresh basin 0.5."])
            self.repairs = []

        def refine_text(self, request, text, **kwargs):
            self.repairs.append((request.seed, text))
            return DiffusionResult(text, [], {})

    proposer, runtime = LocalEdits(), RepairRuntime()
    result = CanonicalRefinementSearch(runtime, edit_proposer=proposer,
        config=settings(max_restarts=1, repair_every=99)).run(Bundle(), (0, 0), guided=False)
    assert result.text == "Initial basin 0.1."
    assert "Fresh basin 0.4." in proposer.sources and "Fresh basin 0.3." in proposer.sources
    assert [row["distance"] for row in result.trace["restarts"][0]["active_improvements"]] == [0.5, 0.4, 0.3]
    assert result.trace["restarts"][0]["best_candidate"]["distance"] == 0.3
    assert runtime.repairs[-1][1] == "Fresh basin 0.3."
    all_seeds = runtime.seeds + [seed for seed, _ in runtime.repairs]
    assert len(all_seeds) == len(set(all_seeds))


def test_no_runtime_stays_cpu_only_without_restart_attempts():
    result = CanonicalRefinementSearch(None, edit_proposer=NoEdits(),
        config=settings()).run(Bundle(), (0, 0), initial_text="CPU basin 0.2.")
    assert result.text == "CPU basin 0.2."
    assert result.trace["restart_count"] == 0 and result.trace["restarts"] == []
    assert result.trace["stop_reason"] == "stalled"


def test_exhausted_restarts_preserve_best_and_use_unique_seeds():
    runtime = Runtime(["Initial basin 0.1."] + [f"Worse basin {i / 10}." for i in range(2, 8)])
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_restarts=3, restart_candidates=2)).run(Bundle(), (0, 0), guided=False)
    assert result.text == "Initial basin 0.1."
    assert result.trace["restart_count"] == 3
    assert result.trace["restart_candidates_attempted"] == 6
    assert len(runtime.seeds) == len(set(runtime.seeds)) == 7
    assert result.trace["stop_reason"] == "restarts_exhausted"
    assert len(result.trace["history"]) == 1


@pytest.mark.parametrize("late_quality", [False, True])
def test_restarts_keep_quality_policy_even_when_rejected_text_hits_target(late_quality):
    runtime = Runtime(["Sound initial 0.2.", "Broken quality 0.0.", "Sound fresh 0.1."])
    checked = []

    def quality(text):
        checked.append(text)
        return {"accepted": not text.startswith("Broken")}

    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_restarts=1, restart_candidates=2),
        quality_callback=quality, late_quality=late_quality).run(Bundle(), (0, 0), guided=False)
    assert result.text == "Sound fresh 0.1."
    assert "Broken quality 0.0." in checked
    assert all(not item["text"].startswith("Broken") for item in result.trace["history"])
    assert not result.trace["epsilon_confirmed"]
    assert result.trace["quality_validation_state"] == "mechanical_checks_and_external_callback"


def test_invalid_initial_generation_does_not_suppress_fresh_restart():
    runtime = Runtime(["## invalid header", "Fresh valid 0.1."])
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_restarts=1)).run(Bundle(), (0, 0), guided=False)
    assert result.text == "Fresh valid 0.1."
    assert result.trace["basins"][0]["stop_reason"] == "no_new_verified_text"
    assert result.trace["status"] == "complete"


def test_cancellation_never_starts_another_basin_and_retains_best():
    runtime = Runtime(["Sound initial 0.2."])
    cancel = threading.Event()
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(), config=settings()).run(
        Bundle(), (0, 0), guided=False, cancel_event=cancel,
        on_candidate=lambda value: cancel.set())
    assert runtime.seeds == [42]
    assert result.text == "Sound initial 0.2."
    assert result.trace["stop_reason"] == result.trace["status"] == "cancelled"
    assert result.trace["restart_count"] == 0


def test_global_query_budget_and_confirmation_reserve_survive_restarts():
    runtime = Runtime(["Sound initial 0.2.", "Fresh nearby 0.00001."])
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_hard_queries=4)).run(Bundle(), (0, 0), guided=False)
    assert result.trace["hard_queries"] == 4
    assert result.trace["epsilon_confirmed"] and result.trace["stop_reason"] == "epsilon_reached"
    assert len(runtime.seeds) == 2


def test_requested_initial_six_are_interleaved_not_silently_dropped():
    timeline = []

    class LoggedRuntime(Runtime):
        def generate(self, request, **kwargs):
            timeline.append(("generate", request.seed))
            return super().generate(request, **kwargs)

    class LoggedProposer(NoEdits):
        def propose(self, text, target, **kwargs):
            timeline.append(("refine", text))
            return []

    runtime = LoggedRuntime([f"Initial candidate {i / 10}." for i in range(6, 0, -1)])
    result = CanonicalRefinementSearch(runtime, edit_proposer=LoggedProposer(),
        config=settings(max_restarts=0, initial_candidates_before_refinement=2)).run(
            Bundle(), (0, 0), guided=False, max_candidates=6)
    assert runtime.seeds == list(range(42, 48))
    assert timeline[0][0] == timeline[1][0] == "generate" and timeline[2][0] == "refine"
    assert result.trace["initial_candidates_attempted"] == 6
    assert result.trace["initial_candidates_pending"] == 0
    assert result.trace["restart_count"] == 0


def test_default_round_allocation_reaches_all_three_restarts_after_initial_six():
    runtime = Runtime([f"Candidate {index / 100}." for index in range(12, 0, -1)])
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(stall_rounds=4, max_rounds=20, max_restarts=3, restart_candidates=2,
                        initial_candidates_before_refinement=2)).run(
            Bundle(), (0, 0), guided=False, max_candidates=6)
    assert result.trace["initial_candidates_attempted"] == 6
    assert result.trace["restart_count"] == 3
    assert runtime.seeds == list(range(42, 54))
    assert result.trace["rounds"] == 20
    assert sum(basin["rounds"] for basin in result.trace["basins"]) == 20


def test_cached_identical_fresh_generation_still_gets_new_seed_repairs():
    class SameDraftRuntime(Runtime):
        def __init__(self):
            super().__init__(["Identical basin 0.2."] * 2)
            self.repairs = []

        def refine_text(self, request, text, **kwargs):
            self.repairs.append((request.seed, text))
            output = text if len(self.repairs) == 1 else "Repaired basin 0.05."
            return DiffusionResult(output, [], {})

    runtime, published = SameDraftRuntime(), []
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_restarts=1), quality_callback=lambda text: True,
        late_quality=True).run(Bundle(), (0, 0), guided=False, on_candidate=published.append)
    assert result.text == "Repaired basin 0.05."
    assert result.trace["verification"]["basin_cache_reactivations"] == 1
    assert result.trace["hard_queries"] == 2
    assert [value.distance for value in published] == [.2, .05]
    assert len(runtime.repairs) >= 2 and runtime.repairs[1][1] == "Identical basin 0.2."
    seeds = runtime.seeds + [seed for seed, _ in runtime.repairs]
    assert len(seeds) == len(set(seeds))


@pytest.mark.parametrize("late_quality", [False, True])
def test_restart_cohort_reuses_its_best_cached_text_regardless_of_generation_order(late_quality):
    best, worse = "Cached best 0.2.", "Cached worse 0.5."
    runtime, proposer = Runtime([best, worse, worse, best]), NoEdits()
    published = []
    result = CanonicalRefinementSearch(runtime, edit_proposer=proposer,
        config=settings(max_restarts=1, restart_candidates=2,
                        initial_candidates_before_refinement=2),
        quality_callback=lambda text: True, late_quality=late_quality).run(
            Bundle(), (0, 0), guided=False, max_candidates=2, on_candidate=published.append)
    assert proposer.sources == [best, best]
    assert result.trace['restarts'][0]['best_candidate']['text'] == best
    assert result.trace['hard_queries'] == 2
    assert [value.text for value in published] == [best]


@pytest.mark.parametrize("late_quality", [False, True])
def test_cache_reactivation_cannot_bypass_current_quality_rejection(late_quality):
    calls = []

    def quality(text):
        calls.append(text)
        return len(calls) == 1

    runtime = Runtime(["Identical basin 0.2."] * 2)
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_restarts=1), quality_callback=quality,
        late_quality=late_quality).run(Bundle(), (0, 0), guided=False)
    assert result.text == "Identical basin 0.2."
    assert result.trace["restarts"][0]["best_candidate"] is None
    assert result.trace["verification"].get("basin_cache_reactivations", 0) == 0
    assert len(result.trace["history"]) == 1


def test_padded_cached_coarse_text_cannot_bypass_its_raw_quality_rejection():
    source = "Quality basin 0.2."
    runtime = Runtime([source, f"  {source}  "])
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_restarts=1), quality_callback=lambda text: text == source,
        late_quality=False).run(Bundle(), (0, 0), guided=False)
    assert result.text == source
    assert result.trace['restarts'][0]['best_candidate'] is None
    assert result.trace['verification'].get('basin_cache_reactivations', 0) == 0


def test_proposal_provider_inherits_local_stop_and_original_callback_is_restored(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("point2text.diffusion.refinement.CanonicalVerifier",
        lambda *args, **kwargs: CanonicalVerifier(*args, **kwargs, clock=lambda: now[0]))

    class Provider:
        def __init__(self):
            self.should_stop = lambda: False

    class CooperativeProposer(NoEdits):
        def __init__(self):
            super().__init__()
            self.allowed_replacements = Provider()
            self.work_per_call = []

        def propose(self, *args, **kwargs):
            count = 0
            for _ in range(100):
                if self.allowed_replacements.should_stop():
                    break
                count += 1
                now[0] += 10
            self.work_per_call.append(count)
            return []

    proposer = CooperativeProposer()
    original_callback = proposer.allowed_replacements.should_stop
    runtime = Runtime(["Fresh basin 0.3."])
    result = CanonicalRefinementSearch(runtime, edit_proposer=proposer,
        config=settings(max_restarts=1, max_seconds=120)).run(
            Bundle(), (0, 0), initial_text="Initial basin 0.2.", guided=False)
    assert runtime.seeds  # First proposal yielded before consuming all120seconds.
    assert proposer.work_per_call[0] == 8
    assert result.trace["basins"][0]["stop_reason"] == "basin_time_slice"
    assert proposer.allowed_replacements.should_stop is original_callback
    assert result.trace["costs"]["cooperative_proposal_calls"] >= 1


def test_original_proposal_stop_callbacks_are_restored_on_error():
    class BrokenProposer(NoEdits):
        def __init__(self):
            self.should_stop = lambda: False
            self.provider = SimpleNamespace(should_stop=None)

        def propose(self, *args, **kwargs):
            assert callable(self.provider.should_stop)
            raise RuntimeError("fixture provider failure")

    proposer = BrokenProposer()
    original_callback = proposer.should_stop
    result = CanonicalRefinementSearch(None, edit_proposer=proposer, config=settings()).run(
        Bundle(), (0, 0), initial_text="Initial basin 0.2.")
    assert result.trace["status"] == "error"
    assert proposer.should_stop is original_callback and proposer.provider.should_stop is None


def test_local_slice_preserves_global_time_for_remaining_candidates(monkeypatch):
    now = [0.0]

    def verifier_factory(*args, **kwargs):
        return CanonicalVerifier(*args, **kwargs, clock=lambda: now[0])

    monkeypatch.setattr("point2text.diffusion.refinement.CanonicalVerifier", verifier_factory)

    class SliceRuntime:
        identity = "local-slice-fixture"

        def __init__(self):
            self.seeds = []
            self.first_deadline = None

        def generate(self, request, *, observer, cancel_event, **kwargs):
            self.seeds.append(request.seed)
            text = f"Candidate{request.seed} {1 / (request.seed - 40)}."
            observer(DiffusionDraft(text, [], [], 0, 32))
            if len(self.seeds) == 1:
                self.first_deadline = cancel_event.deadline
                now[0] = self.first_deadline + .001
                assert cancel_event.is_set() and now[0] < 120
            else:
                now[0] += 1
            return DiffusionResult(text, [], {"cancelled": cancel_event.is_set()})

    runtime, user_cancel = SliceRuntime(), threading.Event()
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_seconds=120, max_restarts=1, initial_candidates_before_refinement=2)).run(
            Bundle(), (0, 0), guided=False, max_candidates=6, cancel_event=user_cancel)
    assert runtime.seeds[:6] == list(range(42, 48))
    assert 0 < runtime.first_deadline < 120
    assert result.trace["native_traces"][0]["operation_time_slice_expired"] is True
    assert result.trace["initial_candidates_attempted"] == 6
    assert result.trace["restart_count"] == 1
    assert not user_cancel.is_set()


def test_global_deadline_stops_restarts_without_mutating_user_cancel(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("point2text.diffusion.refinement.CanonicalVerifier",
        lambda *args, **kwargs: CanonicalVerifier(*args, **kwargs, clock=lambda: now[0]))

    class DeadlineRuntime(Runtime):
        def generate(self, request, *, observer, cancel_event, **kwargs):
            self.seeds.append(request.seed)
            observer(DiffusionDraft("Sound draft 0.2.", [], [], 0, 32))
            now[0] = 2
            assert cancel_event.is_set()
            return DiffusionResult("", [], {"cancelled": True})

    runtime, user_cancel = DeadlineRuntime([]), threading.Event()
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_seconds=1)).run(Bundle(), (0, 0), guided=False, cancel_event=user_cancel)
    assert result.text == "Sound draft 0.2."
    assert runtime.seeds == [42] and not user_cancel.is_set()
    assert result.trace["stop_reason"] == "budget_exhausted"
    assert result.trace["budget_kind"] == "time_budget"


def test_new_generation_is_not_compared_to_previous_basins_sentence_case():
    runtime = Runtime(["Sound initial 0.2.", "lowercase fresh 0.1."])
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_restarts=1)).run(Bundle(), (0, 0), guided=False)
    assert result.text == "lowercase fresh 0.1."
    assert "sentence_case_regression" not in result.trace["verification"]["rejections"]


def test_sentence_span_mode_and_actual_repair_prompt_are_traced():
    source = "First sentence 0.1. Second sentence 0.2."

    class SentenceBundle(Bundle):
        def encode(self, texts):
            self.calls.append(list(texts))
            return np.asarray([[.2, 0] for text in texts])

    class SentenceRuntime:
        identity = "sentence-repair-fixture"

        def __init__(self):
            self.calls = []

        def refine_text(self, request, text, *, edit_span, **kwargs):
            self.calls.append((request, edit_span))
            return DiffusionResult(text, [], {})

    runtime = SentenceRuntime()
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
        config=settings(max_rounds=1, repair_span_mode="sentence")).run(
            SentenceBundle(), (0, 0), initial_text=source, guided=False,
            request=DiffusionRequest("Original composition prompt"))
    request, span = runtime.calls[0]
    assert source[slice(*span)] in {"First sentence 0.1.", "Second sentence 0.2."}
    assert "Revise the editable text" in request.prompt
    assert result.trace["native_traces"][0]["request"]["prompt"] == request.prompt


@pytest.mark.parametrize("kwargs", [dict(max_restarts=-1), dict(max_restarts=True),
    dict(restart_candidates=0), dict(initial_candidates_before_refinement=0),
    dict(basin_generation_fraction=0), dict(basin_generation_fraction=1.1),
    dict(repair_span_mode="fragment")])
def test_restart_config_validation(kwargs):
    with pytest.raises(ValueError):
        settings(**kwargs)
