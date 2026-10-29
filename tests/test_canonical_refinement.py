from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import numpy as np

from point2text.contracts import CorpusRecord
from point2text.diffusion.contracts import DiffusionDraft, DiffusionResult
from point2text.diffusion.refinement import CanonicalRefinementSearch
from point2text.diffusion.refinement_contracts import RefinementConfig, TextProposal
from point2text.diffusion.verification import CanonicalVerifier


class Bundle:
    map_id = "controller-fixture-map"

    def __init__(self):
        self.encoder = self
        self.calls = []
        self.model = SimpleNamespace(tokenizer=self.tokenize)

    @staticmethod
    def tokenize(texts, **kwargs):
        return {"input_ids": [[ord(c) for c in text] for text in texts]}

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([[float(text.split()[-1]), 0] for text in texts])

    def project(self, values):
        return values

    def nearest(self, target, k):
        return [CorpusRecord("1", "topic", "Context", "Independent context paragraph.")]


def config(**kwargs):
    return replace(RefinementConfig(min_words=2, max_words=10, max_rounds=6,
                                   stall_rounds=2, proposals_per_round=4), **kwargs)


class Proposer:
    def __init__(self):
        self.calls = []

    def propose(self, text, target, *, max_proposals):
        self.calls.append(text)
        value = float(text.split()[-1])
        return [TextProposal(f"Paragraph {value / 10}", provenance={"kind": "fixture"}),
                TextProposal("## invalid special heading")][:max_proposals]

    def hotspots(self, text, target, *, limit):
        return [(0, len(text))]


def test_cpu_refinement_has_monotone_callbacks_and_confirmed_epsilon():
    bundle = Bundle()
    published = []
    result = CanonicalRefinementSearch(None, edit_proposer=Proposer(), config=config()).run(
        bundle, (0, 0), initial_text="Paragraph 0.1", on_candidate=published.append)
    distances = [candidate.distance for candidate in published]
    assert all(after < before for before, after in zip(distances, distances[1:]))
    assert result.trace["epsilon_confirmed"] is True
    assert result.trace["stop_reason"] == "epsilon_reached"
    assert result.trace["epsilon_worst_distance"] <= 1e-4
    assert bundle.calls[-2:] == [[result.text], [result.text]]
    assert result.trace["quality_validation_state"] == "mechanical_checks_only"
    assert result.trace["best_candidate"]["text"] == result.text
    json.dumps(result.trace, allow_nan=False)


def test_observed_clean_draft_can_win_even_when_final_text_is_worse():
    class Runtime:
        identity = "fixture-native"

        def generate(self, request, *, observer, **kwargs):
            assert observer(DiffusionDraft("Paragraph 0.02", [], [], 0, 32)) is False
            return DiffusionResult("Paragraph 0.2", [], {"passes": 1})

    result = CanonicalRefinementSearch(Runtime(), edit_proposer=Proposer(),
                                        config=config(max_rounds=1)).run(
        Bundle(), (0, 0), guided=False)
    assert result.trace["history"][0]["text"] == "Paragraph 0.02"
    assert result.trace["best_candidate"]["distance"] < 0.02
    assert result.trace["native_traces"][0]["phase"] == "coarse"


def test_stalled_search_calls_local_repair_with_one_to_four_word_span():
    class NoEdits(Proposer):
        def propose(self, *args, **kwargs):
            return []

    class Runtime:
        identity = "repair-fixture"

        def __init__(self):
            self.calls = []

        def refine_text(self, request, text, *, edit_span, observer, **kwargs):
            self.calls.append((request, text, edit_span))
            return DiffusionResult("Paragraph 0.01", [], {"repair": True})

    runtime = Runtime()
    result = CanonicalRefinementSearch(runtime, edit_proposer=NoEdits(),
                                        config=config(max_rounds=1)).run(
        Bundle(), (0, 0), initial_text="Paragraph 0.1", guided=False)
    assert len(runtime.calls) == 1
    request, text, span = runtime.calls[0]
    assert request.steps == 8
    assert 1 <= len(text[span[0]:span[1]].split()) <= 4
    assert result.trace["best_candidate"]["distance"] == 0.01


def test_cancellation_and_later_errors_keep_the_verified_incumbent():
    event = threading.Event()
    result = CanonicalRefinementSearch(None, edit_proposer=Proposer(), config=config()).run(
        Bundle(), (0, 0), initial_text="Paragraph 0.1", cancel_event=event,
        on_candidate=lambda value: event.set())
    assert result.text == "Paragraph 0.1"
    assert result.trace["status"] == "cancelled"

    class BrokenProposer(Proposer):
        def propose(self, *args, **kwargs):
            raise RuntimeError("proposal backend failed")

    result = CanonicalRefinementSearch(None, edit_proposer=BrokenProposer(), config=config()).run(
        Bundle(), (0, 0), initial_text="Paragraph 0.1")
    assert result.text == "Paragraph 0.1"
    assert result.trace["status"] == "error"
    assert "proposal backend failed" in result.trace["error"]


def test_budget_exhaustion_never_claims_unreachable_or_epsilon_success():
    result = CanonicalRefinementSearch(None, edit_proposer=Proposer(),
                                        config=config(max_hard_queries=3)).run(
        Bundle(), (0, 0), initial_text="Paragraph 0.1")
    assert result.trace["stop_reason"] == "budget_exhausted"
    assert result.trace["epsilon_confirmed"] is False
    assert result.trace["verification"]["hard_queries"] == 1
    assert result.text == "Paragraph 0.1"


def test_invalid_initial_text_is_reported_without_crashing():
    result = CanonicalRefinementSearch(None, edit_proposer=Proposer(), config=config()).run(
        Bundle(), (0, 0), initial_text="## invalid")
    assert result.text == ""
    assert result.trace["status"] == "error"
    assert result.trace["stop_reason"] == "no_valid_text"


def test_bad_fresh_confirmation_does_not_publish_false_success():
    class NoEdits(Proposer):
        def propose(self, *args, **kwargs):
            return []

    bundle = Bundle()
    original = bundle.encode

    def noisy_encode(texts):
        values = original(texts)
        if len(bundle.calls) == 3:
            values[0, 0] = 0.01
        return values

    bundle.encode = noisy_encode
    result = CanonicalRefinementSearch(None, edit_proposer=NoEdits(), config=config()).run(
        bundle, (0, 0), initial_text="Paragraph 0.00001")
    assert result.trace["epsilon_confirmed"] is False
    assert result.trace["best_candidate"]["distance"] == 0.01
    assert result.trace["epsilon_worst_distance"] == 0.01


def test_stall_and_pre_cancel_are_not_epsilon_claims():
    class NoEdits(Proposer):
        def propose(self, *args, **kwargs):
            return []

    result = CanonicalRefinementSearch(None, edit_proposer=NoEdits(), config=config()).run(
        Bundle(), (0, 0), initial_text="Paragraph 0.1")
    assert result.trace["stop_reason"] == "stalled"
    assert result.trace["rounds"] == 2
    assert result.trace["epsilon_confirmed"] is False
    event = threading.Event()
    event.set()
    bundle = Bundle()
    result = CanonicalRefinementSearch(None, config=config()).run(
        bundle, (0, 0), initial_text="Paragraph 0.1", cancel_event=event)
    assert result.trace["status"] == "cancelled"
    assert bundle.calls == []


def test_failure_after_observed_draft_retains_it_and_native_error_trace():
    class Runtime:
        identity = "failed-native-fixture"

        def generate(self, request, *, observer, **kwargs):
            observer(DiffusionDraft("Paragraph 0.1", [], [], 0, 32))
            raise RuntimeError("native pass failed")

    result = CanonicalRefinementSearch(Runtime(), config=config()).run(
        Bundle(), (0, 0), guided=False)
    assert result.text == "Paragraph 0.1"
    assert result.trace["status"] == "error"
    assert result.trace["native_traces"][0]["error"] == "native pass failed"


def test_hotspot_analysis_expiring_deadline_does_not_start_repair(monkeypatch):
    now = [0.0]

    def verifier_factory(*args, **kwargs):
        return CanonicalVerifier(*args, **kwargs, clock=lambda: now[0])

    monkeypatch.setattr("point2text.diffusion.refinement.CanonicalVerifier", verifier_factory)

    class SlowHotspots(Proposer):
        def propose(self, *args, **kwargs):
            return []

        def hotspots(self, text, target, *, limit):
            now[0] = 2.0
            return [(0, len(text))]

    class Runtime:
        identity = "must-not-repair-after-deadline"

        def refine_text(self, *args, **kwargs):
            raise AssertionError("Expired hotspot analysis must not start native repair")

    result = CanonicalRefinementSearch(Runtime(), edit_proposer=SlowHotspots(),
                                        config=config(max_seconds=1)).run(
        Bundle(), (0, 0), initial_text="Paragraph 0.1", guided=False)
    assert result.text == "Paragraph 0.1"
    assert result.trace["status"] == "complete"
    assert result.trace["stop_reason"] == "budget_exhausted"
    assert result.trace["budget_kind"] == "time_budget"
    assert result.trace["costs"]["repair_calls"] == 0


def test_native_deadline_signal_retains_draft_and_is_not_user_cancellation(monkeypatch):
    now = [0.0]
    user_cancel = threading.Event()

    def verifier_factory(*args, **kwargs):
        return CanonicalVerifier(*args, **kwargs, clock=lambda: now[0])

    monkeypatch.setattr("point2text.diffusion.refinement.CanonicalVerifier", verifier_factory)

    class Runtime:
        identity = "deadline-aware-native-fixture"

        def generate(self, request, *, observer, cancel_event, **kwargs):
            assert not cancel_event.is_set()
            observer(DiffusionDraft("Paragraph 0.1", [], [], 0, 32))
            now[0] = 2.0
            assert cancel_event.is_set()
            return DiffusionResult("", [], {"cancelled": True, "passes": 1})

    result = CanonicalRefinementSearch(Runtime(), edit_proposer=Proposer(),
                                        config=config(max_seconds=1)).run(
        Bundle(), (0, 0), guided=False, cancel_event=user_cancel)
    assert result.text == "Paragraph 0.1"
    assert result.trace["status"] == "complete"
    assert result.trace["stop_reason"] == "budget_exhausted"
    assert result.trace["budget_kind"] == "time_budget"
    assert not user_cancel.is_set()
    assert result.trace["native_traces"][0]["runtime"]["cancelled"] is True


def test_contextual_replacement_policy_records_per_run_work_separately():
    class Provider:
        calls = 5  # Simulate a direct API caller reusing an existing provider.

        @property
        def trace(self):
            return {"native_forwards": self.calls, "forward_seconds": self.calls * 0.5,
                    "quality_guarantee": False}

    class ContextProposer(Proposer):
        def __init__(self):
            super().__init__()
            self.allowed_replacements = Provider()

        def propose(self, *args, **kwargs):
            self.allowed_replacements.calls += 1
            return []

    result = CanonicalRefinementSearch(None, edit_proposer=ContextProposer(), config=config()).run(
        Bundle(), (0, 0), initial_text="Paragraph 0.1")
    assert result.trace["proposal_policy"]["native_forwards"] == 7
    assert result.trace["proposal_policy"]["quality_guarantee"] is False
    assert result.trace["proposal_policy_before"]["native_forwards"] == 5
    assert result.trace["costs"]["proposal_policy_numeric_deltas"] == {
        "native_forwards": 2, "forward_seconds": 1.0}
    assert result.trace["hard_queries"] == 1
