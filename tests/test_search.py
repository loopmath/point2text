import threading

import pytest

from point2text.contracts import CorpusRecord
from point2text.search import BaselineSearch


class FakeBundle:
    def __init__(self):
        self.examples = [CorpusRecord("1", "topic", "Title", "An example.")]
        self.scored = []

    def nearest(self, target, k):
        assert target == (0.0, 0.0)
        assert k == 4
        return self.examples

    def score(self, text, target):
        self.scored.append(text)
        distance = {"First paragraph.": 2.0, "Better paragraph.": 0.2}[text]
        return (distance, 0.0), distance


class FakeGenerator:
    identity = "fake"

    def __init__(self, texts):
        self.texts = iter(texts)
        self.calls = []

    def generate(self, examples, *, seed, cancel_event):
        self.calls.append((examples, seed))
        return next(self.texts)


def test_search_hard_scores_text_and_retains_all_results():
    bundle = FakeBundle()
    generator = FakeGenerator(["First paragraph.", "Better paragraph."])
    emitted = []
    candidates = BaselineSearch(generator).run(
        bundle, (0.0, 0.0), max_candidates=2, seed=100,
        on_candidate=emitted.append, cancel_event=threading.Event())
    assert candidates == emitted
    assert bundle.scored == ["First paragraph.", "Better paragraph."]
    assert [item.seed for item in candidates] == [100, 101]
    assert min(candidates, key=lambda item: item.distance).text == "Better paragraph."
    assert all(item.x == item.distance and item.y == 0.0 for item in candidates)
    assert candidates[1].elapsed_seconds >= candidates[0].elapsed_seconds >= 0
    assert generator.calls == [(bundle.examples, 100), (bundle.examples, 101)]


def test_search_skips_empty_and_duplicate_outputs_with_a_fixed_attempt_budget():
    bundle = FakeBundle()
    generator = FakeGenerator(["  ", "First paragraph.", " first  paragraph. "])
    candidates = BaselineSearch(generator).run(
        bundle, (0.0, 0.0), max_candidates=3, seed=7,
        on_candidate=lambda item: None, cancel_event=threading.Event())
    assert len(candidates) == 1
    assert bundle.scored == ["First paragraph."]
    assert len(generator.calls) == 3
    assert candidates[0].seed == 8


def test_cancellation_keeps_previously_verified_candidates():
    event = threading.Event()
    bundle = FakeBundle()
    generator = FakeGenerator(["First paragraph.", "Better paragraph."])
    candidates = BaselineSearch(generator).run(
        bundle, (0.0, 0.0), max_candidates=2, seed=0,
        on_candidate=lambda item: event.set(), cancel_event=event)
    assert len(candidates) == 1
    assert len(generator.calls) == 1


def test_generation_cancelled_midrequest_is_not_scored():
    event = threading.Event()
    bundle = FakeBundle()

    class CancellingGenerator:
        def generate(self, examples, *, seed, cancel_event):
            cancel_event.set()
            return "First paragraph."

    candidates = BaselineSearch(CancellingGenerator()).run(
        bundle, (0.0, 0.0), max_candidates=2, seed=0,
        on_candidate=lambda item: pytest.fail("Cancelled text must not be published"),
        cancel_event=event)
    assert candidates == []
    assert bundle.scored == []


def test_no_usable_output_is_actionable():
    with pytest.raises(RuntimeError, match="no usable paragraph"):
        BaselineSearch(FakeGenerator([""])).run(
            FakeBundle(), (0.0, 0.0), max_candidates=1, seed=0,
            on_candidate=lambda item: None, cancel_event=threading.Event())


def test_nonfinite_hard_score_is_not_published():
    class BrokenBundle(FakeBundle):
        def score(self, text, target):
            return (float("nan"), 0.0), 1.0

    with pytest.raises(RuntimeError, match="non-finite"):
        BaselineSearch(FakeGenerator(["First paragraph."])).run(
            BrokenBundle(), (0.0, 0.0), max_candidates=1, seed=0,
            on_candidate=lambda item: pytest.fail("Invalid scores must not be published"),
            cancel_event=threading.Event())
