"""Actual measurements, deferred quality, unlimited clocks and live drafts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from point2text.diffusion.combinations import MeasuredCombinationPlanner
from point2text.diffusion.contracts import DiffusionDraft, DiffusionRequest, DiffusionResult
from point2text.diffusion.refinement import CanonicalRefinementSearch
from point2text.diffusion.refinement_contracts import RefinementConfig, TextProposal
from point2text.diffusion.verification import CanonicalVerifier


class Bundle:
    map_id = "measured-quality-fixture"

    def __init__(self, points):
        self.points, self.calls = points, []
        self.encoder = self
        self.model = SimpleNamespace(tokenizer=lambda texts, **kwargs: {
            "input_ids": [[ord(char) for char in text] for text in texts]})

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([self.points[text] for text in texts])

    def project(self, values):
        return values


def settings(**kwargs):
    return replace(RefinementConfig(min_words=3, max_words=20, beam_width=1,
                                   max_rounds=2, max_hard_queries=30), **kwargs)


def test_individually_worse_edits_can_cancel_but_only_checked_combination_is_published():
    source, first, second, combined = "Alpha beta gamma.", "Alfa beta gamma.", "Alpha beta gama.", "Alfa beta gama."
    bundle = Bundle({source: (.1, .1), first: (.3, .1), second: (-.2, 0), combined: (0, 0)})
    class Proposer:
        def propose(self, text, target, **kwargs):
            return [TextProposal(first, provenance={"source_spans": [[0, 5]], "replacements": ["Alfa"]}),
                    TextProposal(second, provenance={"source_spans": [[11, 16]], "replacements": ["gama"]})]
    judgments, published = [], []
    def quality(text):
        judgments.append(text)
        return {"accepted": True}
    result = CanonicalRefinementSearch(None, edit_proposer=Proposer(), config=settings(),
        quality_callback=quality, late_quality=True,
        combination_planner=MeasuredCombinationPlanner()).run(
            bundle, (0, 0), initial_text=source, on_candidate=published.append)
    assert result.text == combined and result.trace["epsilon_confirmed"]
    assert [value.text for value in published] == [source, combined]
    assert first not in judgments and second not in judgments
    assert bundle.calls[-2:] == [[combined], [combined]]


def test_quality_rejection_keeps_incumbent_even_at_lower_canonical_error():
    source, bad = "A sound paragraph.", "A broken paragraph."
    bundle = Bundle({source: (.1, 0), bad: (0, 0)})
    class Proposer:
        def propose(self, *args, **kwargs):
            return [TextProposal(bad)]
    result = CanonicalRefinementSearch(None, edit_proposer=Proposer(), config=settings(),
        quality_callback=lambda text: {"accepted": text == source}, late_quality=True).run(
            bundle, (0, 0), initial_text=source)
    assert result.text == source and not result.trace["epsilon_confirmed"]
    assert result.trace["verification"]["quality_rejected"] == 1


def test_fine_edits_cannot_publish_a_chopped_lowercase_sentence_opening():
    source, broken = "Alpha beta gamma.", "lpha beta gamma."
    bundle = Bundle({source: (.1, 0), broken: (0, 0)})
    class Proposer:
        def propose(self, *args, **kwargs):
            return [TextProposal(broken)]
    result = CanonicalRefinementSearch(None, edit_proposer=Proposer(), config=settings()).run(
        bundle, (0, 0), initial_text=source)
    assert result.text == source and not result.trace["epsilon_confirmed"]
    assert result.trace["verification"]["rejections"]["sentence_case_regression"] == 1


def test_late_quality_and_epsilon_work_after_last_reserved_search_query():
    source, close = "A distant paragraph.", "A nearby paragraph."
    bundle = Bundle({source: (.1, 0), close: (.00001, 0)})
    class Proposer:
        def propose(self, *args, **kwargs):
            return [TextProposal(close)]
    result = CanonicalRefinementSearch(None, edit_proposer=Proposer(),
        config=settings(max_hard_queries=4), quality_callback=lambda text: True,
        late_quality=True).run(bundle, (0, 0), initial_text=source)
    assert result.trace["epsilon_confirmed"] and result.trace["hard_queries"] == 4


def test_no_clock_deadline_retains_query_budget_and_json_null():
    bundle = Bundle({"A sound paragraph.": (.1, 0)})
    verifier = CanonicalVerifier(bundle, (0, 0), settings(max_seconds=None, max_hard_queries=3), clock=lambda: 1e12)
    assert verifier.verify_batch(["A sound paragraph."])
    assert verifier.budget_reason() == "query_budget"
    assert verifier.diagnostics["max_seconds"] is None


def test_native_drafts_stream_every_pass_but_only_periodic_clean_text_is_scored():
    texts = ["First clean paragraph.", "Second clean paragraph.", "Third clean paragraph."]
    bundle = Bundle({text: ((3-index)/10, 0) for index, text in enumerate(texts)})
    class Runtime:
        identity = "stream-fixture"
        def generate(self, request, *, observer, observe_every, **kwargs):
            assert observe_every == 1
            for step, text in enumerate(texts):
                observer(DiffusionDraft(text, [], [], step, 3))
            return DiffusionResult(texts[-1], [], {})
    class Proposer:
        def propose(self, *args, **kwargs):
            return []
    drafts = []
    result = CanonicalRefinementSearch(Runtime(), edit_proposer=Proposer(),
        config=settings(max_rounds=1)).run(bundle, (0, 0), guided=False,
        request=DiffusionRequest("fixture"), on_draft=drafts.append)
    assert [value["text"] for value in drafts] == texts
    assert [row[0] for row in bundle.calls] == [texts[0], texts[2]]
    assert result.text == texts[-1]


def test_browser_cli_defaults_and_higher_choices_are_explicit():
    from point2text.diffusion.cli import parser
    args = parser().parse_args([])
    assert (args.candidates, args.inner_updates, args.epsilon, args.refinement_seconds) == (6, 16, .01, 120)
    args = parser().parse_args(["--candidates", "64", "--inner-updates", "64", "--unlimited-time"])
    assert args.candidates == args.inner_updates == 64 and args.unlimited_time
