from dataclasses import replace

import numpy as np
import pytest

from point2text.diffusion.refinement_contracts import RefinementConfig, TextProposal
from point2text.diffusion.verification import CanonicalVerifier


class Tokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["add_special_tokens"] is True
        return {"input_ids": [[101] + [ord(c) for c in text.lower()] + [102]
                              for text in texts]}


class Encoder:
    def __init__(self):
        self.calls = []
        self.model = type("Model", (), {"tokenizer": Tokenizer()})()

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([[float(text.split()[-1].rstrip(".!?")), 0.0] for text in texts])


class Bundle:
    map_id = "fixture-map"

    def __init__(self):
        self.encoder = Encoder()

    def project(self, vectors):
        return vectors


def config(**kwargs):
    return replace(RefinementConfig(min_words=2, max_words=10), **kwargs)


def test_batches_real_text_and_deduplicates_encoder_collisions():
    bundle = Bundle()
    verifier = CanonicalVerifier(bundle, (0, 0), config())
    results = verifier.verify_batch(["Paragraph 0.2.", "Paragraph 0.1.",
                                     "PARAGRAPH 0.1.", "Paragraph 0.2."])
    assert [value.distance for value in results] == [0.2, 0.1]
    assert bundle.encoder.calls == [["Paragraph 0.2.", "Paragraph 0.1."]]
    assert verifier.diagnostics["hard_queries"] == 2
    assert verifier.diagnostics["rejections"]["encoder_token_collision"] == 1
    assert verifier.verify_batch(["Paragraph 0.1."]) == []
    assert len(bundle.encoder.calls) == 1


def test_epsilon_requires_two_fresh_singletons_and_uses_distance_not_square():
    bundle = Bundle()
    verifier = CanonicalVerifier(bundle, (0, 0), config(epsilon=1e-4))
    near, far = verifier.verify_batch(["Paragraph 0.00005.", "Paragraph 0.005."])
    assert far.distance**2 < verifier.config.epsilon
    assert not verifier.confirm_epsilon(far)
    assert len(bundle.encoder.calls) == 1
    assert verifier.confirm_epsilon(near)
    assert bundle.encoder.calls[-2:] == [[near.text], [near.text]]
    assert verifier.diagnostics["confirmation_queries"] == 2
    assert verifier.last_confirmed.distance <= verifier.config.epsilon
    assert all(not row["cached"] for row in verifier.confirmation["checks"])


def test_one_bad_fresh_check_prevents_success_and_records_worst():
    calls = []

    def scorer(texts, target):
        calls.append(texts)
        value = [0.00001, 0.00002, 0.0002][len(calls)-1]
        return [((value, 0), value * value) for _ in texts]

    verifier = CanonicalVerifier(Bundle(), (0, 0), config(), score_batch=scorer)
    value = verifier.verify_batch(["Paragraph 0.1."])[0]
    assert not verifier.confirm_epsilon(value)
    assert verifier.confirmation["worst_distance"] == 0.0002
    assert len(calls) == 3


def test_two_queries_are_reserved_for_confirmation():
    bundle = Bundle()
    verifier = CanonicalVerifier(bundle, (0, 0), config(max_hard_queries=3))
    values = verifier.verify_batch(["Paragraph 0.00001.", "Paragraph 0.1."])
    assert len(values) == 1
    assert verifier.budget_reason() == "query_budget"
    assert verifier.confirm_epsilon(values[0])
    assert verifier.diagnostics["hard_queries"] == 3


def test_invalid_proposals_and_callback_failure_do_not_crash_batch():
    bundle = Bundle()
    verifier = CanonicalVerifier(bundle, (0, 0), config(max_encoder_tokens=30),
                                  quality_callback=lambda text: "forbidden" not in text)
    values = verifier.verify_batch(["", "one", "## Header 0.1", "bad \ufffd 0.1",
                                     "forbidden 0.1.", "Long " * 10 + "0.1.",
                                     "Paragraph 0.1."])
    assert [value.text for value in values] == ["Paragraph 0.1."]
    assert verifier.diagnostics["rejected"] == 6
    assert verifier.diagnostics["coherence_certificate"] is False


def test_over_limit_encoder_tokens_are_rejected_before_encoding():
    bundle = Bundle()
    verifier = CanonicalVerifier(bundle, (0, 0), config(max_encoder_tokens=8))
    assert verifier.verify_batch(["Paragraph 0.1."]) == []
    assert bundle.encoder.calls == []
    assert verifier.diagnostics["rejections"]["encoder_token_limit"] == 1


def test_time_budget_is_separate_from_query_budget():
    now = [0.0]
    verifier = CanonicalVerifier(Bundle(), (0, 0), config(max_seconds=1), clock=lambda: now[0])
    now[0] = 2.0
    assert verifier.budget_reason() == "time_budget"
    assert verifier.verify_batch(["Paragraph 0.1."]) == []
    assert verifier.diagnostics["hard_queries"] == 0


def test_bad_member_of_batch_does_not_discard_good_members():
    def scorer(texts, target):
        if any("bad" in text for text in texts):
            raise ValueError("invalid backend row")
        return [((0.1, 0.0), 0.1) for _ in texts]

    verifier = CanonicalVerifier(Bundle(), (0, 0), config(), score_batch=scorer)
    values = verifier.verify_batch(["bad 0.1.", "good 0.1."])
    assert [value.text for value in values] == ["good 0.1."]
    assert verifier.diagnostics["hard_queries"] == 4


def test_fixture_without_tokenizer_reports_no_id_dedup():
    bundle = type("Scorer", (), {"score": lambda self, text, target: ((0.2, 0), 0.2)})()
    verifier = CanonicalVerifier(bundle, (0, 0), config())
    assert len(verifier.verify_batch(["One paragraph.", "Other paragraph."])) == 2
    assert verifier.diagnostics["encoder_token_fingerprint_available"] is False


def test_default_word_limits_and_raw_model_headers_are_mechanical_rejections():
    class WordTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [list(range(len(text.split()) + 2)) for text in texts]}

    verifier = CanonicalVerifier(Bundle(), (0, 0), tokenizer=WordTokenizer())
    short = " ".join([f"term{i}" for i in range(58)] + ["0.1."])
    valid = " ".join([f"term{i}" for i in range(59)] + ["0.1."])
    header = "<start_of_turn> " + short
    values = verifier.verify_batch([short, header, valid])
    assert len(values) == 1 and values[0].text == valid.strip()
    assert verifier.diagnostics["rejections"]["word_limit"] == 1
    assert verifier.diagnostics["rejections"]["special_header_or_artifact"] == 1


def test_confirmation_errors_do_not_claim_cached_success():
    calls = []

    def scorer(texts, target):
        calls.append(texts)
        if len(calls) == 3:
            raise RuntimeError("singleton confirmation failed")
        return [((0.00001, 0), 0.00001) for _ in texts]

    verifier = CanonicalVerifier(Bundle(), (0, 0), config(), score_batch=scorer)
    value = verifier.verify_batch(["Paragraph 0.1."])[0]
    assert not verifier.confirm_epsilon(value)
    assert verifier.last_confirmed is None
    assert verifier.diagnostics["confirmation_queries"] == 2


@pytest.mark.parametrize("text,reason", [
    ("The the the optimizer converges.", "repeated_word"),
    ("The data data observed were collected.", "repeated_word"),
    ("Use the the same encoder.", "repeated_word"),
    ("Normalize model model model inputs.", "repeated_word"),
    ("The optimizer uses a bounded learning rate", "unfinished_sentence"),
    ("The optimizer converges..", "punctuation_artifact"),
    ("The optimizer converges. ?", "punctuation_artifact"),
])
def test_repetitive_or_unfinished_drafts_are_rejected_before_encoding(text, reason):
    bundle = Bundle()
    verifier = CanonicalVerifier(bundle, (0, 0), config(max_words=100))
    assert verifier.verify_batch([text]) == []
    assert bundle.encoder.calls == []
    assert verifier.diagnostics["rejections"][reason] == 1


@pytest.mark.parametrize("text", [
    "That analysis shows that that assumption is necessary.",
    "The system had had sufficient observations before deployment.",
    'The stability condition is stated as "all eigenvalues are bounded."',
    "The update remains bounded (under the stated assumptions.)",
    "Data. Data analysis uses the ordinary encoder.",
])
def test_legitimate_completed_text_is_not_a_mechanical_grammar_claim(text):
    verifier = CanonicalVerifier(Bundle(), (0, 0), config(max_words=100),
                                  score_batch=lambda texts, target: [((0.1, 0), 0.1)] * len(texts))
    result = verifier.verify_batch([text])
    assert len(result) == 1
    assert result[0].diagnostics["coherence_certified"] is False


def test_default_limits_accept_a_complete_technical_paragraph():
    text = ("Principal component analysis projects centered observations onto orthogonal "
            "directions selected to explain variance. A fixed projection lets new samples "
            "share the same coordinate system without refitting the map. Distances in the "
            "displayed plane retain only those selected components, so nearby points can "
            "still differ in omitted dimensions. Evaluation should therefore report the "
            "projection policy alongside measurements from the full embedding space and "
            "checks on the underlying text.")

    def tokenizer(texts, **kwargs):
        return {"input_ids": [list(range(len(value.split()) + 2)) for value in texts]}

    verifier = CanonicalVerifier(Bundle(), (0, 0), tokenizer=tokenizer,
                                  score_batch=lambda texts, target: [((0.1, 0), 0.1)] * len(texts))
    values = verifier.verify_batch([text])
    assert len(values) == 1
    assert 60 <= values[0].diagnostics["word_count"] <= 100
