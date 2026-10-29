"""Target-blind sentence banks and mocked loopback transport, no model calls."""

import asyncio
import json
import threading

import httpx
import pytest

from point2text.diffusion.combinations import MeasuredCombinationPlanner
from point2text.diffusion.paraphrases import _screen_reason
from point2text.diffusion.refinement_contracts import VerifiedText
from point2text.diffusion.sentence_bank import (
    SCHEMA, OllamaSentenceParaphraseProvider, SentenceParaphraseBank,
    sentence_screen_reason, source_hash,
)
from point2text.diffusion.sentence_edits import sentence_spans


SOURCE = "A model may use 3 samples for analysis. The measurements do not establish a causal relationship."
FIRST = "For analysis, a model may use 3 samples."
SECOND = "A causal relationship is not what the measurements establish."


class Provider:
    identity = "mock:sentence-provider"

    def __init__(self):
        self.calls = []

    def suggestions(self, text, span):
        self.calls.append((text, span))
        return [FIRST] if span[0] == 0 else [SECOND]


def test_actual_sentence_span_roundtrip_target_blindness_and_paging():
    a, b = SentenceParaphraseBank(provider=Provider()), SentenceParaphraseBank(provider=Provider())
    all_a = a.all_proposals(SOURCE)
    assert len(all_a) == 2
    assert len(a.provider.calls) == 2
    pages = [*a.propose(SOURCE, (0, 0), max_proposals=1), *a.propose(SOURCE, (100, -200), max_proposals=1)]
    assert [p.text for p in pages] == [p.text for p in all_a]
    assert a.propose(SOURCE, (0, 0)) == []
    assert [p.text for p in b.propose(SOURCE, (-9, 8))] == [p.text for p in all_a]
    assert a.hotspots(SOURCE, (0, 0)) == sentence_spans(SOURCE)
    assert len(a.provider.calls) == 2
    for proposal in all_a:
        span, = proposal.provenance["source_spans"]
        replacement, = proposal.provenance["replacements"]
        start, end = span
        assert proposal.edit_span == (start, end)
        assert SOURCE[:start] + replacement + SOURCE[end:] == proposal.text
        assert proposal.provenance["source_sha256"] == source_hash(SOURCE)
        assert proposal.provenance["source_sentence_sha256"] == source_hash(SOURCE[start:end])
        assert proposal.provenance["target_blind"] and not proposal.provenance["semantic_verified"]


def test_nonoverlapping_sentence_atoms_integrate_with_measured_combination_planner():
    proposals = SentenceParaphraseBank(provider=Provider()).all_proposals(SOURCE)
    base = VerifiedText(SOURCE, 0, 0, 1)
    observed = [VerifiedText(proposals[0].text, .3, .1, 1), VerifiedText(proposals[1].text, -.2, -.1, 1)]
    planner = MeasuredCombinationPlanner()
    assert planner.record(base, proposals, observed)["accepted"] == 2
    combined = planner.propose(base, (.1, 0))
    assert len(combined) == 1 and combined[0].text == FIRST + " " + SECOND
    assert combined[0].provenance["canonical_verified"] is False


@pytest.mark.parametrize("replacement,reason", [
    ("For analysis, a model may use 4 samples.", "numeric_drift"),
    ("For analysis, a model will use 3 samples.", "qualification_quantity_or_technical_drift"),
    ("For analysis, a method may use 3 samples.", "qualification_quantity_or_technical_drift"),
    ("For analysis, a model may use 3 samples. Extra claim.", "not_one_complete_sentence"),
])
def test_numeric_modal_technical_and_extra_claim_guards(replacement, reason):
    assert sentence_screen_reason(SOURCE[:sentence_spans(SOURCE)[0][1]], replacement) == reason


def test_units_entities_and_contractions_are_retained():
    assert sentence_screen_reason("NASA may use 3 m and 5 s.", "NASA may use 3 s and 5 m.") == "unit_drift"
    assert sentence_screen_reason("NASA may use 3 samples.", "ESA may use 3 samples.") == "capitalized_entity_drift"
    assert sentence_screen_reason("A model can't change.", "A model can change.") is not None
    assert sentence_screen_reason("A model may use 3 samples.", FIRST) is None


def test_old_atomic_word_limit_is_unchanged():
    original = SOURCE[:sentence_spans(SOURCE)[0][1]]
    assert _screen_reason(SOURCE, 0, len(original), FIRST) == "nonalphabetic_or_word_limit"
    assert sentence_screen_reason(original, FIRST) is None


def test_reviewed_json_exact_hash_span_and_metadata(tmp_path):
    span = sentence_spans(SOURCE)[0]
    data = {"schema": SCHEMA, "sources": [{"source_sha256": source_hash(SOURCE), "text": SOURCE,
        "sentences": [{"span": list(span), "original": SOURCE[slice(*span)],
                       "variants": [{"text": FIRST, "review": {"reviewer": "fixture", "accepted": True}}]}]}]}
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(data))
    bank = SentenceParaphraseBank(reviewed_bank=path)
    proposal, = bank.propose(SOURCE, (0, 0))
    assert proposal.provenance["supplied_review"]["accepted"] is True
    assert proposal.provenance["semantic_verified"] is False
    data["sources"][0]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        SentenceParaphraseBank(reviewed_bank=data)


def test_source_capacity_stop_and_transient_construction_resume():
    stop = threading.Event()

    class PausingProvider(Provider):
        def suggestions(self, text, span):
            result = super().suggestions(text, span)
            if len(self.calls) == 1:
                stop.set()
            return result

    provider = PausingProvider()
    bank = SentenceParaphraseBank(provider=provider, should_stop=stop.is_set, max_cached_sources=1)
    assert bank.propose(SOURCE, (0, 0)) == []
    stop.clear()
    assert len(bank.propose(SOURCE, (0, 0))) == 2
    assert len(provider.calls) == 3  # Stopped sentence retried, complete ones frozen.
    assert bank.propose(SOURCE + " A distinct ending.", (0, 0)) == []
    assert bank.diagnostics["source_capacity_rejections"] == 1


def test_variant_limits_and_duplicate_screening():
    class Repeats:
        def suggestions(self, text, span):
            return [FIRST, FIRST, "A model may use 4 samples."]

    bank = SentenceParaphraseBank(provider=Repeats(), max_variants_per_sentence=2)
    assert len(bank.propose(SOURCE, (0, 0), max_proposals=1)) <= 1
    assert bank.diagnostics["accepted_variants"] <= 2 * len(sentence_spans(SOURCE))
    assert bank.diagnostics["rejections"].get("duplicate_variant", 0) >= 1


def test_loopback_provider_mocked_schema_cache_and_target_absence():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "fixture:latest", "digest": "fixed-digest"}]})
        payload = json.loads(request.content)
        assert "target" not in json.loads(payload["prompt"].split("\n", 1)[1])
        assert "sentence" in payload["prompt"] and "paragraph" in payload["prompt"]
        assert "variants" in payload["format"]["properties"]
        output = json.dumps({"variants": [FIRST]})
        return httpx.Response(200, text=json.dumps({"response": output, "done": True,
                                                  "prompt_eval_count": 42, "eval_count": 20}) + "\n")

    provider = OllamaSentenceParaphraseProvider(model="fixture", transport=httpx.MockTransport(handler))
    span = sentence_spans(SOURCE)[0]
    assert provider.suggestions(SOURCE, span) == (FIRST,)
    assert provider.suggestions(SOURCE, span) == (FIRST,)
    assert len(calls) == 2 and provider.diagnostics["calls"] == 1
    assert provider.diagnostics["cache_hits"] == 1
    assert provider.diagnostics["model_digest"] == "fixed-digest"
    assert provider.diagnostics["prompt_tokens"] == 42


def test_provider_is_constructed_without_io_and_rejects_nonlocal_urls():
    with pytest.raises(ValueError, match="loopback"):
        OllamaSentenceParaphraseProvider(base_url="https://example.org")
    stopped = OllamaSentenceParaphraseProvider(should_stop=lambda: True)
    assert stopped.suggestions(SOURCE, sentence_spans(SOURCE)[0]) == ()
    assert stopped.diagnostics["calls"] == 0


def test_provider_cancels_inflight_mock_prefill_without_negative_cache():
    stop = threading.Event()
    posts = []

    async def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "fixture:latest", "digest": "fixed"}]})
        posts.append(request)
        if len(posts) == 1:
            stop.set()
            await asyncio.sleep(1)  # The bounded watcher cancels this mock call.
        return httpx.Response(200, text=json.dumps({"response": json.dumps({"variants": [FIRST]}), "done": True}) + "\n")

    provider = OllamaSentenceParaphraseProvider(model="fixture", should_stop=stop.is_set,
        transport=httpx.MockTransport(handler), max_calls=2, timeout=2)
    span = sentence_spans(SOURCE)[0]
    assert provider.suggestions(SOURCE, span) == ()
    assert provider._sentence_cache == {}
    cancelled, = provider.diagnostics["sources"].values()
    assert cancelled["status"] == "cancelled_or_deadline"
    stop.clear()
    assert provider.suggestions(SOURCE, span) == (FIRST,)
    assert len(posts) == provider.diagnostics["calls"] == 2
    completed, = provider.diagnostics["sources"].values()
    assert completed["status"] == "complete"
    assert "cancelled_or_deadline" in completed["previous_statuses"]


def test_provider_call_limit_prevents_further_mock_requests():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "fixture:latest", "digest": "fixed"}]})
        return httpx.Response(200, text=json.dumps({"response": json.dumps({"variants": [FIRST]}), "done": True}) + "\n")

    provider = OllamaSentenceParaphraseProvider(model="fixture", max_calls=1, transport=httpx.MockTransport(handler))
    first, second = sentence_spans(SOURCE)
    assert provider.suggestions(SOURCE, first) == (FIRST,)
    assert provider.suggestions(SOURCE, second) == ()
    assert len(requests) == 2


def test_repeated_identical_sentences_are_addressed_by_exact_offsets():
    original = SOURCE[:sentence_spans(SOURCE)[0][1]]
    source = original + " " + original
    bank = SentenceParaphraseBank(provider=Provider())
    # This provider selects a different second-sentence response for SOURCE;
    # use an explicit identical-sentence provider for both exact occurrences.
    bank.provider = type("Repeated", (), {"suggestions": lambda self, text, span: [FIRST]})()
    proposals = bank.all_proposals(source)
    assert len(proposals) == 2 and proposals[0].edit_span != proposals[1].edit_span
    assert proposals[0].text != proposals[1].text


def test_raw_good_bad_duplicate_variants_and_reasons_are_retained_target_blind():
    bad = "For analysis, a model may use 4 samples."
    raw = json.dumps({"variants": [bad, FIRST, FIRST]})

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "fixture:latest", "digest": "fixed"}]})
        payload = json.loads(request.content)
        source = json.loads(payload["prompt"].split("\n", 1)[1])
        assert set(source) == {"paragraph", "span", "sentence"}
        return httpx.Response(200, text=json.dumps({"response": raw, "done": True}) + "\n")

    provider = OllamaSentenceParaphraseProvider(model="fixture", transport=httpx.MockTransport(handler))
    span = sentence_spans(SOURCE)[0]
    assert provider.suggestions(SOURCE, span) == (FIRST,)
    receipt, = provider.diagnostics["sources"].values()
    assert receipt["source_sha256"] == source_hash(SOURCE) and receipt["span"] == list(span)
    assert receipt["raw_response"] == raw
    assert receipt["raw_variants"] == [bad, FIRST, FIRST]
    assert receipt["accepted_variants"] == [FIRST]
    assert receipt["rejections"] == {"numeric_drift": 1, "duplicate_variant": 1}
    assert [row["reason"] for row in receipt["variant_checks"]] == ["numeric_drift", None, "duplicate_variant"]
    assert provider.diagnostics["raw_variant_count"] == 3
    assert provider.diagnostics["accepted_variant_count"] == 1
    assert receipt["status"] == "complete" and receipt["target_blind"]


def test_invalid_json_error_receipt_keeps_raw_and_cache_hit_does_not_erase_it():
    raw = '{"variants": [this is invalid JSON]}'

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "fixture:latest", "digest": "fixed"}]})
        return httpx.Response(200, text=json.dumps({"response": raw, "done": True}) + "\n")

    provider = OllamaSentenceParaphraseProvider(model="fixture", transport=httpx.MockTransport(handler))
    span = sentence_spans(SOURCE)[0]
    assert provider.suggestions(SOURCE, span) == ()
    receipt, = provider.diagnostics["sources"].values()
    assert receipt["status"] == "error" and receipt["reason"] == "invalid_json"
    assert receipt["raw_response"] == raw and "JSONDecodeError" in receipt["error"]
    assert provider.suggestions(SOURCE, span) == ()
    assert receipt["status"] == "error" and receipt["raw_response"] == raw
    assert provider.diagnostics["cache_hits"] == 1


def test_diagnostic_source_capacity_is_bounded_even_for_cancelled_requests():
    provider = OllamaSentenceParaphraseProvider(max_cached_sources=1, should_stop=lambda: True)
    first, second = sentence_spans(SOURCE)
    assert provider.suggestions(SOURCE, first) == ()
    assert provider.suggestions(SOURCE, second) == ()
    assert len(provider.diagnostics["sources"]) == 1
    assert provider.diagnostics["diagnostic_capacity_refusals"] == 1
    assert provider.diagnostics["calls"] == 0


def test_capitalized_sentence_initial_technical_word_overrejection_is_explicit():
    # A conservative known false negative: no guard is weakened to allow this.
    assert sentence_screen_reason("Measurements may capture variation.",
        "Variation may be captured by measurements.") == "capitalized_entity_drift"
