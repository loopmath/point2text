"""Atomic reconstruction and bounded target-blind paraphrase suggestion banks."""

import asyncio
import json
import subprocess
import sys
import threading
import time

import httpx
import pytest

from point2text.diffusion.paraphrases import (
    OllamaParaphraseProvider,
    ParaphraseEditProposer,
)


SOURCE = (
    "In this study we use the estimator in order to calculate a posterior. "
    "For example, it is calculated for each sample using the fixed prior. "
    "In addition, the estimate depends on observations taken into account at each step. "
    "In general, the matrix consists of blocks with respect to a wide range of inputs. "
    "Consequently, it is computed approximately as specified, with the exception of invalid inputs."
)


def assert_atomic(source, proposal):
    (span,) = proposal.provenance["source_spans"]
    (replacement,) = proposal.provenance["replacements"]
    start, end = span
    assert proposal.edit_span == (start, end)
    assert 0 <= start < end <= len(source)
    assert source[:start] + replacement + source[end:] == proposal.text
    assert proposal.text != source
    assert proposal.provenance["canonical_verified"] is False
    assert proposal.provenance["semantic_verified"] is False
    assert proposal.provenance["quality_status"] == "unreviewed"


class Provider:
    identity = "test:anchored"

    def __init__(self, items):
        self.items = items
        self.seen = []

    def suggestions(self, text):
        self.seen.append(text)
        return self.items


def test_curated_bank_contains_independent_atomic_variable_length_edits():
    proposer = ParaphraseEditProposer()
    bank = proposer.all_proposals(SOURCE)
    assert len({p.edit_span for p in bank}) >= 12
    assert any(len(p.text) != len(SOURCE) for p in bank)
    assert len({p.text for p in bank}) == len(bank)
    for proposal in bank:
        assert_atomic(SOURCE, proposal)
    # Scientific noun/qualification families deliberately do not appear.
    assert not ParaphraseEditProposer().all_proposals(
        "An accurate model estimates reliable measurements with significant uncertainty.")


def test_pagination_exhausts_bank_without_repeating_first_page_or_provider():
    provider = Provider([{"original": "fixed prior", "replacement": "unchanged prior"}])
    proposer = ParaphraseEditProposer(provider=provider)
    expected = proposer.all_proposals(SOURCE)
    pages = []
    while page := proposer.propose(SOURCE, (0.1, 0.2), max_proposals=7):
        pages.extend(page)
    assert [p.text for p in pages] == [p.text for p in expected]
    assert len({p.text for p in pages}) == len(pages)
    assert proposer.propose(SOURCE, (-5.0, 8.0), max_proposals=64) == []
    assert [p.text for p in proposer.all_proposals(SOURCE)] == [p.text for p in expected]
    assert provider.seen == [SOURCE]


def test_target_blind_deterministic_bank_and_hotspots_do_not_advance_pages():
    a, b = ParaphraseEditProposer(), ParaphraseEditProposer(object())
    hotspots = a.hotspots(SOURCE, (123, -456), limit=4)
    assert len(hotspots) == 4
    assert len(set(hotspots)) == len(hotspots)
    assert [p.text for p in a.propose(SOURCE, (0, 0), max_proposals=100)] == [
        p.text for p in b.propose(SOURCE, (17, -23), max_proposals=100)]


def test_untrusted_anchors_boundaries_quantities_entities_and_strength_are_screened():
    text = (
        "A compact estimator may suggest a relation for NASA under 12 conditions. "
        "The estimator is not unbiased and has some uncertainty. "
        "A repeated prior and repeated prior are ambiguous anchors."
    )
    items = [
        {"original": "compact estimator", "replacement": "estimator that is compact"},
        {"original": "compact estimator", "replacement": "estimator that is compact"},
        {"original": "compact estimator", "replacement": "compact model"},
        {"original": "pact estimator", "replacement": "short estimator"},  # substring
        {"original": "missing anchor", "replacement": "another anchor"},
        {"original": "repeated prior", "replacement": "fixed prior"},  # non-unique
        {"original": "may suggest", "replacement": "demonstrates"},
        {"original": "not unbiased", "replacement": "unbiased"},
        {"original": "some uncertainty", "replacement": "uncertainty"},
        {"original": "NASA", "replacement": "ESA"},
        {"original": "under 12 conditions", "replacement": "under 13 conditions"},
        {"original": "compact estimator", "replacement": "accurate estimator"},
        {"original": "compact estimator", "replacement": "compact [MASK] estimator"},
        {"original": "compact estimator", "replacement": "one two three four five six seven eight nine"},
        {"original": "some uncertainty", "replacement": "some uncertainty."},
    ]
    proposals = ParaphraseEditProposer(provider=Provider(items)).all_proposals(text)
    assert len(proposals) == 1
    assert proposals[0].provenance["replacements"] == ["estimator that is compact"]
    assert_atomic(text, proposals[0])


def test_case_offsets_repeated_curated_spans_and_protected_token_boundaries():
    text = "We use tensors. We use scalars. misuse_the for exampleX for example-safe."
    bank = ParaphraseEditProposer().all_proposals(text)
    assert bank
    assert {p.edit_span[0] for p in bank} == {0, 16}
    for proposal in bank:
        assert_atomic(text, proposal)
        assert proposal.provenance["replacements"][0].startswith("We ")


def test_curated_prefixes_do_not_break_fixed_phrases_or_addition_to():
    text = "In general relativity, curvature matters in addition to matter."
    assert ParaphraseEditProposer().all_proposals(text) == ()


def test_bank_capacity_is_bounded_without_eviction_or_regeneration():
    provider = Provider([])
    proposer = ParaphraseEditProposer(provider=provider, max_bank_size=5, max_cached_sources=1)
    assert len(proposer.all_proposals(SOURCE)) == 5
    assert proposer.all_proposals("For example, a second paragraph.") == ()
    assert len(proposer.all_proposals(SOURCE)) == 5
    assert provider.seen == [SOURCE]


def test_cancel_before_and_after_provider_does_not_emit_proposals():
    cancel = threading.Event()
    provider = Provider([])
    proposer = ParaphraseEditProposer(provider=provider, should_stop=cancel.is_set)
    cancel.set()
    assert proposer.propose(SOURCE, (0, 0)) == []
    assert provider.seen == []
    cancel.clear()

    class CancellingProvider(Provider):
        def suggestions(self, text):
            cancel.set()
            return super().suggestions(text)

    proposer = ParaphraseEditProposer(provider=CancellingProvider([]), should_stop=cancel.is_set)
    assert proposer.propose(SOURCE, (0, 0)) == []


def test_optional_provider_failure_retains_screened_curated_bank_once():
    class FailingProvider(Provider):
        def suggestions(self, text):
            self.seen.append(text)
            raise ValueError("malformed provider output")

    provider = FailingProvider([])
    proposer = ParaphraseEditProposer(provider=provider)
    assert proposer.all_proposals(SOURCE)
    assert proposer.all_proposals(SOURCE)
    assert provider.seen == [SOURCE]
    assert proposer.diagnostics["provider_errors"] == ["malformed provider output"]


def test_import_does_not_import_torch():
    subprocess.run([sys.executable, "-c", (
        "import sys; from point2text.diffusion.paraphrases import ParaphraseEditProposer; "
        "assert 'torch' not in sys.modules; assert ParaphraseEditProposer().all_proposals('For example, text.')"
    )], check=True)


def transport_for(items, *, digest="sha256:frozen", done_reason="stop", malformed=False):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "gemma4:12b", "digest": digest}]})
        assert request.url.path == "/api/generate"
        response = "not JSON" if malformed else json.dumps({"edits": items})
        return httpx.Response(200, text=json.dumps({
            "response": response, "done": True, "done_reason": done_reason,
            "prompt_eval_count": 41, "eval_count": 23}) + "\n")

    return httpx.MockTransport(respond), requests


def test_local_provider_structured_source_only_requests_frozen_digest_and_cache():
    transport, requests = transport_for([
        {"original": "fixed prior", "replacement": "unchanged prior"},
        {"original": "fixed prior", "replacement": "unchanged prior"},
        {"original": "fixed prior", "replacement": "accurate prior"},
    ])
    provider = OllamaParaphraseProvider(transport=transport, max_calls=1)
    assert provider.suggestions(SOURCE) == ({"original": "fixed prior", "replacement": "unchanged prior"},)
    assert provider.suggestions(SOURCE) == provider.suggestions(SOURCE)
    assert provider.suggestions("A different fixed prior.") == ()
    assert [r.url.path for r in requests] == ["/api/tags", "/api/generate"]
    payload = json.loads(requests[1].content)
    assert payload["model"] == "gemma4:12b"
    assert payload["options"]["temperature"] == 0
    assert payload["format"]["properties"]["edits"]["maxItems"] == 96
    assert json.loads(payload["prompt"].split("\n", 1)[1]) == SOURCE
    assert "target" not in payload and "distance" not in payload
    assert provider.identity.endswith("#sha256:frozen")
    assert provider.diagnostics["calls"] == 1
    assert provider.diagnostics["prompt_tokens"] == 41
    assert provider.diagnostics["completion_tokens"] == 23
    assert provider.diagnostics["raw_edit_count"] == 3
    assert provider.diagnostics["accepted_edit_count"] == 1
    assert provider.diagnostics["rejections"]["duplicate_edit"] == 1
    assert provider.diagnostics["rejections"]["qualification_quantity_or_technical_drift"] == 1
    assert len(provider.diagnostics["last_raw_edits"]) == 3
    assert len(provider.diagnostics["last_accepted_edits"]) == 1
    assert provider.diagnostics["sources"]


@pytest.mark.parametrize("kwargs", [{"malformed": True}, {"done_reason": "length"}])
def test_invalid_or_truncated_model_json_is_not_retried(kwargs):
    transport, requests = transport_for([], **kwargs)
    provider = OllamaParaphraseProvider(transport=transport)
    assert provider.suggestions(SOURCE) == ()
    assert provider.suggestions(SOURCE) == ()
    assert len(requests) == 2
    assert provider.diagnostics["last_error"]


def test_changed_digest_is_rejected_without_second_generation():
    digest = ["first"]
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma4:12b", "digest": digest[0]}]})
        return httpx.Response(200, text=json.dumps({"response": '{"edits":[]}', "done": True}) + "\n")

    provider = OllamaParaphraseProvider(transport=httpx.MockTransport(respond))
    provider.suggestions(SOURCE)
    digest[0] = "second"
    assert provider.suggestions(SOURCE + " Extra sentence.") == ()
    assert [r.url.path for r in requests].count("/api/generate") == 1
    assert "digest changed" in provider.diagnostics["last_error"]
    assert provider.identity.endswith("#first")


@pytest.mark.parametrize("metadata", [[], {"models": "invalid"}, {"models": []},
    {"models": [{"name": "gemma4:12b", "digest": "local", "remote_host": "cloud"}]}])
def test_missing_malformed_or_remote_model_metadata_fails_closed_without_generation(metadata):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=metadata)

    provider = OllamaParaphraseProvider(transport=httpx.MockTransport(respond))
    assert provider.suggestions(SOURCE) == ()
    assert [r.url.path for r in requests] == ["/api/tags"]
    assert provider.diagnostics["calls"] == 0
    assert provider.diagnostics["last_error"]


def test_provider_cancellation_interrupts_prefill_and_deadline_has_no_io():
    cancel = threading.Event()
    started = threading.Event()
    closed = threading.Event()

    async def respond(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma4:12b", "digest": "frozen"}]})
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            closed.set()

    provider = OllamaParaphraseProvider(transport=httpx.MockTransport(respond), should_stop=cancel.is_set)
    timer = threading.Thread(target=lambda: (started.wait(1), cancel.set()))
    timer.start()
    before = time.monotonic()
    assert provider.suggestions(SOURCE) == ()
    timer.join(1)
    assert time.monotonic() - before < 1
    assert closed.is_set()
    assert provider.diagnostics["last_error"] == "cancelled_or_deadline"
    transport, requests = transport_for([])
    provider = OllamaParaphraseProvider(transport=transport, deadline=time.monotonic() - 1)
    assert provider.suggestions(SOURCE) == ()
    assert requests == []


@pytest.mark.parametrize("url", ["https://ollama.com", "http://192.168.1.2:11434", "http://user@localhost:11434"])
def test_provider_rejects_nonlocal_or_credentialled_urls(url):
    with pytest.raises(ValueError, match="loopback"):
        OllamaParaphraseProvider(base_url=url)


@pytest.mark.parametrize("target", [(float("nan"), 0), (1,), ("a", "b")])
def test_target_and_bounds_validation(target):
    with pytest.raises(ValueError):
        ParaphraseEditProposer().propose(SOURCE, target)
    with pytest.raises(ValueError):
        ParaphraseEditProposer(max_bank_size=0)
    with pytest.raises(ValueError):
        ParaphraseEditProposer().propose(SOURCE, (0, 0), max_proposals=-1)


PROSE = (
    "The process by which a system loses quantum properties involves interaction. Researchers employ error correction codes and dynamical decoupling techniques. They distinguish between noise introduced by hardware and intrinsic rates to characterize fidelity.",
    "Algorithms ensure that nodes agree on a state transition despite the presence of network partitions or partial failures. Protocols typically utilize a leader-based approach or a quorum-based mechanism. By requiring a majority, they preserve term numbers and log replication.",
    "DNA sequences associated with proteins are identified. By using antibodies researchers distinguish true binding regions from noise. Integration allows for the determination of how structural changes drive expression in response to specific environmental stimuli or developmental signals.",
    "The forcing changes with altered greenhouse gas concentrations. To ensure accuracy researchers distinguish between direct forcing and indirect feedbacks. By isolating components they estimate the climate sensitivity of the planet and surface temperatures over time.",
    "Federated learning enables the training of models across decentralized devices. Instead of centralizing information they send updates. Evaluation considers communication overhead on convergence speed and global model performance against privacy risks associated with potential gradient leakage during synchronization.",
    "Localization tracks location within it. This process relies on estimation. However, errors in motion estimation drift over time. Algorithms utilize loop closure detection to recognize previously visited landmarks, correcting accumulated errors in generated spatial data.",
    "Plasticity involves the strengthening or weakening of connections based on patterns of activity. It often relies on the trafficking of receptors, altering the sensitivity of the cell. To accurately quantify changes researchers distinguish between functional modifications and structural fluctuations.",
    "Transitions occur under thermal or mechanical stress. The process depends on the competition between chemical energy and energy associated with boundaries. By controlling the grain size and solute distribution, researchers observe changes. Analyzing shifts allows for a precise understanding of energy dissipation.",
    "Protocols ensure that nodes agree despite network partitions or component failures. Mechanisms typically utilize a leader-based voting process or a quorum approach. Research addresses the trade-off between consistency guarantees and availability.",
    "Spectroscopy relies on the analysis of absorption lines for chemical composition and physical conditions. The width and shift allow researchers to calculate surface temperature and surface gravity. By modeling profiles, astronomers distinguish between stages and refine parameters governing evolution.",
)


@pytest.mark.parametrize("text", PROSE)
def test_direct_technical_prose_has_useful_atomic_syntax_bank(text):
    proposer = ParaphraseEditProposer()
    bank = proposer.all_proposals(text)
    assert len(bank) >= 6
    assert len({proposal.edit_span for proposal in bank}) >= 4
    assert len({proposal.text for proposal in bank}) == len(bank)
    for proposal in bank:
        assert_atomic(text, proposal)
    stats = proposer.diagnostics["sources"][bank[0].provenance["source_sha256"]]
    assert stats["bank_size"] == len(bank)
    assert stats["distinct_spans"] == len({proposal.edit_span for proposal in bank})
    assert stats["curated_candidates"] >= stats["curated_accepted"] >= len(bank)


def test_equivalent_whole_phrase_and_word_atoms_are_deduplicated_by_text():
    bank = ParaphraseEditProposer().all_proposals("We utilize tensors to calculate a quantity.")
    assert len({p.text for p in bank}) == len(bank)
    use = next(p for p in bank if p.text == "We use tensors to calculate a quantity.")
    assert use.edit_span == (3, 10)  # Keep the smaller span for composition.


def test_surface_preposition_is_not_a_quantity_and_partial_bound_edits_are_rejected():
    provider = Provider([
        {"original": "surface temperature", "replacement": "temperature at the surface"},
        {"original": "at", "replacement": "in"},
    ])
    text = "We compare surface temperature for at least two conditions."
    proposer = ParaphraseEditProposer(provider=provider)
    bank = proposer.all_proposals(text)
    assert any("temperature at the surface" in p.text for p in bank)
    assert all("at least two" in p.text for p in bank)
    stats = next(iter(proposer.diagnostics["sources"].values()))
    assert stats["provider_rejections"]["quantity_bound_drift"] == 1


def test_plural_analyses_is_not_misread_as_a_british_verb():
    assert ParaphraseEditProposer().all_proposals("The analyses are complete.") == ()
