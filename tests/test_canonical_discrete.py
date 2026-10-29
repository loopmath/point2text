"""Canonical word boundaries, bounded proposals, gradients, and provenance."""

from types import SimpleNamespace
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from point2text.contracts import CorpusRecord
from point2text.diffusion.discrete import CanonicalEditProposer
from point2text.diffusion.encoder import TorchEmbeddingReward


class OffsetTokenizer:
    pad_token_id, cls_token_id, sep_token_id, unk_token_id, mask_token_id = range(5)
    all_special_ids = list(range(5))
    words = "a an useful helpful valuable model method framework system procedure process mechanism strategy compares evaluates assesses measured observed recorded observations measurements recordings with explicit clear precise assumptions constraints requirements careful direct local independent controlled . in order to for example instance can be used rather than instead of requires needs demands research technical is compare".split()
    vocabulary = {word: index + 5 for index, word in enumerate(words)}

    def __init__(self):
        self.offset_calls = 0

    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False,
                 return_tensors=None, **kwargs):
        matches = list(re.finditer(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b|[^\w\s]", text))
        ids = [self.vocabulary.get(match.group().lower(), self.unk_token_id) for match in matches]
        offsets = [(match.start(), match.end()) for match in matches]
        if add_special_tokens:
            ids = [self.cls_token_id, *ids, self.sep_token_id]
            offsets = [(0, 0), *offsets, (0, 0)]
        encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            self.offset_calls += 1
            encoded["offset_mapping"] = offsets
        if return_tensors == "pt":
            encoded = {key: torch.tensor([value]) for key, value in encoded.items()}
        return encoded


class TinyBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(128, 12)
        self.linear = torch.nn.Linear(12, 12)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, *, inputs_embeds, attention_mask, return_dict=True):
        mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        context = (inputs_embeds * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        positions = torch.arange(inputs_embeds.shape[1], dtype=inputs_embeds.dtype).reshape(1, -1, 1)
        return SimpleNamespace(last_hidden_state=torch.tanh(self.linear(inputs_embeds) + 0.2 * context + 0.01 * positions))


@pytest.fixture
def bundle():
    torch.manual_seed(37)
    body, tokenizer = TinyBody(), OffsetTokenizer()

    class Model:
        max_seq_length = 128

        def __init__(self):
            self.tokenizer = tokenizer

        def __getitem__(self, index):
            return SimpleNamespace(auto_model=body)

    class Encoder:
        identity = "tiny-canonical"

        def __init__(self):
            self.model = Model()

        def encode(self, texts):
            raise AssertionError("Proposer must not claim canonical verification or rank final text")

    encoder = Encoder()
    return SimpleNamespace(encoder=encoder, map_id="tiny-canonical-map", mean=np.zeros(12),
                           components=np.eye(12)[:2], records=[CorpusRecord("1", "research", "Tiny", "A useful model.")])


SOURCE = "A useful model compares measured observations with explicit assumptions."


def test_proposals_preserve_whole_word_offsets_and_reconstruct_text(bundle):
    proposer = CanonicalEditProposer(bundle)
    proposals = proposer.propose(SOURCE, (0.1, -0.2), max_proposals=24)
    assert proposals
    for proposal in proposals:
        spans = proposal.provenance["source_spans"]
        replacements = proposal.provenance["replacements"]
        reconstructed = SOURCE
        for (start, end), replacement in sorted(zip(spans, replacements), reverse=True):
            assert 0 <= start <= end <= len(SOURCE)
            assert start == 0 or not SOURCE[start - 1].isalnum()
            assert end == len(SOURCE) or end == start or SOURCE[end - 1].isspace() or not SOURCE[end].isalnum()
            reconstructed = reconstructed[:start] + replacement + reconstructed[end:]
        assert reconstructed == proposal.text
        assert proposal.edit_span == (min(span[0] for span in spans), max(span[1] for span in spans))
        assert "[CLS]" not in proposal.text and "[SEP]" not in proposal.text and "##" not in proposal.text
        assert proposal.provenance["canonical_verified"] is False


def test_gradients_use_normal_offsets_and_keep_parameters_frozen(bundle):
    reward = TorchEmbeddingReward(bundle.encoder, bundle)
    proposer = CanonicalEditProposer(bundle, reward=reward)
    proposals = proposer.propose(SOURCE, (0.1, -0.2), max_proposals=8)
    assert bundle.encoder.model.tokenizer.offset_calls == 1
    assert all(np.isfinite(p.provenance["predicted_map_delta"]).all() for p in proposals)
    assert all(p.provenance["source_gradient_norm"] > 0 for p in proposals)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in reward.body.parameters())
    proposer.hotspots(SOURCE, (0.1, -0.2))
    assert bundle.encoder.model.tokenizer.offset_calls == 1  # Reuse detached analysis.


def test_two_coordinate_prediction_is_actual_input_linearization(bundle):
    proposer = CanonicalEditProposer(bundle)
    analysis = proposer._analyze(SOURCE, (0.1, -0.2))
    word = next(word for word in analysis.words if word.text == "useful")
    delta, kind = proposer._predict(analysis, word.positions, "helpful")
    assert kind == "same-length-input-linearization"
    surface = proposer._surface("helpful")
    direction = np.zeros_like(analysis.inputs)
    direction[list(word.positions)] = surface[1] - analysis.inputs[list(word.positions)]
    inputs = torch.tensor(analysis.inputs[None], dtype=proposer.reward.dtype)
    direction = torch.tensor(direction[None], dtype=proposer.reward.dtype)
    mask = torch.ones(inputs.shape[:2], dtype=torch.long)
    epsilon = 1e-3
    with torch.no_grad():
        plus = proposer.reward.project(proposer.reward.embed_inputs(inputs + epsilon * direction, mask))[0]
        minus = proposer.reward.project(proposer.reward.embed_inputs(inputs - epsilon * direction, mask))[0]
    expected = ((plus - minus) / (2 * epsilon)).numpy()
    np.testing.assert_allclose(delta, expected, rtol=0.02, atol=5e-5)


def test_caps_token_sequence_dedup_and_zero_budget(bundle):
    proposer = CanonicalEditProposer(bundle, allowed_replacements=lambda word: (
        [word, word.upper(), "helpful", "helpful", "valuable", "madeupunknown"] if word.lower() == "useful" else []))
    assert proposer.propose(SOURCE, (0, 0), max_proposals=0) == []
    for cap in (1, 3, 12):
        proposals = proposer.propose(SOURCE, (0.1, -0.2), max_proposals=cap)
        assert len(proposals) <= cap
        ids = [tuple(proposer.tokenizer(p.text)["input_ids"]) for p in proposals]
        assert len(ids) == len(set(ids))
        assert tuple(proposer.tokenizer(SOURCE)["input_ids"]) not in ids
        assert all("madeupunknown" not in p.text for p in proposals)


def test_default_proposals_include_disjoint_pairs_and_bounded_structural_edits(bundle):
    proposer = CanonicalEditProposer(bundle)
    proposals = proposer.propose(SOURCE, (0.1, -0.2), max_proposals=64)
    assert any(p.provenance["kind"] == "disjoint-pair" for p in proposals)
    singles = proposer._single_edits(proposer._analyze(SOURCE, (0.1, -0.2)))
    assert any(edit.kind == "delete-optional-word" for edit in singles)
    assert any(edit.kind == "insert-modifier" for edit in singles)
    phrase_text = "A useful model can be used to compare measured observations in order to research assumptions."
    phrases = proposer._single_edits(proposer._analyze(phrase_text, (0.1, -0.2)))
    assert any(edit.kind == "replace-short-phrase" for edit in phrases)
    for proposal in proposals:
        if proposal.provenance["kind"] == "disjoint-pair":
            a, b = proposal.provenance["source_spans"]
            assert a[1] < b[0] or b[1] < a[0]


def test_hotspots_are_whole_words_and_limits_are_honored(bundle):
    proposer = CanonicalEditProposer(bundle)
    spans = proposer.hotspots(SOURCE, (0.1, -0.2), limit=4)
    assert len(spans) == len(set(spans)) == 4
    assert all(re.fullmatch(r"[A-Za-z]+", SOURCE[start:end]) for start, end in spans)
    assert proposer.hotspots(SOURCE, (0, 0), limit=0) == []
    with pytest.raises(ValueError, match="nonnegative"):
        proposer.propose(SOURCE, (0, 0), max_proposals=-1)
    with pytest.raises(ValueError, match="finite"):
        proposer.propose(SOURCE, (float("nan"), 0))


def test_custom_vocabulary_blocks_unapproved_replacement_words(bundle):
    proposer = CanonicalEditProposer(bundle, allowed_replacements=lambda word: ["helpful", "valuable"],
                                    vocabulary=["helpful"], max_pair_edits=0)
    proposals = proposer.propose("Useful observations.", (0.1, -0.2))
    assert proposals
    assert all("valuable" not in proposal.text.lower() for proposal in proposals)


def test_default_context_protects_technical_terms_and_noun_verb_ambiguity(bundle):
    proposer = CanonicalEditProposer(bundle)
    # Fixture vocabulary includes 'estimates' only via an explicit addition.
    tokenizer = bundle.encoder.model.tokenizer
    tokenizer.vocabulary = dict(tokenizer.vocabulary, gradient=70, estimates=71, local=72, minima=73)
    text = "A useful model compares gradient estimates with local minima."
    analysis = proposer._analyze(text, (0.1, -0.2))
    edits = proposer._single_edits(analysis)
    protected = [(text.index("gradient estimates"), text.index("gradient estimates") + len("gradient estimates")),
                 (text.index("local minima"), text.index("local minima") + len("local minima"))]
    assert all(not (start < edit.end and end > edit.start) for edit in edits for start, end in protected)
    estimates = next(word for word in analysis.words if word.text == "estimates")
    assert not proposer._default_context_ok(analysis, estimates, "quantifies")


def test_contextual_provider_receives_exact_span_and_empty_fails_closed(bundle):
    class Provider:
        def __init__(self):
            self.calls = []

        def replacements(self, text, span):
            self.calls.append((text, span))
            return ["helpful"] if text[slice(*span)].lower() == "useful" else []

    provider = Provider()
    proposer = CanonicalEditProposer(bundle, allowed_replacements=provider)
    proposals = proposer.propose(SOURCE, (0.1, -0.2))
    assert proposals and provider.calls
    assert all(p.text == SOURCE.replace("useful", "helpful") for p in proposals)
    assert all(p.provenance["lexical_policy"] == "contextual-provider" for p in proposals)
    assert not proposer.allow_insertions and not proposer.allow_deletions and not proposer.allow_phrases
    provider.replacements = lambda text, span: []
    empty = CanonicalEditProposer(bundle, allowed_replacements=provider)
    assert empty.propose(SOURCE, (0.1, -0.2)) == []


def test_structural_edit_flags_are_independent_of_word_replacements(bundle):
    proposer = CanonicalEditProposer(bundle, allow_insertions=False, allow_deletions=False, allow_phrases=False)
    edits = proposer._single_edits(proposer._analyze(SOURCE, (0.1, -0.2)))
    assert edits
    assert all(edit.kind == "replace-word" for edit in edits)


def test_conservative_native_candidates_intersect_family_and_protect_terms(bundle):
    tokenizer = bundle.encoder.model.tokenizer
    tokenizer.vocabulary = dict(tokenizer.vocabulary, ensure=80, stability=81,
                                gradient=82, estimates=83, quantifies=84, **{",": 85})
    text = "To ensure stability, clear assumptions require gradient estimates."

    class Provider:
        def __init__(self):
            self.sources = []

        def replacements(self, source, span):
            word = source[slice(*span)].lower()
            self.sources.append(word)
            return {"stability": ["model"], "estimates": ["quantifies"],
                    "clear": ["explicit", "model"]}.get(word, [])

    provider = Provider()
    proposer = CanonicalEditProposer(bundle, allowed_replacements=provider, max_hotspots=4)
    proposals = proposer.propose(text, (0.1, -0.2))
    assert proposals
    assert all(p.text == text.replace("clear", "explicit") for p in proposals)
    assert "stability" not in provider.sources  # No family: fail before native work.
    assert "estimates" not in provider.sources  # Protected collocation.
    assert all(p.provenance["conservative_contextual"] is True for p in proposals)


def test_contextual_opt_out_is_explicit_and_recorded(bundle):
    tokenizer = bundle.encoder.model.tokenizer
    tokenizer.vocabulary = dict(tokenizer.vocabulary, ensure=80, stability=81)

    class Provider:
        def replacements(self, text, span):
            return ["model"] if text[slice(*span)] == "stability" else []

    text = "To ensure stability."
    assert CanonicalEditProposer(bundle, allowed_replacements=Provider()).propose(text, (0, 0)) == []
    proposals = CanonicalEditProposer(bundle, allowed_replacements=Provider(),
                                      conservative_contextual=False).propose(text, (0, 0))
    assert len(proposals) == 1 and proposals[0].text == "To ensure model."
    assert proposals[0].provenance["conservative_contextual"] is False
