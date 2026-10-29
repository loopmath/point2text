"""Tiny differentiability checks; real cached-model parity is explicitly opt-in."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from point2text.diffusion.encoder import TorchEmbeddingReward
from point2text.geometry import MapBundle


class TinyTokenizer:
    pad_token_id, cls_token_id, sep_token_id, unk_token_id, mask_token_id = range(5)
    all_special_ids = [0, 1, 2, 3, 4]
    words = {"alpha": 5, "beta": 6, "gamma": 7, "delta": 8, "multi": 9, "word": 10}
    surfaces = {0: "", 1: "<bos>", 2: "<eos>", 4: "<mask>", 5: "alpha", 6: "beta",
                7: "gamma", 8: "delta", 9: "multi word", 10: " ", 11: "unseen"}

    def __init__(self):
        self.decoded = []

    def decode(self, ids, **kwargs):
        self.decoded.extend(ids)
        return "".join(self.surfaces.get(int(i), "unseen") for i in ids)

    def __call__(self, text, *, add_special_tokens=True, **kwargs):
        ids = [self.words.get(word, self.unk_token_id) for word in text.lower().split()]
        if add_special_tokens:
            ids = [self.cls_token_id, *ids, self.sep_token_id]
        return {"input_ids": ids}


class TinyBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(16, 8)
        self.projection = torch.nn.Linear(8, 8)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, *, inputs_embeds, attention_mask, return_dict=True):
        mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        context = (inputs_embeds * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        position = torch.arange(inputs_embeds.shape[1], dtype=inputs_embeds.dtype).reshape(1, -1, 1)
        hidden = torch.tanh(self.projection(inputs_embeds) + 0.2 * context + position * 0.01)
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def reward():
    torch.manual_seed(13)
    body = TinyBody()
    tokenizer = TinyTokenizer()

    class TinySentenceModel:
        max_seq_length = 16

        def __init__(self):
            self.tokenizer = tokenizer

        def __getitem__(self, key):
            assert key == 0
            return SimpleNamespace(auto_model=body)

    encoder = SimpleNamespace(identity="tiny-frozen-encoder", model=TinySentenceModel())
    bundle = SimpleNamespace(encoder=encoder, mean=np.zeros(8),
                             components=np.eye(8)[:2], map_id="tiny-map")
    return TorchEmbeddingReward(encoder, bundle)


def proposal():
    logits = np.asarray([[0.4, -0.2, 0.1], [-0.1, 0.3, 0.5], [0.8, -0.4, 0.2]], dtype=np.float32)
    ids = np.asarray([[5, 6, 7], [6, 7, 8], [7, 8, 5]])
    current = np.asarray([5, 6, 7])
    editable = np.asarray([True, False, True])
    return logits, ids, current, editable


def test_input_gradients_are_finite_and_weights_frozen(reward):
    ids = torch.tensor([[1, 5, 6, 2]])
    inputs = reward.word_embeddings(ids).detach().requires_grad_(True)
    embedding = reward.embed_inputs(inputs, torch.ones_like(ids))
    loss = reward.map_loss(embedding, (0.3, -0.4))
    gradient, = torch.autograd.grad(loss, inputs)
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0
    assert all(not parameter.requires_grad for parameter in reward.body.parameters())
    assert all(parameter.grad is None for parameter in reward.body.parameters())
    torch.testing.assert_close(embedding.norm(dim=-1), torch.ones(1))


def test_sparse_logit_gradient_matches_directional_finite_difference(reward):
    logits, ids, current, editable = proposal()
    reference = logits - np.asarray([[0.1, 0.0, -0.1]], dtype=np.float32)
    kwargs = dict(temperature=0.9, kl_weight=0.15, reference_logits=reference)
    tokenizer = TinyTokenizer()
    loss, gradient, diagnostics = reward.loss_and_gradient(
        logits, ids, current, editable, tokenizer, (0.3, -0.4), **kwargs)
    direction = np.asarray([[0.3, -0.2, 0.1], [0.0, 0.0, 0.0], [-0.4, 0.1, 0.3]], dtype=np.float32)
    epsilon = 1e-3
    plus, _, _ = reward.loss_and_gradient(logits + epsilon * direction, ids, current, editable,
                                         tokenizer, (0.3, -0.4), **kwargs)
    minus, _, _ = reward.loss_and_gradient(logits - epsilon * direction, ids, current, editable,
                                          tokenizer, (0.3, -0.4), **kwargs)
    expected = (plus - minus) / (2 * epsilon)
    actual = float(np.sum(gradient * direction))
    assert actual == pytest.approx(expected, rel=0.03, abs=5e-5)
    assert np.isfinite(loss) and np.isfinite(gradient).all()
    assert np.linalg.norm(gradient[editable]) > 0
    np.testing.assert_array_equal(gradient[~editable], 0)
    assert diagnostics["bridge_coverage"] == pytest.approx(1.0)
    assert diagnostics["topk_retained_mass"] is None
    assert diagnostics["bridge_exact"] is False


def test_small_step_reduces_soft_objective(reward):
    logits, ids, current, editable = proposal()
    tokenizer = TinyTokenizer()
    loss, gradient, _ = reward.loss_and_gradient(logits, ids, current, editable, tokenizer, (0.3, -0.4))
    next_loss, _, _ = reward.loss_and_gradient(logits - 0.1 * gradient, ids, current, editable,
                                              tokenizer, (0.3, -0.4))
    assert next_loss < loss


def test_padding_is_excluded_from_encoder_mean_pooling(reward):
    compact = reward.embed_encoder_tokens(torch.tensor([[1, 5, 6, 2]]))
    padded = reward.embed_encoder_tokens(torch.tensor([[1, 5, 6, 2, 0, 0]]))
    torch.testing.assert_close(compact, padded)


def test_special_empty_unknown_and_retained_mass_are_reported(reward):
    logits = np.zeros((2, 3), dtype=np.float32)
    ids = np.asarray([[5, 10, 2], [9, 11, 6]])
    loss, gradient, diagnostics = reward.loss_and_gradient(
        logits, ids, np.asarray([4, 5]), np.asarray([True, True]), TinyTokenizer(),
        (0.1, 0.2), retained_mass=np.asarray([0.8, 0.6]))
    assert np.isfinite(loss) and np.isfinite(gradient).all()
    assert diagnostics["special_mass"] > 0
    assert diagnostics["empty_surface_mass"] > 0
    assert diagnostics["unknown_mass"] > 0
    assert diagnostics["one_to_many_mass"] > 0
    assert diagnostics["topk_retained_mass"] == pytest.approx(0.7)


def test_shape_budget_and_identity_errors_are_clear(reward):
    with pytest.raises(ValueError, match="maximum"):
        reward.loss_and_gradient(np.zeros((15, 2)), np.full((15, 2), 5), np.full(15, 5),
                                  np.ones(15, dtype=bool), TinyTokenizer(), (0.0, 0.0))
    logits, ids, current, editable = proposal()
    with pytest.raises(ValueError, match="temperature"):
        reward.loss_and_gradient(logits, ids, current, editable, TinyTokenizer(), (0, 0), temperature=0)


@pytest.mark.skipif(os.environ.get("POINT2TEXT_REAL_ENCODER_TEST") != "1",
                    reason="Opt in to cached pinned MiniLM parity; ordinary tests never download models")
def test_real_minilm_canonical_token_parity(tmp_path):
    from point2text.encoder import SentenceEncoder

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    encoder = SentenceEncoder()
    bundle = MapBundle.build(encoder, tmp_path)
    reward = TorchEmbeddingReward(encoder, bundle)
    texts = ["Neural representations connect observations with useful features.",
             "A replicated service compares message histories after a node restarts.",
             "Protein folding, quantum circuits, and a short Unicode check: café."]
    encoded = encoder.model.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    normal = encoder.encode(texts)
    with torch.no_grad():
        differentiable = reward.embed_encoder_tokens(encoded["input_ids"], encoded["attention_mask"]).cpu().numpy()
    np.testing.assert_allclose(differentiable, normal, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(reward.project(torch.tensor(differentiable)).numpy(),
                               bundle.project(normal), rtol=2e-5, atol=2e-6)
    assert max(encoder.token_lengths([record.text for record in bundle.records])) <= 256
