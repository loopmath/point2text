import numpy as np
import pytest

torch = pytest.importorskip("torch")

from point2text.diffusion.bridge import SparseTokenBridge


class EncoderTokenizer:
    unk_token_id = 1
    mask_token_id = 2

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return {"input_ids": {"research": [3], "multi piece": [4, 5], "unknown": [1]}[text]}


class GeneratorTokenizer:
    all_special_ids = [0, 9]
    mask_token_id = 9

    def __init__(self):
        self.calls = []

    def decode(self, ids, **kwargs):
        self.calls.extend(ids)
        return {0: "<eos>", 3: "research", 4: "multi piece", 5: " ", 6: "", 7: "unknown", 9: "<mask>"}[ids[0]]


def test_one_to_many_averages_real_embeddings_and_caches_selected_ids():
    generator = GeneratorTokenizer()
    layer = torch.nn.Embedding(10, 4)
    bridge = SparseTokenBridge(generator, EncoderTokenizer(), layer)
    vectors, present, metadata = bridge.lookup(np.asarray([[3, 4, 3], [5, 7, 0]]))
    torch.testing.assert_close(vectors[0, 1], layer.weight[[4, 5]].mean(dim=0))
    assert vectors.requires_grad is False
    assert len(generator.calls) == len(set(generator.calls)) == 5
    bridge.lookup(np.asarray([3, 4]))
    assert len(generator.calls) == 5
    assert present.tolist() == [[True, True, True], [False, True, False]]
    assert metadata["encoder_token_count"][0, 1] == 2
    assert metadata["unknown_fraction"][1, 1] == 1
    assert metadata["empty"][1, 0]
    assert metadata["special"][1, 2]


def test_mask_placeholder_uses_encoder_mask_without_zero_filling():
    layer = torch.nn.Embedding(10, 4)
    bridge = SparseTokenBridge(GeneratorTokenizer(), EncoderTokenizer(), layer)
    vectors, present, metadata = bridge.lookup(np.asarray([9, 6]))
    torch.testing.assert_close(vectors[0], layer.weight[2])
    torch.testing.assert_close(vectors[1], layer.weight[1])
    assert present.tolist() == [True, False]
    assert metadata["mask"].tolist() == [True, False]


def test_bridge_rejects_noninteger_or_negative_ids():
    bridge = SparseTokenBridge(GeneratorTokenizer(), EncoderTokenizer(), torch.nn.Embedding(10, 4))
    with pytest.raises(ValueError, match="nonnegative integers"):
        bridge.lookup(np.asarray([-1]))
    with pytest.raises(ValueError, match="nonnegative integers"):
        bridge.lookup(np.asarray([1.2]))
