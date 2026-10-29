import json

import numpy as np
import pytest

from point2text.diffusion.contracts import DiffusionStep
from point2text.diffusion.guidance import GuidanceConfig, SparseMapGuidance


class TokenPreferenceReward:
    """A differentiable categorical toy reward with an independently known optimum."""
    def __init__(self, desired=1):
        self.desired = desired
        self.calls = []

    def loss_and_gradient(self, logits, ids, tokens, editable, tokenizer, target,
                          **kwargs):
        self.calls.append((logits.copy(), ids.copy(), kwargs))
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradients = probabilities.copy()
        loss = 0.0
        for row in np.flatnonzero(editable):
            desired = int(np.flatnonzero(ids[row] == self.desired)[0])
            loss -= np.log(probabilities[row, desired])
            gradients[row, desired] -= 1
        return loss, gradients, {"soft_map_location": [0.1, 0.2]}


def config(**kwargs):
    return GuidanceConfig(top_k=2, guided_steps=1, start_fraction=0,
                          end_fraction=1, **kwargs)


def step(logits=None, editable=None, **kwargs):
    if logits is None:
        logits = np.array([[[3.0, 2.5, 0.0, -1.0],
                            [1.0, 0.8, -2.0, -3.0]]], dtype=np.float32)
    if editable is None:
        editable = np.array([[False, True]])
    return DiffusionStep(logits=logits, token_ids=np.array([[0, 0]]),
                         editable_mask=editable, tokenizer=object(),
                         step=kwargs.get("step_index", 0),
                         num_steps=kwargs.get("num_steps", 1))


def test_guidance_changes_native_token_decision_and_preserves_committed_values():
    initial = step()
    before = initial.logits.copy()
    hook = SparseMapGuidance(TokenPreferenceReward(), (0, 0),
                              config(step_size=2, normalize_gradient=False))
    result = hook(initial)
    assert result.shape == before.shape and result.dtype == before.dtype
    assert np.argmax(before[0, 1]) == 0
    assert np.argmax(result[0, 1]) == 1
    np.testing.assert_array_equal(result[0, 0], before[0, 0])
    np.testing.assert_array_equal(result[0, :, 2:], before[0, :, 2:])
    np.testing.assert_array_equal(initial.logits, before)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result - before)) <= 1.0
    assert hook.trace["steps"][0]["changed_logits"] > 0
    json.dumps(hook.trace, allow_nan=False)


def test_zero_strength_returns_exact_object_without_calling_reward():
    class FailingReward:
        def loss_and_gradient(self, *args, **kwargs):
            pytest.fail("Identity guidance must not evaluate the reward")

    initial = step()
    hook = SparseMapGuidance(FailingReward(), (0, 0), config(strength=0))
    assert hook(initial) is initial.logits
    assert hook.trace["guided_step_count"] == 0


def test_no_editable_positions_is_identity():
    initial = step(editable=np.array([[False, False]]))
    reward = TokenPreferenceReward()
    assert SparseMapGuidance(reward, (0, 0), config())(initial) is initial.logits
    assert reward.calls == []


def test_schedule_is_sparse_and_kl_reference_stays_at_native_proposal():
    reward = TokenPreferenceReward()
    hook = SparseMapGuidance(reward, (0, 0), config(inner_updates=2))
    initial = step()
    hook(initial)
    assert len(reward.calls) == 2
    first, second = reward.calls
    np.testing.assert_array_equal(first[2]["reference_logits"],
                                  second[2]["reference_logits"])
    assert not np.array_equal(first[0], second[0])
    assert 0 < first[2]["retained_mass"][1] < 1
    scheduled = SparseMapGuidance(reward, (0, 0), GuidanceConfig(guided_steps=4))
    assert sum(scheduled._due(i, 32) for i in range(32)) == 4


@pytest.mark.parametrize("initial,error", [
    (step(editable=np.array([[0, 1]])), "boolean"),
    (step(editable=np.array([True, False])), "shape"),
    (step(logits=np.zeros((2, 2, 4))), "logits shaped"),
    (step(step_index=2), "step index"),
])
def test_invalid_masks_and_shapes_fail_before_reward(initial, error):
    with pytest.raises(ValueError, match=error):
        SparseMapGuidance(TokenPreferenceReward(), (0, 0), config())(initial)


def test_canvas_budget_and_nonfinite_gradient_are_rejected():
    with pytest.raises(ValueError, match="canvas positions"):
        SparseMapGuidance(TokenPreferenceReward(), (0, 0), config(max_positions=1))(step())

    class BrokenReward:
        def loss_and_gradient(self, logits, *args, **kwargs):
            return 0.0, np.full_like(logits, np.nan), {}

    with pytest.raises(ValueError, match="finite"):
        SparseMapGuidance(BrokenReward(), (0, 0), config())(step())


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"strength": -1},
                                   {"top_k": 0}, {"max_delta": float("nan")},
                                   {"start_fraction": 0.9, "end_fraction": 0.2}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GuidanceConfig(**kwargs)


def test_native_mlx_hook_preserves_dtype_and_fixed_positions():
    mx = pytest.importorskip("mlx.core")
    initial = step()
    native = DiffusionStep(logits=mx.array(initial.logits),
                           token_ids=mx.array(initial.token_ids),
                           editable_mask=mx.array(initial.editable_mask),
                           step=0, num_steps=1, tokenizer=object())
    result = SparseMapGuidance(TokenPreferenceReward(), (0, 0), config())(native)
    assert result.dtype == native.logits.dtype
    result_array = np.asarray(result)
    assert result_array[0, 1].argmax() == 1
    np.testing.assert_array_equal(result_array[0, 0], initial.logits[0, 0])
