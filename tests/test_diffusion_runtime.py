"""Sampler-seam tests use real MLX tensors, without model downloads."""

import importlib
from threading import Event
from time import perf_counter
from types import SimpleNamespace

import pytest

from point2text.diffusion.contracts import DiffusionRequest
from point2text.diffusion.runtime import MLXDiffusionRuntime


mx = pytest.importorskip("mlx.core")
native = pytest.importorskip("mlx_vlm.generate.diffusion")


class TinyTokenizer:
    eos_token_id = 99

    def decode(self, ids, **kwargs):
        return " ".join(str(token_id) for token_id in ids)


class TinyModel:
    def __init__(self):
        self.config = SimpleNamespace(
            canvas_length=4,
            text_config=SimpleNamespace(vocab_size=8),
            generation_config={"t_min": 1.0, "t_max": 1.0},
        )
        self.self_conditioned = []

    def make_cache(self):
        return []

    def diffusion_prefill_cache(self, ids, **kwargs):
        return []

    def diffusion_prepare_self_conditioning(self):
        return None

    def diffusion_decoder_masks(self, *args):
        return None

    def diffusion_decoder_logits(self, canvas, **kwargs):
        return mx.broadcast_to(mx.array([0., 2., 0., 0., 0., 0., 0., 0.]), (1, 4, 8))

    def diffusion_self_conditioning(self, logits, context):
        self.self_conditioned.append(mx.argmax(logits, axis=-1).tolist())
        return mx.zeros((1, 4, 1))


def run_tiny(hook=None, sampler="entropy-bound", event=None):
    runtime = MLXDiffusionRuntime(sampler=sampler)
    runtime._model = TinyModel()
    runtime._tokenizer = TinyTokenizer()
    mx.random.seed(42)
    result = runtime._generate_canvas(
        DiffusionRequest("Explain a model", max_tokens=4, steps=4),
        mx.array([[1, 2, 3]]), 4, native, hook, event, perf_counter(),
    )
    return runtime, result


def force_token_three(step):
    return mx.where(mx.arange(step.logits.shape[-1]) == 3, 60., step.logits)


def test_identity_hook_preserves_seeded_tokens():
    _, plain = run_tiny()
    _, identity = run_tiny(lambda step: step.logits)
    assert plain.token_ids == identity.token_ids
    assert plain.text == identity.text
    assert identity.trace["hook_calls"] == identity.trace["passes"]
    assert all(step["hook_max_abs_delta"] == 0 for step in identity.trace["step_trace"])


@pytest.mark.parametrize("sampler", ["entropy-bound", "confidence-threshold"])
def test_hook_changes_sampler_input_and_final_decisions(monkeypatch, sampler):
    observed = []
    original = native._diffusion_sample_canvas

    def capture(logits, *args):
        observed.append(mx.argmax(logits, axis=-1).tolist())
        return original(logits, *args)

    monkeypatch.setattr(native, "_diffusion_sample_canvas", capture)
    runtime, result = run_tiny(force_token_three, sampler)
    assert result.token_ids == [3, 3, 3, 3]
    assert observed and all(row == [[3, 3, 3, 3]] for row in observed)
    if sampler == "entropy-bound":
        assert runtime._model.self_conditioned
        assert all(row == [[3, 3, 3, 3]] for row in runtime._model.self_conditioned)
    assert result.trace["step_trace"][0]["accepted"] == 4
    assert result.trace["step_trace"][0]["hook_max_abs_delta"] > 0


def test_hook_sees_native_shapes_and_zero_based_steps():
    seen = []

    def hook(step):
        seen.append(step.step)
        assert step.logits.shape == (1, 4, 8)
        assert step.token_ids.shape == step.editable_mask.shape == (1, 4)
        assert step.editable_mask.dtype == mx.bool_
        assert step.context_ids.shape == (1, 3)
        return step.logits

    run_tiny(hook)
    assert seen == [0, 1, 2, 3]


def test_invalid_hook_shape_is_rejected():
    with pytest.raises(ValueError, match="original logits shape"):
        run_tiny(lambda step: step.logits[:, :1])


def test_cancellation_does_not_publish_incomplete_canvas():
    event = Event()

    def cancel_after_first_pass(step):
        event.set()
        return step.logits

    _, result = run_tiny(cancel_after_first_pass, event=event)
    assert result.trace["cancelled"]
    assert result.trace["passes"] == result.trace["hook_calls"] == 1
    assert not result.text and not result.token_ids


def test_cancelled_before_load_needs_no_model():
    event = Event()
    event.set()
    result = MLXDiffusionRuntime().generate(DiffusionRequest("Hello"), cancel_event=event)
    assert result.trace["cancelled"] and result.trace["passes"] == 0
