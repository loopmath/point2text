"""Sampler-seam tests use real MLX tensors, without model downloads."""

import importlib
from threading import Event
from time import perf_counter
from types import SimpleNamespace

import pytest

from point2text.diffusion.contracts import DiffusionRequest, DiffusionResult
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


def run_tiny(hook=None, sampler="entropy-bound", event=None, **kwargs):
    runtime = MLXDiffusionRuntime(sampler=sampler)
    runtime._model = TinyModel()
    runtime._tokenizer = TinyTokenizer()
    mx.random.seed(42)
    result = runtime._generate_canvas(
        DiffusionRequest("Explain a model", max_tokens=4, steps=4),
        mx.array([[1, 2, 3]]), 4, native, hook, event, perf_counter(),
        **kwargs,
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


def test_passive_observer_preserves_tokens_and_rng_state():
    _, plain = run_tiny()
    next_plain = mx.random.uniform(shape=(8,)).tolist()
    drafts = []
    _, observed = run_tiny(observer=lambda draft: drafts.append(draft) or False, observe_every=3)
    next_observed = mx.random.uniform(shape=(8,)).tolist()
    assert observed.token_ids == plain.token_ids
    assert next_observed == next_plain
    assert [(draft.step, draft.terminal) for draft in drafts] == [(2, False), (3, True)]
    assert drafts[-1].text == observed.text
    assert drafts[-1].token_ids == observed.token_ids
    assert drafts[-1].canvas_token_ids == observed.trace["canvas_token_ids"]


def test_observer_can_stop_at_a_clean_draft():
    drafts = []
    _, result = run_tiny(observer=lambda draft: drafts.append(draft) or True, observe_every=1)
    assert result.trace["passes"] == 1
    assert result.trace["stopping_reason"] == "observer_requested"
    assert drafts[-1].terminal
    assert drafts[0].text == drafts[-1].text == result.text


@pytest.mark.parametrize("sampler", ["entropy-bound", "confidence-threshold"])
def test_single_position_repair_pins_every_other_id(sampler, monkeypatch):
    seen_inputs = []
    original = TinyModel.diffusion_decoder_logits

    def decoder(model, canvas, **kwargs):
        seen_inputs.append(canvas.tolist()[0])
        return original(model, canvas, **kwargs)

    monkeypatch.setattr(TinyModel, "diffusion_decoder_logits", decoder)
    drafts = []
    runtime, result = run_tiny(
        force_token_three, sampler=sampler,
        initial_token_ids=[5, 2, 6, 7], edit_mask=[False, True, False, False],
        observer=lambda draft: drafts.append(draft) or False, observe_every=1,
    )
    assert result.trace["canvas_token_ids"] == [5, 3, 6, 7]
    for ids in seen_inputs + [draft.canvas_token_ids for draft in drafts]:
        assert [ids[i] for i in (0, 2, 3)] == [5, 6, 7]
    assert all(ids == [[5, 3, 6, 7]] for ids in runtime._model.self_conditioned)
    assert result.trace["editable_positions"] == 1


def test_all_position_repair_matches_ordinary_generation_rng():
    _, plain = run_tiny()
    next_plain = mx.random.uniform(shape=(8,)).tolist()
    _, repaired = run_tiny(initial_token_ids=[5, 2, 6, 7], edit_mask=[True] * 4)
    next_repaired = mx.random.uniform(shape=(8,)).tolist()
    assert repaired.token_ids == plain.token_ids
    assert next_repaired == next_plain


def test_zero_position_repair_returns_identity_without_prefill(monkeypatch):
    def unexpected_prefill(*args, **kwargs):
        raise AssertionError("zero-position repair must not prefill")

    monkeypatch.setattr(TinyModel, "diffusion_prefill_cache", unexpected_prefill)
    drafts = []
    _, result = run_tiny(initial_token_ids=[5, 2, 6, 7], edit_mask=[False] * 4,
                         observer=lambda draft: drafts.append(draft) or False)
    assert result.token_ids == [5, 2, 6, 7]
    assert result.trace["passes"] == 0
    assert result.trace["stopping_reason"] == "no_editable_positions"
    assert drafts[0].terminal


def test_entropy_and_stability_only_receive_editable_positions(monkeypatch):
    transfer_shapes = []
    stability_shapes = []
    original_transfer = native._diffusion_entropy_transfer_mask
    original_stability = native._diffusion_stable_and_confident

    def transfer(entropy, *args):
        transfer_shapes.append(entropy.shape)
        return original_transfer(entropy, *args)

    def stability(ids, logits, *args):
        stability_shapes.append((ids.shape, logits.shape))
        return original_stability(ids, logits, *args)

    monkeypatch.setattr(native, "_diffusion_entropy_transfer_mask", transfer)
    monkeypatch.setattr(native, "_diffusion_stable_and_confident", stability)
    run_tiny(initial_token_ids=[5, 2, 6, 7], edit_mask=[False, True, False, False])
    assert transfer_shapes and all(shape == (1, 1) for shape in transfer_shapes)
    assert stability_shapes and all(shape == ((1, 1), (1, 1, 8)) for shape in stability_shapes)


def test_repair_observer_uses_same_eos_and_channel_cleanup(monkeypatch):
    def encode(tokenizer, text, **kwargs):
        return [0, 4] if text == "<|channel>thought\n<channel|>" else [7, 7]

    original_init = TinyModel.__init__

    def init(model):
        original_init(model)
        model.config.generation_config["eos_token_id"] = [7]

    monkeypatch.setattr(TinyTokenizer, "encode", encode, raising=False)
    monkeypatch.setattr(TinyModel, "__init__", init)
    drafts = []
    _, result = run_tiny(initial_token_ids=[0, 4, 1, 7], edit_mask=[False] * 4,
                         observer=lambda draft: drafts.append(draft) or False)
    assert result.token_ids == drafts[0].token_ids == [1]
    assert result.text == drafts[0].text == "1"
    assert drafts[0].canvas_token_ids == [0, 4, 1, 7]
    assert result.trace["channel_header"] == "thought"


@pytest.mark.parametrize("ids, mask", [([], []), ([1], []), ([1], [1]),
                         ([1.0], [True]), ([1], None), ([[1], [2]], [[True], [False]])])
def test_repair_input_validation(ids, mask):
    with pytest.raises(ValueError):
        MLXDiffusionRuntime._edit_inputs(ids, mask)


def test_each_repair_has_a_fresh_cache(monkeypatch):
    caches = []
    def cache(model):
        value = []
        caches.append(value)
        return value

    monkeypatch.setattr(TinyModel, "make_cache", cache)
    runtime = MLXDiffusionRuntime()
    runtime._model = TinyModel()
    runtime._tokenizer = TinyTokenizer()
    for _ in range(2):
        runtime._generate_canvas(
            DiffusionRequest("Explain", max_tokens=4, steps=4), mx.array([[1, 2]]),
            4, native, None, None, perf_counter(),
            initial_token_ids=[5, 2, 6, 7], edit_mask=[False, True, False, False],
        )
    assert len(caches) == 2 and caches[0] is not caches[1]


def test_refine_text_uses_unicode_offsets_and_reports_expanded_span(monkeypatch):
    class OffsetTokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs["return_offsets_mapping"] and not kwargs["truncation"]
            return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 1), (1, 3), (3, 5)]}

        def decode(self, ids, **kwargs):
            return "A α B"

    runtime = MLXDiffusionRuntime()
    runtime._tokenizer = OffsetTokenizer()
    captured = {}

    def refine(request, ids, mask, **kwargs):
        captured.update(ids=ids, mask=mask)
        return DiffusionResult("A α B", ids, {})

    monkeypatch.setattr(runtime, "refine", refine)
    result = runtime.refine_text(DiffusionRequest("Explain"), "A α B", (2, 3))
    assert captured == {"ids": [1, 2, 3], "mask": [False, True, False]}
    assert result.trace["expanded_edit_span"] == [1, 3]


def test_partial_repair_cannot_introduce_eos_before_protected_suffix(monkeypatch):
    original_init = TinyModel.__init__

    def init(model):
        original_init(model)
        model.config.generation_config["eos_token_id"] = [3]

    monkeypatch.setattr(TinyModel, "__init__", init)
    _, result = run_tiny(force_token_three, initial_token_ids=[5, 2, 6, 7],
                         edit_mask=[False, True, False, False])
    assert result.trace["canvas_token_ids"][1] != 3
    assert len(result.token_ids) == 4
    assert result.token_ids[2:] == [6, 7]


def test_observer_cancellation_keeps_no_unfinished_final_canvas():
    event = Event()

    def observer(draft):
        event.set()
        return False

    _, result = run_tiny(observer=observer, observe_every=1, event=event)
    assert result.trace["cancelled"]
    assert result.trace["observer_calls"] == 1
    assert result.trace["passes"] == 1
    assert not result.text
