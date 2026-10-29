"""A small, hookable DiffusionGemma runtime on Apple Silicon.

The sampler uses mlx-vlm 0.7.11's native helpers and model cache/mask methods.
Only the text-only, single-canvas loop lives here so the guidance seam is
explicit: after native temperature shaping, before every token decision.
MLX and model weights are optional and imported/loaded only on first use.
The native sampler adaptation retains the upstream MIT notice in NOTICE.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from threading import Event, RLock
from time import perf_counter
from typing import Any

from .contracts import DiffusionRequest, DiffusionResult, DiffusionStep, GuidanceHook


DEFAULT_MODEL_ID = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
DEFAULT_REVISION = "a7a81407613811e8ba63af92ac0d852b809e191f"
MLX_VLM_VERSION = "0.7.11"
# MLX's global RNG and Metal stream are shared between runtime instances.
_GENERATION_LOCK = RLock()


class MLXDiffusionRuntime:
    """Frozen DiffusionGemma with an editable, native-tensor sampler seam.

    ``hook`` receives float logits [1, canvas_length, vocabulary], token IDs
    and boolean editable masks [1, canvas_length]. It returns an MLX array of
    the same shape. Step numbers start at zero. Hooks must not consume MLX's
    global random state. Native entropy-bound sampling can revisit all token
    positions, hence its editable mask is all true.

    This first implementation deliberately supports at most one 256-token
    canvas. It does not silently substitute autoregressive generation.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        *,
        sampler: str = "entropy-bound",
        confidence_threshold: float = 0.9,
    ) -> None:
        if sampler not in {"entropy-bound", "confidence-threshold"}:
            raise ValueError("sampler must be entropy-bound or confidence-threshold")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.model_id = model_id
        self.revision = revision or (DEFAULT_REVISION if model_id == DEFAULT_MODEL_ID else "main")
        self.sampler = sampler
        self.confidence_threshold = confidence_threshold
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None

    @property
    def identity(self) -> str:
        return f"mlx-vlm/{MLX_VLM_VERSION}:{self.model_id}@{self.revision}:{self.sampler}"

    @property
    def tokenizer(self) -> Any:
        """Load tokenizer files on first use, without loading model weights."""
        with _GENERATION_LOCK:
            if self._tokenizer is None:
                try:
                    from transformers import AutoTokenizer
                except ImportError as exc:
                    raise RuntimeError(self._install_hint()) from exc
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, revision=self.revision
                )
            return self._tokenizer

    @staticmethod
    def _install_hint() -> str:
        return (
            "Diffusion requires Apple Silicon and the optional runtime: "
            f"pip install mlx-vlm=={MLX_VLM_VERSION}"
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            installed = version("mlx-vlm")
        except PackageNotFoundError as exc:
            raise RuntimeError(self._install_hint()) from exc
        if installed != MLX_VLM_VERSION:
            raise RuntimeError(
                f"This sampler adapter is audited against mlx-vlm=={MLX_VLM_VERSION}; "
                f"found {installed}. Install the pinned version."
            )
        try:
            from mlx_vlm import load
        except ImportError as exc:
            raise RuntimeError(self._install_hint()) from exc
        model, processor = load(self.model_id, revision=self.revision)
        if getattr(model.config, "model_type", None) != "diffusion_gemma":
            raise ValueError("MLXDiffusionRuntime requires a diffusion_gemma checkpoint")
        model.eval()
        model.freeze()
        self._model = model
        self._processor = processor
        self._tokenizer = getattr(processor, "tokenizer", processor)

    def _prepare_prompt(self, prompt: str) -> Any:
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        formatted = apply_chat_template(
            self._processor, self._model.config, prompt, enable_thinking=False
        )
        inputs = prepare_inputs(
            self._processor,
            prompts=formatted,
            add_special_tokens=should_add_special_tokens(
                self._model.config.model_type, self._processor
            ),
        )
        return inputs["input_ids"]

    def generate(
        self,
        request: DiffusionRequest,
        *,
        hook: GuidanceHook | None = None,
        cancel_event: Event | None = None,
    ) -> DiffusionResult:
        """Generate real text; cancellation returns only a completed canvas.

        An interrupted denoising canvas is not presented as completed text.
        The trace always records cancellation, passes and native hook calls.
        """
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if not 1 <= request.max_tokens <= 256:
            raise ValueError("single-canvas max_tokens must be in [1, 256]")
        if not 1 <= request.steps <= 256:
            raise ValueError("steps must be in [1, 256]")
        if not 0 <= request.temperature <= 2:
            raise ValueError("temperature must be in [0, 2]")
        started = perf_counter()
        with _GENERATION_LOCK:
            if cancel_event is not None and cancel_event.is_set():
                return self._cancelled_result(started)
            self._ensure_loaded()
            import mlx.core as mx
            from mlx_vlm.generate.common import generation_stream, wired_limit

            native = importlib.import_module("mlx_vlm.generate.diffusion")
            model = self._model
            tokenizer = self.tokenizer
            context_ids = self._prepare_prompt(request.prompt)
            if context_ids.ndim != 2 or context_ids.shape[0] != 1:
                raise ValueError("Only one text prompt at a time is supported")
            canvas_length = min(request.max_tokens, int(model.config.canvas_length))
            mx.random.seed(request.seed)
            mx.reset_peak_memory()
            try:
                with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
                    result = self._generate_canvas(
                        request, context_ids, canvas_length, native, hook,
                        cancel_event, started,
                    )
                result.trace["peak_memory_bytes"] = int(mx.get_peak_memory())
                return result
            finally:
                # Requests own their KV cache and graph. Keep frozen weights.
                mx.synchronize()
                mx.clear_cache()

    def _cancelled_result(self, started: float, **trace: Any) -> DiffusionResult:
        return DiffusionResult("", [], {
            "runtime": self.identity,
            "model_revision": self.revision,
            "cancelled": True,
            "passes": 0,
            "hook_calls": 0,
            "hook_count": 0,
            "elapsed_seconds": perf_counter() - started,
            **trace,
        })

    def _strip_channel_header(self, ids: list[int]) -> tuple[list[int], str | None]:
        """Remove only an explicit leading Gemma channel metadata header.

        ``skip_special_tokens`` removes delimiter IDs but leaves the word
        "thought" behind. Strip the complete known delimiter sequence before
        decoding so that metadata is not scored as paragraph content.
        """
        encode = getattr(self.tokenizer, "encode", None)
        if encode is not None:
            for channel in ("thought", "analysis", "final"):
                prefix = encode(
                    f"<|channel>{channel}\n<channel|>", add_special_tokens=False
                )
                if prefix and ids[:len(prefix)] == prefix:
                    return ids[len(prefix):], channel
        return ids, None

    def _generate_canvas(
        self, request: DiffusionRequest, context_ids: Any, canvas_length: int,
        native: Any, hook: GuidanceHook | None, cancel_event: Event | None,
        started: float,
    ) -> DiffusionResult:
        import mlx.core as mx

        model = self._model
        config = model.config.generation_config or {}
        schedule = config.get("linear_temperature_schedule_config") or {
            "t_min": config.get("t_min", 0.4), "t_max": config.get("t_max", 0.8)
        }
        stopping = config.get("diffusion_stopping_config") or {
            key: config[key] for key in ("confidence_threshold", "stability_threshold")
            if key in config
        } or None
        sampler_config = config.get("sampler_config") or {}
        entropy_bound = float(sampler_config.get("entropy_bound", 0.1))
        vocab_size = int(model.config.text_config.vocab_size)
        prefill_started = perf_counter()
        cache = model.make_cache()
        # No padding: one text-only prompt, native dynamic hybrid KV cache.
        cache = model.diffusion_prefill_cache(context_ids, cache=cache)
        mx.eval([entry.state for entry in cache])
        prefill_seconds = perf_counter() - prefill_started
        current_canvas = native._diffusion_initial_canvas(
            None, 0, 1, canvas_length, vocab_size, context_ids.dtype
        )
        revealed = mx.zeros(current_canvas.shape, dtype=mx.bool_)
        draft_canvas = current_canvas
        self_conditioning = None
        self_conditioning_context = model.diffusion_prepare_self_conditioning()
        masks = model.diffusion_decoder_masks(current_canvas, cache, None)
        decoder_without_sc, decoder_with_sc = native._make_diffusion_decoder_logits_fns(
            model, cache, masks, compile_graph=False
        )
        history: list[Any] = []
        step_trace: list[dict[str, Any]] = []
        hook_calls = 0
        stopping_reason = "step_budget"
        argmax_canvas = current_canvas
        for index, remaining in enumerate(reversed(range(1, request.steps + 1))):
            if cancel_event is not None and cancel_event.is_set():
                return self._cancelled_result(
                    started, passes=index, hook_calls=hook_calls,
                    hook_count=hook_calls, step_trace=step_trace,
                    prefill_seconds=prefill_seconds,
                )
            pass_started = perf_counter()
            logits = decoder_without_sc(current_canvas) if self_conditioning is None else decoder_with_sc(current_canvas, self_conditioning)
            temperature = native._diffusion_linear_temperature(remaining, request.steps, schedule)
            if temperature is not None:
                logits = logits / temperature
            logits = mx.stop_gradient(logits)
            editable = mx.ones(current_canvas.shape, dtype=mx.bool_) if self.sampler == "entropy-bound" else ~revealed
            delta = 0.0
            if hook is not None:
                guided = hook(DiffusionStep(
                    logits=logits, token_ids=current_canvas, editable_mask=editable,
                    step=index, num_steps=request.steps, tokenizer=self.tokenizer,
                    context_ids=context_ids,
                ))
                if not isinstance(guided, mx.array) or guided.shape != logits.shape:
                    raise ValueError("guidance hook must return an MLX array with the original logits shape")
                if not bool(mx.all(mx.isfinite(guided)).item()):
                    raise ValueError("guidance hook returned nonfinite logits")
                delta = float(mx.max(mx.abs(guided.astype(mx.float32) - logits.astype(mx.float32))).item())
                logits = mx.stop_gradient(guided)
                hook_calls += 1

            # Every decision below consumes the SAME post-hook logits.
            argmax_canvas = mx.argmax(logits, axis=-1).astype(context_ids.dtype)
            if remaining == 1:
                mx.eval(argmax_canvas)
                step_trace.append({"step": index, "hook_max_abs_delta": delta, "accepted": canvas_length, "elapsed_seconds": perf_counter() - pass_started})
                break
            sampled = argmax_canvas if request.temperature <= 0 else native._diffusion_sample_canvas(logits, context_ids.dtype, request.temperature)
            next_sc = None
            if self.sampler == "entropy-bound":
                entropy = native._diffusion_token_entropy(logits)
                next_sc = model.diffusion_self_conditioning(logits, self_conditioning_context)
                acceptance = native._diffusion_entropy_transfer_mask(entropy, entropy_bound)
                accepted = mx.where(acceptance, sampled, current_canvas)
                current_canvas = mx.where(
                    acceptance, accepted,
                    native._diffusion_initialize_canvas(1, canvas_length, vocab_size, context_ids.dtype),
                )
                revealed = acceptance
                draft_canvas = argmax_canvas
            else:
                confidence = native._diffusion_token_probability(logits, sampled)
                acceptance = native._diffusion_confidence_transfer_mask(
                    confidence, ~revealed, self.confidence_threshold,
                )
                accepted = mx.where(acceptance, sampled, draft_canvas)
                current_canvas = mx.where(
                    revealed | acceptance, accepted,
                    native._diffusion_initialize_canvas(1, canvas_length, vocab_size, context_ids.dtype),
                )
                revealed = revealed | acceptance
                draft_canvas = mx.where(acceptance, accepted, draft_canvas)
            mx.eval(current_canvas, argmax_canvas, revealed)
            step_trace.append({"step": index, "hook_max_abs_delta": delta, "accepted": int(mx.sum(acceptance).item()), "elapsed_seconds": perf_counter() - pass_started})
            if self.sampler == "confidence-threshold" and bool(mx.all(revealed).item()):
                stopping_reason = "all_confident"
                break
            if native._diffusion_stable_and_confident(argmax_canvas, logits, history, stopping):
                stopping_reason = "stable_and_confident"
                break
            if next_sc is None:
                next_sc = model.diffusion_self_conditioning(logits, self_conditioning_context)
            self_conditioning = next_sc

        mx.eval(argmax_canvas)
        ids = [int(token_id) for token_id in argmax_canvas[0].tolist()]
        eos = config.get("eos_token_id", getattr(self.tokenizer, "eos_token_id", None))
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        for index, token_id in enumerate(ids):
            if token_id in eos_ids:
                ids = ids[:index]
                break
        raw_token_count = len(ids)
        ids, channel_header = self._strip_channel_header(ids)
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return DiffusionResult(text, ids, {
            "runtime": self.identity,
            "model_id": self.model_id,
            "model_revision": self.revision,
            "mlx_vlm_version": MLX_VLM_VERSION,
            "sampler": self.sampler,
            "native_schedule": schedule,
            "cancelled": False,
            "seed": request.seed,
            "canvas_length": canvas_length,
            "context_tokens": int(context_ids.size),
            "raw_token_count": raw_token_count,
            "channel_header": channel_header,
            "requested_steps": request.steps,
            "passes": len(step_trace),
            "hook_calls": hook_calls,
            "hook_count": hook_calls,
            "stopping_reason": stopping_reason,
            "prefill_seconds": prefill_seconds,
            "elapsed_seconds": perf_counter() - started,
            "step_trace": step_trace,
        })
