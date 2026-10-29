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
from numbers import Integral
import re
from threading import Event, RLock
from time import perf_counter
from typing import Any

from .contracts import (
    DiffusionDraft, DiffusionRequest, DiffusionResult, DiffusionStep,
    DraftObserver, GuidanceHook,
)


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
        observer: DraftObserver | None = None,
        observe_every: int = 3,
    ) -> DiffusionResult:
        """Generate real text; cancellation returns only a completed canvas.

        An interrupted denoising canvas is not presented as completed text.
        The trace always records cancellation, passes and native hook calls.
        """
        return self._run(
            request, hook=hook, cancel_event=cancel_event, observer=observer,
            observe_every=observe_every,
        )

    @staticmethod
    def _edit_inputs(initial_token_ids: Any, edit_mask: Any) -> tuple[list[int], list[bool]]:
        def row(value: Any, name: str) -> list[Any]:
            try:
                value = value.tolist() if hasattr(value, "tolist") else list(value)
            except TypeError as exc:
                raise ValueError(f"{name} must be a token row") from exc
            if value and isinstance(value[0], (list, tuple)):
                if len(value) != 1:
                    raise ValueError(f"{name} must have shape [positions] or [1, positions]")
                value = list(value[0])
            return value

        ids, editable = row(initial_token_ids, "initial_token_ids"), row(edit_mask, "edit_mask")
        if not 1 <= len(ids) <= 256:
            raise ValueError("repair requires 1–256 initial token IDs")
        if len(editable) != len(ids) or any(type(value) is not bool for value in editable):
            raise ValueError("edit_mask must contain one boolean per initial token")
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in ids):
            raise ValueError("initial_token_ids must contain nonnegative integer IDs")
        return [int(value) for value in ids], editable

    def refine(
        self, request: DiffusionRequest, initial_token_ids: Any, edit_mask: Any,
        *, hook: GuidanceHook | None = None, cancel_event: Event | None = None,
        observer: DraftObserver | None = None, observe_every: int = 3,
    ) -> DiffusionResult:
        """Uniformly corrupt selected positions, then repair a fixed-length canvas.

        Protected IDs remain bit-exact. The canvas length is the supplied ID
        count (1–256), rather than padding/truncating it to request.max_tokens.
        Every repair owns a fresh prompt cache and self-conditioning state.
        """
        ids, editable = self._edit_inputs(initial_token_ids, edit_mask)
        return self._run(
            request, hook=hook, cancel_event=cancel_event, observer=observer,
            observe_every=observe_every, initial_token_ids=ids, edit_mask=editable,
        )

    def refine_text(
        self, request: DiffusionRequest, text: str, edit_span: tuple[int, int],
        *, hook: GuidanceHook | None = None, cancel_event: Event | None = None,
        observer: DraftObserver | None = None, observe_every: int = 3,
    ) -> DiffusionResult:
        """Repair tokens overlapping a Unicode character span [start, end).

        Token boundaries can expand the selected span; trace records the actual
        expanded span. Text must round-trip exactly, so normalization outside
        the chosen span is never silently accepted.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a nonempty string")
        if (not isinstance(edit_span, (tuple, list)) or len(edit_span) != 2
                or any(isinstance(value, bool) or not isinstance(value, Integral) for value in edit_span)):
            raise ValueError("edit_span must contain two integer character offsets")
        start, end = map(int, edit_span)
        if not 0 <= start <= end <= len(text):
            raise ValueError("edit_span must satisfy 0 <= start <= end <= len(text)")
        if cancel_event is not None and cancel_event.is_set():
            return self._cancelled_result(perf_counter())
        tokenizer = self.tokenizer
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        except (NotImplementedError, TypeError) as exc:
            raise ValueError("refine_text requires tokenizer character-offset mapping") from exc
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if len(offsets) != len(ids) or any(
            len(offset) != 2 or not 0 <= offset[0] <= offset[1] <= len(text)
            for offset in offsets
        ):
            raise ValueError("tokenizer returned invalid character offsets")
        if tokenizer.decode(ids, skip_special_tokens=True) != text:
            raise ValueError("text does not round-trip exactly through the generator tokenizer")
        editable = [start < end and b > a and a < end and b > start for a, b in offsets]
        selected_offsets = [offset for offset, edit in zip(offsets, editable) if edit]
        if start < end and not selected_offsets:
            raise ValueError("edit_span overlaps no generator token")
        result = self.refine(
            request, ids, editable, hook=hook, cancel_event=cancel_event,
            observer=observer, observe_every=observe_every,
        )
        result.trace["edit_span"] = [start, end]
        result.trace["expanded_edit_span"] = (
            [min(a for a, _ in selected_offsets), max(b for _, b in selected_offsets)]
            if selected_offsets else [start, end]
        )
        return result

    def replacement_candidates(
        self, request: DiffusionRequest, text: str, span: tuple[int, int],
        top_k: int = 32,
    ) -> dict[str, Any]:
        """One denoiser forward supplies contextual one-token word alternatives.

        This is a vocabulary screen, not sequence likelihood or a grammaticality
        guarantee. Unsupported spans fail closed without running the model.
        """
        started = perf_counter()

        def unsupported(reason: str) -> dict[str, Any]:
            return {"replacements": [], "trace": {
                "supported": False, "reason": reason, "denoising_forwards": 0,
                "elapsed_seconds": perf_counter() - started,
            }}

        if (not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 256):
            raise ValueError("top_k must be in [1, 256]")
        if (not isinstance(text, str) or not isinstance(span, (tuple, list))
                or len(span) != 2 or any(type(value) is not int for value in span)):
            return unsupported("invalid_text_or_span")
        start, end = span
        if not 0 <= start < end <= len(text) or not re.fullmatch(r"[A-Za-z]+", text[start:end]):
            return unsupported("not_an_alphabetic_word")
        if (start and text[start - 1].isalpha()) or (end < len(text) and text[end].isalpha()):
            return unsupported("not_a_whole_word_boundary")
        with _GENERATION_LOCK:
            tokenizer = self.tokenizer
            try:
                encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
            except (NotImplementedError, TypeError):
                return unsupported("tokenizer_has_no_character_offsets")
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            if not 1 <= len(ids) <= 256 or len(ids) != len(offsets):
                return unsupported("text_does_not_fit_one_canvas")
            if tokenizer.decode(ids, skip_special_tokens=True) != text:
                return unsupported("text_does_not_roundtrip_exactly")
            selected = [index for index, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
            if len(selected) != 1:
                return unsupported("word_spans_multiple_generator_tokens")
            position = selected[0]
            a, b = offsets[position]
            if a > start or b < end or text[a:start].strip() or text[end:b].strip():
                return unsupported("generator_token_crosses_word_boundary")

            self._ensure_loaded()
            import mlx.core as mx
            from mlx_vlm.generate.common import generation_stream, wired_limit

            native = importlib.import_module("mlx_vlm.generate.diffusion")
            model = self._model
            context_ids = self._prepare_prompt(request.prompt)
            vocab_size = int(model.config.text_config.vocab_size)
            if len(ids) > int(model.config.canvas_length):
                return unsupported("text_exceeds_model_canvas")
            mx.random.seed(request.seed)
            prefill_started = perf_counter()
            try:
                with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
                    cache = model.make_cache()
                    cache = model.diffusion_prefill_cache(context_ids, cache=cache)
                    mx.eval([entry.state for entry in cache])
                    prefill_seconds = perf_counter() - prefill_started
                    canvas = mx.array([ids], dtype=context_ids.dtype)
                    noise = native._diffusion_initialize_canvas(1, 1, vocab_size, context_ids.dtype)
                    canvas = mx.where(mx.arange(len(ids))[None, :] == position, noise, canvas)
                    masks = model.diffusion_decoder_masks(canvas, cache, None)
                    probe_started = perf_counter()
                    logits = model.diffusion_decoder_logits(
                        canvas, cache=cache, self_conditioning=None,
                        decoder_attention_mask=masks,
                    )[0, position].astype(mx.float32)
                    k = min(top_k, vocab_size)
                    indices = mx.argpartition(logits, vocab_size - k)[-k:]
                    values = mx.take(logits, indices)
                    order = mx.argsort(-values)
                    indices, values = mx.take(indices, order), mx.take(values, order)
                    mx.eval(indices, values)
                    ranked_ids, ranked_logits = indices.tolist(), values.tolist()
                    probe_seconds = perf_counter() - probe_started
            finally:
                mx.synchronize()
                mx.clear_cache()

            special = set(getattr(tokenizer, "all_special_ids", []) or [])
            prefix, suffix = text[:start], text[end:]
            replacements: list[str] = []
            for token_id in ranked_ids:
                if token_id in special:
                    continue
                proposal = list(ids)
                proposal[position] = int(token_id)
                decoded = tokenizer.decode(proposal, skip_special_tokens=True)
                if not decoded.startswith(prefix) or not decoded.endswith(suffix):
                    continue
                stop = len(decoded) - len(suffix) if suffix else len(decoded)
                word = decoded[len(prefix):stop]
                if re.fullmatch(r"[A-Za-z]+", word) and word not in replacements:
                    replacements.append(word)
            return {"replacements": replacements, "trace": {
                "supported": True, "reason": None,
                "denoising_forwards": 1, "self_conditioning": False,
                "corruption": "one_uniform_vocabulary_token",
                "token_position": position, "canvas_length": len(ids),
                "top_k": k, "accepted_words": len(replacements),
                "prefill_seconds": prefill_seconds, "probe_seconds": probe_seconds,
                "elapsed_seconds": perf_counter() - started,
                "model_revision": self.revision,
                "scores_are_sequence_likelihoods": False,
            }}

    def _run(
        self, request: DiffusionRequest, *, hook: GuidanceHook | None,
        cancel_event: Event | None, observer: DraftObserver | None,
        observe_every: int, initial_token_ids: list[int] | None = None,
        edit_mask: list[bool] | None = None,
    ) -> DiffusionResult:
        if isinstance(observe_every, bool) or not isinstance(observe_every, int) or observe_every < 1:
            raise ValueError("observe_every must be a positive integer")
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
            context_ids = self._prepare_prompt(request.prompt)
            if context_ids.ndim != 2 or context_ids.shape[0] != 1:
                raise ValueError("Only one text prompt at a time is supported")
            canvas_length = len(initial_token_ids) if initial_token_ids is not None else min(request.max_tokens, int(model.config.canvas_length))
            if canvas_length > int(model.config.canvas_length):
                raise ValueError("initial canvas exceeds the model canvas length")
            mx.random.seed(request.seed)
            mx.reset_peak_memory()
            try:
                with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
                    result = self._generate_canvas(
                        request, context_ids, canvas_length, native, hook,
                        cancel_event, started,
                        observer=observer, observe_every=observe_every,
                        initial_token_ids=initial_token_ids, edit_mask=edit_mask,
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

    def _decode_canvas(self, canvas_ids: list[int]) -> tuple[str, list[int], dict[str, Any]]:
        """One cleanup path for intermediate drafts and final results."""
        config = self._model.config.generation_config or {}
        eos = config.get("eos_token_id", getattr(self.tokenizer, "eos_token_id", None))
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        ids = list(canvas_ids)
        for index, token_id in enumerate(ids):
            if token_id in eos_ids:
                ids = ids[:index]
                break
        raw_token_count = len(ids)
        ids, channel_header = self._strip_channel_header(ids)
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip(), ids, {
            "raw_token_count": raw_token_count, "channel_header": channel_header,
        }

    def _generate_canvas(
        self, request: DiffusionRequest, context_ids: Any, canvas_length: int,
        native: Any, hook: GuidanceHook | None, cancel_event: Event | None,
        started: float,
        *, observer: DraftObserver | None = None, observe_every: int = 3,
        initial_token_ids: list[int] | None = None,
        edit_mask: list[bool] | None = None,
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
        repairing = initial_token_ids is not None
        if repairing:
            initial_token_ids, edit_mask = self._edit_inputs(initial_token_ids, edit_mask)
            if len(initial_token_ids) != canvas_length or max(initial_token_ids) >= vocab_size:
                raise ValueError("initial token IDs must match canvas length and model vocabulary")
        persistent_mask = mx.array([edit_mask], dtype=mx.bool_) if repairing else None
        pinned = mx.array([initial_token_ids], dtype=context_ids.dtype) if repairing else None
        editable_indices = [index for index, edit in enumerate(edit_mask or []) if edit]
        has_protected = repairing and len(editable_indices) < canvas_length
        index_array = mx.array([editable_indices], dtype=mx.int32) if has_protected else None
        control_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        eos = config.get("eos_token_id", getattr(self.tokenizer, "eos_token_id", None))
        control_ids.update(eos if isinstance(eos, (tuple, list)) else [eos])
        control_ids = sorted(value for value in control_ids if isinstance(value, Integral) and 0 <= value < vocab_size)
        control_indices = mx.broadcast_to(mx.array(control_ids)[None, None, :], (1, canvas_length, len(control_ids))) if has_protected and control_ids else None

        def constrain(logits: Any) -> Any:
            if not has_protected:
                return logits
            if control_indices is not None:
                # A repair cannot introduce an early EOS/control marker that
                # would hide protected suffix text during ordinary decoding.
                values = mx.take_along_axis(logits, control_indices, axis=-1)
                values = mx.where(pinned[..., None] == control_indices, values, -10000.0)
                logits = mx.put_along_axis(logits, control_indices, values, axis=-1)
            pinned_logits = mx.where(mx.arange(vocab_size) == pinned[..., None], 0.0, -10000.0).astype(logits.dtype)
            return mx.where(persistent_mask[..., None], logits, pinned_logits)
        observer_calls = 0
        terminal_observed = False

        def observe(canvas: Any, step: int, terminal: bool) -> bool:
            nonlocal observer_calls, terminal_observed
            if observer is None:
                return False
            full_ids = [int(token_id) for token_id in canvas[0].tolist()]
            text, clean_ids, _ = self._decode_canvas(full_ids)
            observer_calls += 1
            terminal_observed = terminal
            return bool(observer(DiffusionDraft(
                text=text, token_ids=clean_ids, canvas_token_ids=full_ids,
                step=step, num_steps=request.steps, terminal=terminal,
            )))

        if repairing and not editable_indices:
            text, ids, metadata = self._decode_canvas(initial_token_ids)
            observe(pinned, 0, True)
            return DiffusionResult(text, ids, {
                "runtime": self.identity, "model_revision": self.revision,
                "cancelled": False, "passes": 0, "hook_calls": 0,
                "hook_count": 0, "observer_calls": observer_calls,
                "stopping_reason": "no_editable_positions", "refinement": True,
                "editable_positions": 0, "protected_positions": canvas_length,
                "canvas_length": canvas_length, "canvas_token_ids": list(initial_token_ids),
                "elapsed_seconds": perf_counter() - started, "step_trace": [],
                **metadata,
            })
        prefill_started = perf_counter()
        cache = model.make_cache()
        # No padding: one text-only prompt, native dynamic hybrid KV cache.
        cache = model.diffusion_prefill_cache(context_ids, cache=cache)
        mx.eval([entry.state for entry in cache])
        prefill_seconds = perf_counter() - prefill_started
        current_canvas = native._diffusion_initial_canvas(
            None, 0, 1, canvas_length, vocab_size, context_ids.dtype
        )
        if has_protected:
            current_canvas = mx.where(persistent_mask, current_canvas, pinned)
        revealed = ~persistent_mask if has_protected else mx.zeros(current_canvas.shape, dtype=mx.bool_)
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
                    observer_calls=observer_calls, refinement=repairing,
                )
            pass_started = perf_counter()
            logits = decoder_without_sc(current_canvas) if self_conditioning is None else decoder_with_sc(current_canvas, self_conditioning)
            temperature = native._diffusion_linear_temperature(remaining, request.steps, schedule)
            if temperature is not None:
                logits = logits / temperature
            logits = mx.stop_gradient(logits)
            editable = mx.ones(current_canvas.shape, dtype=mx.bool_) if self.sampler == "entropy-bound" else ~revealed
            if repairing:
                editable = editable & persistent_mask
            if has_protected:
                # Finite one-hot logits keep frozen IDs in self-conditioning;
                # explicit ID clamps below also protect against hostile hooks.
                logits = constrain(logits)
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
                if has_protected:
                    guided = constrain(guided)
                delta = float(mx.max(mx.abs(guided.astype(mx.float32) - logits.astype(mx.float32))).item())
                logits = mx.stop_gradient(guided)
                hook_calls += 1

            # Every decision below consumes the SAME post-hook logits.
            argmax_canvas = mx.argmax(logits, axis=-1).astype(context_ids.dtype)
            if has_protected:
                argmax_canvas = mx.where(persistent_mask, argmax_canvas, pinned)
            if remaining == 1:
                mx.eval(argmax_canvas)
                step_trace.append({"step": index, "hook_max_abs_delta": delta, "accepted": len(editable_indices) if repairing else canvas_length, "elapsed_seconds": perf_counter() - pass_started})
                break
            sampled = argmax_canvas if request.temperature <= 0 else native._diffusion_sample_canvas(logits, context_ids.dtype, request.temperature)
            next_sc = None
            if self.sampler == "entropy-bound":
                entropy = native._diffusion_token_entropy(logits)
                next_sc = model.diffusion_self_conditioning(logits, self_conditioning_context)
                if has_protected:
                    editable_entropy = mx.take_along_axis(entropy, index_array, axis=-1)
                    selected = native._diffusion_entropy_transfer_mask(editable_entropy, entropy_bound)
                    acceptance = mx.put_along_axis(mx.zeros_like(revealed), index_array, selected, axis=-1)
                else:
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
            if has_protected:
                current_canvas = mx.where(persistent_mask, current_canvas, pinned)
                draft_canvas = mx.where(persistent_mask, draft_canvas, pinned)
            mx.eval(current_canvas, argmax_canvas, revealed)
            step_trace.append({"step": index, "hook_max_abs_delta": delta, "accepted": int(mx.sum(acceptance).item()), "elapsed_seconds": perf_counter() - pass_started})
            complete = self.sampler == "confidence-threshold" and bool(mx.all(revealed).item())
            if complete:
                stopping_reason = "all_confident"
            stable_ids, stable_logits = argmax_canvas, logits
            if has_protected:
                stable_ids = mx.take(argmax_canvas, index_array[0], axis=1)
                stable_logits = mx.take(logits, index_array[0], axis=1)
            stable = False if complete else native._diffusion_stable_and_confident(stable_ids, stable_logits, history, stopping)
            if stable:
                stopping_reason = "stable_and_confident"
            natural_stop = complete or stable
            if observer is not None and (natural_stop or (index + 1) % observe_every == 0):
                if observe(argmax_canvas, index, natural_stop):
                    if not natural_stop:
                        stopping_reason = "observer_requested"
                    break
            if natural_stop:
                break
            if next_sc is None:
                next_sc = model.diffusion_self_conditioning(logits, self_conditioning_context)
            self_conditioning = next_sc

        mx.eval(argmax_canvas)
        canvas_ids = [int(token_id) for token_id in argmax_canvas[0].tolist()]
        if observer is not None and not terminal_observed:
            observe(argmax_canvas, len(step_trace) - 1, True)
        text, ids, metadata = self._decode_canvas(canvas_ids)
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
            **metadata,
            "requested_steps": request.steps,
            "passes": len(step_trace),
            "hook_calls": hook_calls,
            "hook_count": hook_calls,
            "observer_calls": observer_calls,
            "refinement": repairing,
            "editable_positions": len(editable_indices) if repairing else canvas_length,
            "protected_positions": canvas_length - len(editable_indices) if repairing else 0,
            "canvas_token_ids": canvas_ids,
            "stopping_reason": stopping_reason,
            "prefill_seconds": prefill_seconds,
            "elapsed_seconds": perf_counter() - started,
            "step_trace": step_trace,
        })
