"""Experimental sparse map-loss steering of native diffusion logits.

Only a compact top-k proposal crosses to the CPU reward adapter. The generator
and sampler retain native tensors; no generator backward graph is constructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np

from .contracts import DiffusionStep


@dataclass(frozen=True)
class GuidanceConfig:
    strength: float = 1.0
    top_k: int = 32
    guided_steps: int = 8
    inner_updates: int = 1
    step_size: float = 4.0
    temperature: float = 1.0
    kl_weight: float = 0.01
    gradient_clip: float = 1.0
    max_delta: float = 1.0
    normalize_gradient: bool = True
    start_fraction: float = 0.2
    end_fraction: float = 0.9
    max_positions: int = 254

    def __post_init__(self) -> None:
        for name in ("strength", "step_size", "kl_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("temperature", "gradient_clip", "max_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("top_k", "guided_steps", "inner_updates", "max_positions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.start_fraction <= self.end_fraction <= 1:
            raise ValueError("Guidance schedule must satisfy 0 <= start <= end <= 1")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _compact_numpy(logits: np.ndarray, k: int, temperature: float):
    ids = np.argpartition(logits, logits.shape[-1] - k, axis=-1)[:, -k:]
    selected = np.take_along_axis(logits, ids, axis=-1)
    maximum = np.max(logits / temperature, axis=-1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = np.exp(logits / temperature - maximum).sum(axis=-1)
        numerator = np.exp(selected / temperature - maximum).sum(axis=-1)
    return selected.astype(np.float32), ids.astype(np.int64), numerator / denominator


def _updated_compact(base: np.ndarray, current: np.ndarray,
                     gradients: np.ndarray, editable: np.ndarray,
                     config: GuidanceConfig):
    """Apply a bounded descent direction while retaining fixed-token logits."""
    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.shape != current.shape or not np.isfinite(gradients).all():
        raise ValueError("Reward gradients must be finite and match top-k logits")
    gradients = np.where(editable[:, None], gradients, 0.0)
    norm = float(np.linalg.norm(gradients))
    direction = gradients
    if config.normalize_gradient and norm > 0:
        direction = direction / norm
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm > config.gradient_clip:
        direction = direction * (config.gradient_clip / direction_norm)
    proposal = current - config.strength * config.step_size * direction
    delta = np.clip(proposal - base, -config.max_delta, config.max_delta)
    delta = np.where(editable[:, None], delta, 0.0)
    updated = (base + delta).astype(np.float32)
    return updated, norm, float(np.linalg.norm(direction)), float(np.max(np.abs(delta)))


class SparseMapGuidance:
    """A pre-commit hook using a frozen encoder reward's input gradients.

    The reward implements ``loss_and_gradient`` on NumPy [positions, top_k]
    proposals. Its token bridge is an approximation; final decoded text must be
    scored separately through the ordinary encoder, as DiffusionSearch does.
    """

    def __init__(self, reward: Any, target: tuple[float, float],
                 config: GuidanceConfig | None = None) -> None:
        target_array = np.asarray(target, dtype=float)
        if target_array.shape != (2,) or not np.isfinite(target_array).all():
            raise ValueError("Target must contain two finite map coordinates")
        self.reward = reward
        self.target = tuple(float(value) for value in target_array)
        self.config = config or GuidanceConfig()
        self.reset()

    def reset(self) -> None:
        self.entries: list[dict] = []
        self.calls = 0

    @property
    def trace(self) -> dict:
        return {"method": "sparse-topk-map-gradient-v1",
                "config": asdict(self.config), "hook_calls": self.calls,
                "guided_step_count": len(self.entries),
                "steps": _json_safe(self.entries),
                "approximation": "top-k generator-token to encoder-input bridge; "
                                 "hard decoded-text scoring required"}

    def _due(self, step: int, count: int) -> bool:
        first = round(self.config.start_fraction * (count - 1))
        last = round(self.config.end_fraction * (count - 1))
        schedule = np.linspace(first, last, min(self.config.guided_steps,
                                                last - first + 1), dtype=int)
        return step in schedule

    def __call__(self, step: DiffusionStep):
        self.calls += 1
        config = self.config
        # Preserve the exact native object and avoid importing MLX for identity.
        if config.strength == 0 or config.step_size == 0:
            return step.logits
        if step.num_steps < 1 or not 0 <= step.step < step.num_steps:
            raise ValueError("Invalid diffusion step index or step count")
        if not self._due(step.step, step.num_steps):
            return step.logits
        started = time.perf_counter()
        shape = tuple(step.logits.shape)
        if len(shape) != 3 or shape[0] != 1 or shape[2] < 1:
            raise ValueError("Guidance requires logits shaped [1, positions, vocabulary]")
        positions, vocabulary = shape[1:]
        if not 1 <= positions <= config.max_positions:
            raise ValueError(f"Guidance supports 1–{config.max_positions} canvas positions")
        expected = (1, positions)
        if tuple(step.token_ids.shape) != expected or tuple(step.editable_mask.shape) != expected:
            raise ValueError("Token IDs and editable mask must have shape [1, positions]")
        editable = np.asarray(step.editable_mask).reshape(-1)
        tokens = np.asarray(step.token_ids).reshape(-1)
        if editable.dtype.kind != "b":
            raise ValueError("The editable mask must be boolean")
        if tokens.dtype.kind not in "iu" or np.any(tokens < 0) or np.any(tokens >= vocabulary):
            raise ValueError("Canvas token IDs must be valid integer vocabulary IDs")
        if not editable.any():
            return step.logits
        k = min(config.top_k, vocabulary)
        native_numpy = isinstance(step.logits, np.ndarray)
        flat = step.logits[0]
        if native_numpy:
            base, ids, retained_mass = _compact_numpy(flat, k, config.temperature)
        else:
            # Lazy import keeps package import and NumPy fixture tests GPU-free.
            import mlx.core as mx
            ids_native = mx.argpartition(flat, vocabulary - k, axis=-1)[:, -k:]
            selected = mx.take_along_axis(flat, ids_native, axis=-1)
            scaled = flat.astype(mx.float32) / config.temperature
            retained = mx.exp(mx.logsumexp(selected.astype(mx.float32) / config.temperature,
                                           axis=-1) - mx.logsumexp(scaled, axis=-1))
            mx.eval(selected, ids_native, retained)
            base = np.asarray(selected.astype(mx.float32)).copy()
            ids = np.asarray(ids_native).astype(np.int64)
            retained_mass = np.asarray(retained).astype(np.float64)
        if not np.isfinite(base).all() or not np.isfinite(retained_mass).all():
            raise ValueError("Guidance needs finite top-k logits and proposal probabilities")
        current = base.copy()
        updates = []
        for update in range(config.inner_updates):
            before = time.perf_counter()
            loss, gradients, diagnostics = self.reward.loss_and_gradient(
                current, ids, tokens, editable, step.tokenizer, self.target,
                temperature=config.temperature, kl_weight=config.kl_weight,
                reference_logits=base, retained_mass=retained_mass,
            )
            if not math.isfinite(float(loss)):
                raise ValueError("The encoder reward returned a non-finite loss")
            current, norm, direction_norm, max_delta = _updated_compact(
                base, current, gradients, editable, config)
            updates.append({"update": update, "loss": float(loss),
                            "gradient_norm": norm, "direction_norm": direction_norm,
                            "max_delta": max_delta,
                            "seconds": time.perf_counter() - before,
                            "diagnostics": _json_safe(diagnostics)})
        delta = current - base
        if native_numpy:
            updated = flat.copy()
            np.put_along_axis(updated, ids,
                              np.take_along_axis(flat, ids, axis=-1) + delta, axis=-1)
            result = np.where(editable[:, None], updated, flat)[None, ...]
            applied_delta = np.take_along_axis(result[0] - flat, ids, axis=-1)
        else:
            values = selected + mx.array(delta, dtype=selected.dtype)
            updated = mx.put_along_axis(flat, ids_native, values, axis=-1)
            result = mx.where(step.editable_mask[0, :, None], updated, flat)[None, ...]
            applied = mx.take_along_axis(result[0], ids_native, axis=-1) - selected
            mx.eval(result, applied)
            applied_delta = np.asarray(applied.astype(mx.float32))
        self.entries.append({"step": step.step, "positions": positions,
                             "editable_positions": int(editable.sum()), "top_k": k,
                             "retained_mass_mean": float(np.mean(retained_mass[editable])),
                             "max_delta": float(np.max(np.abs(delta))),
                             "applied_max_delta": float(np.max(np.abs(applied_delta))),
                             "changed_logits": int(np.count_nonzero(applied_delta)),
                             "seconds": time.perf_counter() - started,
                             "updates": updates})
        return result
