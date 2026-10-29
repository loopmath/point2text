"""Plain-text contracts for canonical verification and local refinement."""

from dataclasses import dataclass, field
import math
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class TextProposal:
    text: str
    edit_span: tuple[int, int] | None = None  # Character offsets in the source text.
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifiedText:
    text: str
    x: float
    y: float
    distance: float
    encoder_token_ids: tuple[int, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementConfig:
    epsilon: float = 1e-4
    max_seconds: float | None = 120.0
    max_hard_queries: int = 1024
    max_rounds: int = 20
    observe_every: int = 3
    beam_width: int = 4
    proposals_per_round: int = 64
    stall_rounds: int = 4
    min_improvement: float = 2e-7
    min_words: int = 60
    max_words: int = 100
    max_encoder_tokens: int = 256
    repair_steps: int = 8
    repair_every: int = 2

    def __post_init__(self):
        for name in ("epsilon", "max_seconds"):
            value = getattr(self, name)
            if name == "max_seconds" and value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.min_improvement) or self.min_improvement < 0:
            raise ValueError("min_improvement must be finite and nonnegative")
        for name in ("max_hard_queries", "max_rounds", "observe_every", "beam_width",
                     "proposals_per_round", "stall_rounds", "min_words", "max_words",
                     "max_encoder_tokens", "repair_steps", "repair_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_words > self.max_words:
            raise ValueError("min_words must not exceed max_words")


class EditProposer(Protocol):
    def propose(self, text: str, target: tuple[float, float], *,
                max_proposals: int = 64) -> Sequence[TextProposal]: ...

    def hotspots(self, text: str, target: tuple[float, float], *,
                 limit: int = 4) -> Sequence[tuple[int, int]]: ...
