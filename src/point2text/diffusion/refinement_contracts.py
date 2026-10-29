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
    max_restarts: int = 3
    restart_candidates: int = 2
    initial_candidates_before_refinement: int = 2
    basin_generation_fraction: float = 0.6
    repair_span_mode: str = "words"

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
        for name in ("restart_candidates", "initial_candidates_before_refinement"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.max_restarts, bool) or not isinstance(self.max_restarts, int) or self.max_restarts < 0:
            raise ValueError("max_restarts must be a nonnegative integer")
        if (isinstance(self.basin_generation_fraction, bool) or not math.isfinite(self.basin_generation_fraction)
                or not 0 < self.basin_generation_fraction <= 1):
            raise ValueError("basin_generation_fraction must be in (0, 1]")
        if not isinstance(self.repair_span_mode, str) or self.repair_span_mode not in {"words", "sentence"}:
            raise ValueError("repair_span_mode must be 'words' or 'sentence'")
        if self.min_words > self.max_words:
            raise ValueError("min_words must not exceed max_words")


class EditProposer(Protocol):
    def propose(self, text: str, target: tuple[float, float], *,
                max_proposals: int = 64) -> Sequence[TextProposal]: ...

    def hotspots(self, text: str, target: tuple[float, float], *,
                 limit: int = 4) -> Sequence[tuple[int, int]]: ...
