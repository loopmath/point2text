"""Native tensors stay within runtime/guidance; final results contain real text."""

from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class DiffusionRequest:
    prompt: str
    seed: int = 42
    max_tokens: int = 128
    steps: int = 32
    temperature: float = 0.8


@dataclass(frozen=True)
class DiffusionStep:
    logits: Any
    token_ids: Any
    editable_mask: Any
    step: int
    num_steps: int
    tokenizer: Any
    context_ids: Any = None


@dataclass(frozen=True)
class DiffusionResult:
    text: str
    token_ids: list[int]
    trace: dict[str, Any] = field(default_factory=dict)


GuidanceHook = Callable[[DiffusionStep], Any]


class DiffusionRuntime(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def tokenizer(self) -> Any: ...

    def generate(self, request: DiffusionRequest, *, hook: GuidanceHook | None = None,
                 cancel_event: Event | None = None) -> DiffusionResult: ...
