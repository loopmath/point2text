"""Small, backend-independent seams shared by the first demo."""

from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class CorpusRecord:
    id: str
    topic: str
    title: str
    text: str


class TextEncoder(Protocol):
    @property
    def identity(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class TextGenerator(Protocol):
    @property
    def identity(self) -> str: ...

    def generate(self, examples: Sequence[CorpusRecord], *, seed: int,
                 cancel_event: object) -> str: ...


@dataclass(frozen=True)
class Candidate:
    text: str
    x: float
    y: float
    distance: float
    elapsed_seconds: float
    seed: int

    def payload(self) -> dict:
        return asdict(self)
