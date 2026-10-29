"""Sparse, explicitly approximate generator-token to encoder-vector bridge.

One generator token is decoded in isolation, tokenized with the encoder without
special tokens, and represented by the mean of those encoder word embeddings.
This is NOT differentiable re-tokenization: word boundaries, sequence length,
and context-dependent decoding are lost. Canonical scores must re-encode text.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class BridgedToken:
    surface: str
    vector: torch.Tensor
    encoder_token_count: int
    unknown_fraction: float
    kind: str  # lexical, whitespace, empty, special, or mask

    @property
    def present(self) -> bool:
        return self.kind in {"lexical", "mask"}


class SparseTokenBridge:
    """Decode only requested unique IDs, retaining their detached CPU vectors."""

    def __init__(self, generator_tokenizer: Any, encoder_tokenizer: Any,
                 embedding_layer: torch.nn.Module) -> None:
        self.generator_tokenizer = generator_tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.embedding_layer = embedding_layer
        self.cache: dict[int, BridgedToken] = {}
        self.special_ids = set(getattr(generator_tokenizer, "all_special_ids", None) or ())
        self.mask_id = getattr(generator_tokenizer, "mask_token_id", None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id"):
            value = getattr(generator_tokenizer, name, None)
            if isinstance(value, int):
                self.special_ids.add(value)
        self.unknown_id = encoder_tokenizer.unk_token_id
        if self.unknown_id is None:
            raise ValueError("The sparse bridge requires an encoder unknown token")

    def _lookup(self, ids: list[int]) -> torch.Tensor:
        device = self.embedding_layer.weight.device
        with torch.no_grad():
            vectors = self.embedding_layer(torch.tensor(ids, dtype=torch.long, device=device))
            return vectors.mean(dim=0).detach()

    def _decode(self, token_id: int) -> str:
        try:
            return self.generator_tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
        except TypeError:
            # MLX wrappers may expose decode(ids) without HF keyword arguments.
            return self.generator_tokenizer.decode([token_id])

    def _token(self, token_id: int) -> BridgedToken:
        surface = self._decode(token_id)
        if not isinstance(surface, str):
            raise ValueError("Generator tokenizer decode must return a string")
        if token_id == self.mask_id:
            encoder_mask = getattr(self.encoder_tokenizer, "mask_token_id", None)
            vector = self._lookup([encoder_mask if encoder_mask is not None else self.unknown_id])
            return BridgedToken(surface, vector, 1, 0.0, "mask")
        if token_id in self.special_ids:
            return BridgedToken(surface, self._lookup([self.unknown_id]), 0, 0.0, "special")
        if not surface.strip():
            kind = "whitespace" if surface else "empty"
            return BridgedToken(surface, self._lookup([self.unknown_id]), 0, 0.0, kind)
        encoded = self.encoder_tokenizer(surface, add_special_tokens=False, truncation=False)
        ids = list(encoded["input_ids"])
        # Some token surfaces normalize away. Record omission rather than zero-fill.
        if not ids:
            return BridgedToken(surface, self._lookup([self.unknown_id]), 0, 0.0, "empty")
        return BridgedToken(surface, self._lookup(ids), len(ids),
                            sum(token == self.unknown_id for token in ids) / len(ids), "lexical")

    def lookup(self, token_ids: np.ndarray) -> tuple[torch.Tensor, np.ndarray, dict[str, np.ndarray]]:
        ids = np.asarray(token_ids)
        if ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError("Generator token IDs must be nonnegative integers")
        for token_id in np.unique(ids):
            key = int(token_id)
            if key not in self.cache:
                self.cache[key] = self._token(key)
        tokens = [self.cache[int(token_id)] for token_id in ids.flat]
        shape = tuple(ids.shape)
        vectors = torch.stack([token.vector for token in tokens]).reshape(*shape, -1)
        present = np.asarray([token.present for token in tokens], dtype=bool).reshape(shape)
        metadata = {
            "unknown_fraction": np.asarray([t.unknown_fraction for t in tokens]).reshape(shape),
            "encoder_token_count": np.asarray([t.encoder_token_count for t in tokens]).reshape(shape),
            "special": np.asarray([t.kind == "special" for t in tokens]).reshape(shape),
            "empty": np.asarray([t.kind in {"empty", "whitespace"} for t in tokens]).reshape(shape),
            "mask": np.asarray([t.kind == "mask" for t in tokens]).reshape(shape),
        }
        return vectors, present, metadata
