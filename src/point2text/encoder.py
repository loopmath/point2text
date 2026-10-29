"""Frozen scoring encoder; intentionally makes no differentiable-bridge claim."""

from collections.abc import Sequence

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Verified using https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_TOKENS = 256


class SentenceEncoder:
    """Download once to the user's HF cache, then run deterministic CPU scoring."""

    identity = (
        f"{MODEL_ID}@{MODEL_REVISION};backend=torch;device=cpu;dtype=float32;"
        "pooling=attention-mask-mean;normalize=l2;prompt=none;"
        f"tokenizer=checkpoint;truncation=longest-first:{MAX_TOKENS};special_tokens=true"
    )

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install the local encoder extra: 'uv sync --extra encoder' in the "
                "checkout, or 'python -m pip install .[encoder]' from source."
            ) from exc

        self.model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
        self.model.max_seq_length = MAX_TOKENS
        self.model.float()
        self.model.eval()
        self.model.requires_grad_(False)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        for index, count in enumerate(self.token_lengths(texts), 1):
            if count > MAX_TOKENS:
                raise ValueError(
                    f"Embedding input {index} has {count} tokens; maximum is {MAX_TOKENS} "
                    "including special tokens. Shorten the paragraph before scoring."
                )
        # Explicit empty prompt avoids model-config defaults changing this contract.
        vectors = self.model.encode(
            list(texts), batch_size=32, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
            precision="float32", prompt="",
        )
        return np.asarray(vectors, dtype=np.float32)

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Count the normal tokenizer's complete input, before any truncation."""
        encoded = self.model.tokenizer(
            list(texts), add_special_tokens=True, truncation=False,
            padding=False, return_attention_mask=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]
