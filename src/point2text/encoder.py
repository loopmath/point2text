"""Frozen scoring encoder; intentionally makes no differentiable-bridge claim."""

from collections.abc import Sequence
import hashlib
import inspect
import json

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

    max_length = MAX_TOKENS
    normalize = True
    prompt = ""

    def __init__(self, checkpoint: str = MODEL_ID, revision: str | None = MODEL_REVISION,
                 max_length: int = MAX_TOKENS, normalize: bool = True,
                 prompt: str = "") -> None:
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("checkpoint must be nonempty text")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be nonempty text when supplied")
        if type(max_length) is not int or max_length < 3:
            raise ValueError("max_length must be an integer >= 3")
        if type(normalize) is not bool or not isinstance(prompt, str):
            raise ValueError("normalize must be boolean and prompt must be text")
        # Direct construction, like profiles, must not pass MiniLM's revision to
        # an unrelated checkpoint. Use a real immutable revision for new models.
        if checkpoint != MODEL_ID and revision == MODEL_REVISION:
            revision = None
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install the local encoder extra: 'uv sync --extra encoder' in the "
                "checkout, or 'python -m pip install .[encoder]' from source."
            ) from exc

        self.checkpoint, self.revision = checkpoint, revision
        self.max_length, self.normalize, self.prompt = max_length, normalize, prompt
        self.model = SentenceTransformer(checkpoint, revision=revision, device="cpu",
                                         trust_remote_code=False)
        self.model.max_seq_length = max_length
        self.model.float()
        self.model.eval()
        self.model.requires_grad_(False)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        self.preprocessing = self._preprocessing()
        if (checkpoint, revision, max_length, normalize, prompt) != (
                MODEL_ID, MODEL_REVISION, MAX_TOKENS, True, ""):
            self.identity = "sentence-transformer-v1:" + json.dumps(
                self.preprocessing, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                default=str)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        for index, count in enumerate(self.token_lengths(texts), 1):
            if count > self.max_length:
                raise ValueError(
                    f"Embedding input {index} has {count} tokens; maximum is {self.max_length} "
                    "including special tokens. Shorten the paragraph before scoring."
                )
        # Explicit prompt (including empty) overrides model-config defaults.
        vectors = self.model.encode(
            list(texts), batch_size=32, show_progress_bar=False,
            normalize_embeddings=self.normalize, convert_to_numpy=True,
            precision="float32", prompt=self.prompt,
        )
        return np.asarray(vectors, dtype=np.float32)

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Count the normal tokenizer's complete input, before any truncation."""
        inputs = [self.prompt + text for text in texts]
        modules = list(getattr(self.model, "_modules", {}).values())
        if modules and type(modules[0]).__name__ == "Transformer":
            # Match the built-in ST Transformer's text preparation before its
            # tokenizer call, while deliberately disabling truncation here.
            inputs = [text.strip() for text in inputs]
            if getattr(modules[0], "do_lower_case", False):
                inputs = [text.lower() for text in inputs]
        encoded = self.model.tokenizer(
            inputs, add_special_tokens=True, truncation=False,
            padding=False, return_attention_mask=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def _preprocessing(self) -> dict:
        tokenizer = self.model.tokenizer
        modules = list(getattr(self.model, "_modules", {}).values())
        pipeline = []
        for module in modules:
            config = getattr(module, "get_config_dict", lambda: {})()
            pipeline.append({"type": f"{type(module).__module__}.{type(module).__name__}",
                             "config": config})
        backend = getattr(tokenizer, "backend_tokenizer", None)
        tokenizer_content = backend.to_str() if backend is not None else json.dumps(
            getattr(tokenizer, "get_vocab", lambda: {})(), sort_keys=True)
        first = modules[0] if modules else None
        body = getattr(first, "auto_model", None)
        return {
            "checkpoint": self.checkpoint, "revision": self.revision,
            "resolved_revision": getattr(getattr(body, "config", None), "_commit_hash", None),
            "backend": "torch", "device": "cpu", "dtype": "float32",
            "max_length": self.max_length, "normalize_embeddings": self.normalize,
            "prompt": self.prompt, "batch_size": 32, "precision": "float32",
            "pipeline": pipeline, "truncate_dim": getattr(self.model, "truncate_dim", None),
            "tokenizer": {"class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
                          "content_sha256": hashlib.sha256(tokenizer_content.encode()).hexdigest(),
                          "settings": json.loads(json.dumps(getattr(tokenizer, "init_kwargs", {}), default=str)),
                          "special_tokens": {k: str(v) for k, v in getattr(tokenizer, "special_tokens_map", {}).items()},
                          "padding_side": getattr(tokenizer, "padding_side", None),
                          "truncation_side": getattr(tokenizer, "truncation_side", None),
                          "add_special_tokens": True, "reject_overlength": True},
        }

    def gradient_capability(self) -> dict:
        """Conservatively audit the implemented token-input reward adapter.

        Check the actual loaded graph, not a checkpoint-name allowlist. Scoring
        remains available when a graph needs a different differentiable adapter.
        """
        reasons = []
        modules = list(getattr(self.model, "_modules", {}).values())
        names = [type(module).__name__ for module in modules]
        trusted = all(type(module).__module__.startswith("sentence_transformers.models.")
                      for module in modules)
        if not trusted or names not in (["Transformer", "Pooling"],
                                        ["Transformer", "Pooling", "Normalize"]):
            reasons.append("requires the audited Transformer → mean Pooling → optional Normalize graph")
        if self.prompt:
            reasons.append("embedding prompt prefixes need a prefix-aware token-input adapter")
        if getattr(self.model, "truncate_dim", None) is not None:
            reasons.append("embedding dimension truncation needs a matching reward adapter")
        if len(modules) >= 2 and names[:2] == ["Transformer", "Pooling"]:
            transformer, pooler = modules[:2]
            if getattr(transformer, "do_lower_case", False):
                reasons.append("Transformer text lowercasing needs a matching token bridge")
            modes = ["cls_token", "max_tokens", "mean_sqrt_len_tokens", "weightedmean_tokens", "lasttoken"]
            if (not getattr(pooler, "pooling_mode_mean_tokens", False)
                    or any(getattr(pooler, "pooling_mode_" + mode, False) for mode in modes)):
                reasons.append("only attention-mask mean pooling is implemented")
            if not getattr(pooler, "include_prompt", True):
                reasons.append("prompt-excluding pooling needs a matching reward adapter")
            body = getattr(transformer, "auto_model", None)
            if body is None or not callable(getattr(body, "get_input_embeddings", None)):
                reasons.append("Transformer must expose its input embedding layer")
            elif "inputs_embeds" not in inspect.signature(body.forward).parameters:
                reasons.append("Transformer forward must accept inputs_embeds")
        tokenizer = self.model.tokenizer
        cls_id, sep_id = getattr(tokenizer, "cls_token_id", None), getattr(tokenizer, "sep_token_id", None)
        if cls_id is None or sep_id is None or getattr(tokenizer, "pad_token_id", None) is None:
            reasons.append("requires fixed CLS, SEP and padding token IDs")
        else:
            build = getattr(tokenizer, "build_inputs_with_special_tokens", None)
            if build is None or build([123]) != [cls_id, 123, sep_id]:
                reasons.append("requires the single-sequence [CLS] tokens [SEP] layout")
        return {"supported": not reasons, "adapter": "torch-masked-mean-v1",
                "reasons": reasons, "normalize": self.normalize or "Normalize" in names,
                "max_length": self.max_length,
                "fallback": "Use unguided generation plus canonical measured text search, or implement and verify a matching differentiable adapter."}
