"""Frozen MiniLM input gradients and a sparse, approximate map reward.

Only logits for the selected generator candidates enter autograd. Generator
weights and computation never enter this graph. Exact parity applies to normal
encoder token IDs, including CLS/SEP; mismatched generator tokens use bridge.py.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..encoder import MAX_TOKENS, SentenceEncoder
from ..geometry import MapBundle
from .bridge import SparseTokenBridge


class TorchEmbeddingReward:
    def __init__(self, encoder: SentenceEncoder, bundle: MapBundle) -> None:
        if encoder.identity != bundle.encoder.identity:
            raise ValueError("Reward encoder identity does not match the frozen map")
        self.encoder = encoder
        self.tokenizer = encoder.model.tokenizer
        self.body = encoder.model[0].auto_model
        self.body.eval()
        self.body.requires_grad_(False)
        self.word_embeddings = self.body.get_input_embeddings()
        self.device = self.word_embeddings.weight.device
        self.dtype = self.word_embeddings.weight.dtype
        self.mean = torch.tensor(np.array(bundle.mean, copy=True), dtype=self.dtype, device=self.device)
        self.components = torch.tensor(np.array(bundle.components, copy=True), dtype=self.dtype, device=self.device)
        self.map_id = bundle.map_id
        self.max_tokens = min(int(encoder.model.max_seq_length), MAX_TOKENS)
        self.cls_id, self.sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        if self.cls_id is None or self.sep_id is None:
            raise ValueError("This MiniLM adapter requires fixed CLS and SEP tokens")
        self._bridges: dict[int, SparseTokenBridge] = {}

    def embed_inputs(self, input_vectors: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mask-aware mean pooling plus L2 normalization, retaining input gradients."""
        if input_vectors.ndim != 3 or attention_mask.shape != input_vectors.shape[:2]:
            raise ValueError("Expected input vectors [batch,length,hidden] and matching mask")
        if input_vectors.shape[1] > self.max_tokens:
            raise ValueError(f"Encoder reward input exceeds {self.max_tokens} tokens")
        outputs = self.body(inputs_embeds=input_vectors, attention_mask=attention_mask,
                            return_dict=True)
        mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        return F.normalize(pooled, p=2, dim=-1)

    def embed_encoder_tokens(self, input_ids: torch.Tensor,
                             attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Exact ordinary encoder-token path; callers include normal CLS and SEP."""
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if attention_mask is None:
            attention_mask = ids.ne(self.tokenizer.pad_token_id).to(torch.long)
        else:
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long, device=self.device)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
        return self.embed_inputs(self.word_embeddings(ids), attention_mask)

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        return (embeddings - self.mean) @ self.components.T

    def map_loss(self, embeddings: torch.Tensor, target: tuple[float, float]) -> torch.Tensor:
        location = torch.as_tensor(target, dtype=self.dtype, device=self.device)
        if location.shape != (2,) or not torch.isfinite(location).all():
            raise ValueError("Target must contain two finite PCA coordinates")
        return (self.project(embeddings) - location).square().sum(dim=-1).mean()

    def _bridge(self, generator_tokenizer: Any) -> SparseTokenBridge:
        key = id(generator_tokenizer)
        if key not in self._bridges:
            self._bridges[key] = SparseTokenBridge(generator_tokenizer, self.tokenizer, self.word_embeddings)
        return self._bridges[key]

    def loss_and_gradient(
        self, topk_logits: np.ndarray, topk_ids: np.ndarray, token_ids: np.ndarray,
        editable_mask: np.ndarray, generator_tokenizer: Any,
        target: tuple[float, float], temperature: float = 1.0, kl_weight: float = 0.0,
        *, reference_logits: np.ndarray | None = None,
        retained_mass: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, dict]:
        """Squared map error + mean KL(q || fixed proposal), over editable rows.

        Pass original proposal reference_logits for multiple inner updates. Top-k
        retained mass is supplied by the caller; selected logits cannot reveal it.
        Empty/whitespace/special surfaces are omitted from the lexical mixture;
        mixtures are renormalized over represented surfaces. Generator mask tokens
        use encoder MASK as a placeholder. None of these are exact re-tokenization.
        """
        logits_array = np.asarray(topk_logits, dtype=np.float32)
        ids = np.asarray(topk_ids)
        current_ids = np.asarray(token_ids)
        editable = np.asarray(editable_mask, dtype=bool)
        if logits_array.ndim != 2 or logits_array.shape != ids.shape or logits_array.shape[1] == 0:
            raise ValueError("Expected matching nonempty [positions,topk] logits and IDs")
        n, _ = logits_array.shape
        if current_ids.shape != (n,) or editable.shape != (n,) or n == 0:
            raise ValueError("Current IDs and editable mask must match the position count")
        if n + 2 > self.max_tokens:
            raise ValueError(f"Surrogate canvas has {n + 2} encoder positions; maximum is {self.max_tokens}")
        if not np.isfinite(logits_array).all() or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Reward requires finite logits and positive temperature")
        if not np.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("KL weight must be finite and nonnegative")
        if retained_mass is not None:
            retained_mass = np.asarray(retained_mass, dtype=np.float64)
            if retained_mass.shape != (n,) or not np.isfinite(retained_mass).all() or np.any((retained_mass < 0) | (retained_mass > 1 + 1e-5)):
                raise ValueError("Retained mass must have one probability per position")

        bridge = self._bridge(generator_tokenizer)
        candidate_vectors, present, metadata = bridge.lookup(ids)
        hard_vectors, hard_present, _ = bridge.lookup(current_ids)
        logits = torch.tensor(logits_array, dtype=self.dtype, device=self.device, requires_grad=True)
        probabilities = torch.softmax(logits / temperature, dim=-1)
        valid = torch.tensor(present, dtype=self.dtype, device=self.device)
        represented_mass = (probabilities * valid).sum(dim=-1, keepdim=True)
        weights = probabilities * valid / represented_mass.clamp_min(1e-12)
        soft_vectors = (weights.unsqueeze(-1) * candidate_vectors).sum(dim=1)
        has_surface = present.any(axis=1)
        edit_tensor = torch.tensor(editable & has_surface, dtype=torch.bool, device=self.device)
        vectors = torch.where(edit_tensor.unsqueeze(-1), soft_vectors, hard_vectors)
        position_mask = np.where(editable & has_surface, True, hard_present)
        special = self.word_embeddings(torch.tensor([self.cls_id, self.sep_id], device=self.device))
        inputs = torch.cat((special[:1], vectors, special[1:]), dim=0).unsqueeze(0)
        mask = torch.tensor([True, *position_mask.tolist(), True], dtype=torch.long, device=self.device).unsqueeze(0)
        embedding = self.embed_inputs(inputs, mask)
        objective = self.map_loss(embedding, target)

        baseline_array = logits_array if reference_logits is None else np.asarray(reference_logits, dtype=np.float32)
        if baseline_array.shape != logits_array.shape or not np.isfinite(baseline_array).all():
            raise ValueError("Reference logits must match selected logits and be finite")
        baseline = torch.tensor(baseline_array, dtype=self.dtype, device=self.device)
        log_reference = torch.log_softmax(baseline / temperature, dim=-1)
        per_position_kl = (probabilities * (torch.log_softmax(logits / temperature, dim=-1) - log_reference)).sum(dim=-1)
        editable_tensor = torch.tensor(editable, dtype=self.dtype, device=self.device)
        kl = (per_position_kl * editable_tensor).sum() / editable_tensor.sum().clamp_min(1)
        loss = objective + kl_weight * kl
        gradients, = torch.autograd.grad(loss, logits)
        gradients = gradients * editable_tensor.unsqueeze(-1)
        if not torch.isfinite(loss) or not torch.isfinite(gradients).all():
            raise ValueError("Encoder reward produced nonfinite loss or gradients")
        p = probabilities.detach().cpu().numpy()
        rows = editable if editable.any() else np.ones(n, dtype=bool)

        def weighted_mean(values: np.ndarray) -> float:
            return float(np.mean(np.sum(p[rows] * values[rows], axis=-1)))

        diagnostics = {
            "map_id": self.map_id, "map_loss": float(objective.detach()),
            "kl_loss": float(kl.detach()), "soft_map_location": self.project(embedding).detach().cpu().numpy()[0].tolist(),
            "bridge_exact": False, "bridge_method": "isolated-surface-mean-encoder-embeddings",
            "bridge_coverage": weighted_mean(present),
            "unknown_mass": weighted_mean(metadata["unknown_fraction"]),
            "special_mass": weighted_mean(metadata["special"]),
            "empty_surface_mass": weighted_mean(metadata["empty"]),
            "mask_mass": weighted_mean(metadata["mask"]),
            "one_to_many_mass": weighted_mean(metadata["encoder_token_count"] > 1),
            "topk_retained_mass": None if retained_mass is None else float(retained_mass[rows].mean()),
            "cached_token_ids": len(bridge.cache), "canvas_positions": n,
            "editable_positions": int(editable.sum()),
            "unrepresented_editable_positions": int(np.sum(editable & ~has_surface)),
            "gradient_norm": float(torch.linalg.vector_norm(gradients).detach()),
            "constraints": "fixed CLS/SEP; mask-aware mean; L2 norm; no residual or quality loss; canonical acceptance required",
        }
        return float(loss.detach()), gradients.detach().cpu().numpy(), diagnostics
