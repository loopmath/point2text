"""Whole-word proposals from gradients at the actual canonical encoder input.

This module proposes text, not verified results. Its first-order predictions do
not certify decoded-text distance or semantic quality. The separate canonical
verifier must re-tokenize, encode, and assess every accepted candidate.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
import re
from typing import Any, Protocol

import numpy as np
import torch

from ..geometry import MapBundle
from .encoder import TorchEmbeddingReward
from .refinement_contracts import TextProposal


_WORD = re.compile(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b")
_SURFACE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*(?: [A-Za-z]+(?:[-'][A-Za-z]+)*){0,3}\Z")

# Curated grammatical families are a conservative proposal vocabulary, not a
# contextual synonym oracle. Scientific meaning still requires quality review.
_FAMILIES = (
    ("useful", "helpful", "valuable"), ("clear", "explicit", "precise"),
    ("reliable", "robust", "stable"), ("important", "essential", "critical"),
    ("different", "distinct", "varied"), ("small", "limited", "modest"),
    ("large", "substantial", "extensive"), ("observed", "measured", "recorded"),
    ("estimated", "inferred", "predicted"), ("uncertain", "unknown", "ambiguous"),
    ("computational", "numerical", "analytical"), ("physical", "experimental", "empirical"),
    ("specific", "particular", "individual"), ("relevant", "appropriate", "related"),
    ("underlying", "internal", "latent"), ("independent", "separate", "distinct"),
    ("systematic", "consistent", "structured"), ("possible", "potential", "plausible"),
    ("accurate", "precise", "reliable"), ("complex", "intricate", "complicated"),
    ("local", "regional", "spatial"), ("global", "overall", "aggregate"),
    ("continuous", "ongoing", "sustained"), ("repeated", "successive", "multiple"),
    ("use", "apply", "employ"), ("uses", "applies", "employs"),
    ("using", "applying", "employing"), ("measure", "estimate", "quantify"),
    ("measures", "estimates", "quantifies"), ("measuring", "estimating", "quantifying"),
    ("compare", "evaluate", "assess"), ("compares", "evaluates", "assesses"),
    ("comparing", "evaluating", "assessing"), ("examines", "investigates", "studies"),
    ("examine", "investigate", "study"), ("requires", "needs", "demands"),
    ("require", "need", "demand"), ("includes", "contains", "incorporates"),
    ("include", "contain", "incorporate"), ("combines", "integrates", "joins"),
    ("combine", "integrate", "join"), ("reveals", "exposes", "identifies"),
    ("reveal", "expose", "identify"), ("controls", "regulates", "modulates"),
    ("control", "regulate", "modulate"), ("changes", "alters", "modifies"),
    ("change", "alter", "modify"), ("depends", "relies"),
    ("model", "method", "framework", "procedure", "system", "process", "mechanism", "strategy"),
    ("models", "methods", "frameworks", "procedures", "systems", "processes", "mechanisms", "strategies"),
    ("analysis", "investigation", "study", "evaluation", "comparison"),
    ("analyses", "investigations", "studies", "evaluations", "comparisons"),
    ("effect", "impact", "influence"), ("effects", "impacts", "influences"),
    ("result", "finding", "outcome"), ("results", "findings", "outcomes"),
    ("evidence", "information", "knowledge"), ("features", "representations", "embeddings"),
    ("conditions", "settings", "contexts"), ("assumptions", "constraints", "requirements"),
    ("relationship", "association", "connection"), ("relationships", "associations", "connections"),
    ("observations", "measurements", "recordings"), ("variation", "variability", "uncertainty"),
    ("often", "typically", "generally"), ("carefully", "explicitly", "systematically"),
    ("directly", "independently", "separately"),
)
_REPLACEMENTS: dict[str, tuple[str, ...]] = {}
for _family in _FAMILIES:
    for _word in _family:
        _REPLACEMENTS[_word] = tuple(dict.fromkeys((*_REPLACEMENTS.get(_word, ()), *[w for w in _family if w != _word])))

_OPTIONAL = frozenset(("carefully", "explicitly", "directly", "clearly", "often", "typically",
                       "generally", "specific", "particular", "relevant", "useful", "important"))
_MODIFIED_NOUNS = frozenset(("model", "analysis", "study", "evaluation", "comparison", "measurement",
                             "method", "estimate", "experiment", "assumption", "data", "observations",
                             "results", "representation", "procedure", "system", "measurements"))
_MODIFIERS = ("careful", "explicit", "direct", "local", "independent", "controlled")
_PHRASES = {
    "in order to": ("to",), "for example": ("for instance",),
    "as a result": ("consequently",), "such as": ("including",),
    "can be used to": ("can",), "is able to": ("can",),
    "in this context": ("in this setting",), "with respect to": ("concerning",),
    "a number of": ("several",), "rather than": ("instead of",),
    "depends on": ("relies on",), "based on": ("derived from",),
}
_FINITE_VERBS = frozenset(("uses", "applies", "employs", "measures", "estimates", "quantifies",
    "compares", "evaluates", "assesses", "examines", "investigates", "studies", "requires", "needs",
    "demands", "includes", "contains", "incorporates", "combines", "integrates", "joins", "reveals",
    "exposes", "identifies", "controls", "regulates", "modulates", "changes", "alters", "modifies", "depends", "relies"))
_BASE_VERBS = frozenset(("use", "apply", "employ", "measure", "estimate", "quantify", "compare", "evaluate",
    "assess", "examine", "investigate", "study", "require", "need", "demand", "include", "contain", "incorporate",
    "combine", "integrate", "join", "reveal", "expose", "identify", "control", "regulate", "modulate", "change", "alter", "modify"))
_SINGULAR_SUBJECTS = frozenset(("model", "method", "framework", "system", "algorithm", "approach", "procedure",
    "process", "mechanism", "strategy", "analysis", "evaluation", "comparison", "study", "experiment", "controller",
    "encoder", "decoder", "network", "agent", "service", "protocol", "estimator", "detector", "measurement", "policy",
    "distribution", "relationship", "representation", "it", "this", "that", "which"))
_BASE_CONTEXT = frozenset(("to", "can", "may", "must", "should", "will", "would", "could", "might", "researchers",
    "investigators", "scientists", "we", "they", "systems", "models", "methods", "experiments", "analyses"))
_TECHNICAL_PHRASES = (
    "local minimum", "local minima", "global minimum", "global minima", "gradient estimate", "gradient estimates",
    "gradient estimator", "gradient estimators", "learning rate", "batch size", "loss function", "loss functions",
    "neural network", "neural networks", "confidence interval", "confidence intervals", "posterior distribution",
    "prior distribution", "training data", "test data", "control group", "control groups", "statistical significance",
    "random variable", "random variables", "quantum state", "quantum states", "physical system", "physical systems",
    "gene expression", "protein structure", "phase transition", "phase transitions", "climate model", "climate models",
    "distributed system", "distributed systems", "access control", "error correction", "error rate", "error rates",
    "deterministic method", "deterministic methods",
)


@dataclass(frozen=True)
class _Word:
    start: int
    end: int
    text: str
    positions: tuple[int, ...]


@dataclass(frozen=True)
class _Edit:
    start: int
    end: int
    replacement: str
    kind: str
    delta: np.ndarray
    prediction: str


@dataclass(frozen=True)
class _Analysis:
    text: str
    target: tuple[float, float]
    ids: tuple[int, ...]
    words: tuple[_Word, ...]
    inputs: np.ndarray
    jacobian: np.ndarray  # [encoder positions, map coordinates, hidden]
    point: np.ndarray
    residual: np.ndarray
    gradient_norm: float


class ReplacementProvider(Protocol):
    def replacements(self, text: str, span: tuple[int, int]) -> Sequence[str]: ...


class CanonicalEditProposer:
    """Bounded mechanically screened edits; canonical acceptance is external.

    ``allowed_replacements(word)`` optionally supplies explicit domain candidates;
    an object exposing ``replacements(text, span)`` supplies contextual candidates.
    Empty provider results fail closed, without fallback or unscreened structural
    edits. ``vocabulary`` optionally restricts candidate words.
    Defaults use curated grammatical families of technical vocabulary.
    The two-coordinate Jacobian is computed at normal encoder IDs, without the
    cross-tokenizer bridge. Same-length input-vector linearizations are precise
    derivatives, while finite edits and changed token counts remain approximations.
    """

    def __init__(self, bundle: MapBundle, *, reward: TorchEmbeddingReward | None = None,
                 allowed_replacements: Callable[[str], Sequence[str]] | ReplacementProvider | None = None,
                 vocabulary: Iterable[str] | None = None, max_hotspots: int = 12,
                 replacements_per_word: int = 8, max_pair_edits: int = 16,
                 allow_insertions: bool = True, allow_deletions: bool = True,
                 allow_phrases: bool = True) -> None:
        for name, value in (("max_hotspots", max_hotspots), ("replacements_per_word", replacements_per_word),
                            ("max_pair_edits", max_pair_edits)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if max_hotspots < 1 or replacements_per_word < 1:
            raise ValueError("Hotspot and replacement limits must be positive")
        self.bundle = bundle
        self.reward = reward or TorchEmbeddingReward(bundle.encoder, bundle)
        if self.reward.map_id != bundle.map_id:
            raise ValueError("Proposer reward must use the same frozen map")
        self.tokenizer = self.reward.tokenizer
        self.allowed_replacements = allowed_replacements
        self._provider = getattr(allowed_replacements, "replacements", None)
        if self._provider is not None and not callable(self._provider):
            raise ValueError("Contextual replacement provider must expose a callable replacements method")
        for flag in (allow_insertions, allow_deletions, allow_phrases):
            if not isinstance(flag, bool):
                raise ValueError("Structural edit flags must be boolean")
        self.allow_insertions = allow_insertions and self._provider is None
        self.allow_deletions = allow_deletions and self._provider is None
        self.allow_phrases = allow_phrases and self._provider is None
        self.vocabulary = None if vocabulary is None else frozenset(w.lower() for w in vocabulary)
        self.max_hotspots = max_hotspots
        self.replacements_per_word = replacements_per_word
        self.max_pair_edits = max_pair_edits
        self._last: _Analysis | None = None
        self._surface_cache: dict[str, tuple[tuple[int, ...], np.ndarray] | None] = {}

    def _analyze(self, text: str, target: tuple[float, float]) -> _Analysis:
        location = np.asarray(target, dtype=np.float64)
        if location.shape != (2,) or not np.isfinite(location).all():
            raise ValueError("Target must contain two finite PCA coordinates")
        target_key = (float(location[0]), float(location[1]))
        if self._last is not None and (self._last.text, self._last.target) == (text, target_key):
            return self._last
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Canonical edit source must be nonempty text")
        try:
            encoded = self.tokenizer(text, add_special_tokens=True, truncation=False,
                                     return_offsets_mapping=True, return_tensors="pt")
        except (NotImplementedError, TypeError) as exc:
            raise ValueError("Canonical edits require a tokenizer with exact character offsets") from exc
        ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long, device=self.reward.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.shape[0] != 1 or ids.shape[1] > self.reward.max_tokens:
            raise ValueError(f"Canonical edit source exceeds {self.reward.max_tokens} encoder tokens")
        offsets = np.asarray(encoded["offset_mapping"])
        if offsets.ndim == 3:
            offsets = offsets[0]
        if offsets.shape != (ids.shape[1], 2):
            raise ValueError("Encoder tokenizer returned invalid character offsets")
        mask = torch.as_tensor(encoded.get("attention_mask", torch.ones_like(ids)),
                               dtype=torch.long, device=self.reward.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        inputs = self.reward.word_embeddings(ids).detach().requires_grad_(True)
        embedding = self.reward.embed_inputs(inputs, mask)
        point = self.reward.project(embedding)[0]
        jacobians = [torch.autograd.grad(point[axis], inputs, retain_graph=axis == 0)[0][0]
                     for axis in range(2)]
        jacobian = torch.stack(jacobians, dim=1).detach().cpu().numpy().astype(np.float64)
        point_array = point.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(jacobian).all() or not np.isfinite(point_array).all():
            raise ValueError("Canonical input gradient is nonfinite")
        id_tuple = tuple(int(value) for value in ids[0].detach().cpu().tolist())
        protected = set(getattr(self.tokenizer, "all_special_ids", ()) or ())
        words = []
        for match in _WORD.finditer(text):
            positions = tuple(i for i, (start, end) in enumerate(offsets)
                              if end > start and end > match.start() and start < match.end())
            if (not positions or any(id_tuple[i] in protected for i in positions)
                    or int(offsets[positions[0], 0]) != match.start()
                    or int(offsets[positions[-1], 1]) != match.end()
                    or any(offsets[i, 0] < match.start() or offsets[i, 1] > match.end() for i in positions)):
                continue
            words.append(_Word(match.start(), match.end(), match.group(), positions))
        residual = point_array - location
        gradient = 2 * np.einsum("c,pch->ph", residual, jacobian)
        result = _Analysis(text, target_key, id_tuple, tuple(words),
                           inputs.detach().cpu().numpy()[0].astype(np.float64),
                           jacobian, point_array, residual, float(np.linalg.norm(gradient)))
        self._last = result
        return result

    def _ranked_words(self, analysis: _Analysis) -> list[_Word]:
        gradient = 2 * np.einsum("c,pch->ph", analysis.residual, analysis.jacobian)
        return sorted(analysis.words, key=lambda word: (
            -float(np.linalg.norm(gradient[list(word.positions)])), word.start))

    def hotspots(self, text: str, target: tuple[float, float], *, limit: int = 4) -> list[tuple[int, int]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Hotspot limit must be a nonnegative integer")
        if limit == 0:
            return []
        return [(word.start, word.end) for word in self._ranked_words(self._analyze(text, target))[:limit]]

    def _surface(self, text: str) -> tuple[tuple[int, ...], np.ndarray] | None:
        if text in self._surface_cache:
            return self._surface_cache[text]
        value = None
        if _SURFACE.fullmatch(text):
            ids = tuple(int(i) for i in self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
            protected = set(getattr(self.tokenizer, "all_special_ids", ()) or ())
            unknown = getattr(self.tokenizer, "unk_token_id", None)
            if ids and unknown not in ids and not any(i in protected for i in ids):
                with torch.no_grad():
                    vectors = self.reward.word_embeddings(torch.tensor(ids, dtype=torch.long, device=self.reward.device))
                value = ids, vectors.detach().cpu().numpy().astype(np.float64)
        self._surface_cache[text] = value
        return value

    @staticmethod
    def _case(source: str, replacement: str) -> str:
        if source.isupper():
            return replacement.upper()
        return replacement[:1].upper() + replacement[1:] if source[:1].isupper() else replacement

    @staticmethod
    def _article_ok(text: str, start: int, replacement: str) -> bool:
        preceding = re.search(r"\b(a|an)\s+$", text[:start], flags=re.IGNORECASE)
        if preceding is None or not replacement:
            return True
        vowel = replacement[0].lower() in "aeiou"
        return (preceding.group(1).lower() == "an") == vowel

    def _predict(self, analysis: _Analysis, positions: tuple[int, ...], replacement: str) -> tuple[np.ndarray, str] | None:
        surface = self._surface(replacement)
        if surface is None:
            return None
        _, new = surface
        old = analysis.inputs[list(positions)]
        if len(new) == len(old):
            difference = new - old
            kind = "same-length-input-linearization"
        else:
            difference = new.mean(axis=0, keepdims=True) - old
            kind = "changed-length-mean-input-approximation"
        delta = np.einsum("pch,ph->c", analysis.jacobian[list(positions)], difference)
        return delta, kind

    @staticmethod
    def _default_context_ok(analysis: _Analysis, word: _Word, replacement: str) -> bool:
        """Conservative ambiguity guard, not a parser or semantic quality model."""
        previous = [match.group().lower() for match in _WORD.finditer(analysis.text[:word.start])]
        while previous and (previous[-1].endswith("ly") or previous[-1] in {"often", "typically", "generally"}):
            previous.pop()
        preceding = previous[-1] if previous else ""
        source, candidate = word.text.lower(), replacement.lower()
        finite_context = preceding in _SINGULAR_SUBJECTS
        base_context = preceding in _BASE_CONTEXT or word.start == 0
        if source in _FINITE_VERBS:
            return (candidate in _FINITE_VERBS) == finite_context
        if source in _BASE_VERBS:
            return (candidate in _BASE_VERBS) == base_context
        if candidate in _FINITE_VERBS or candidate in _BASE_VERBS:
            return False
        # Avoid converting a noun used as a verb into another technical noun.
        if preceding in _BASE_CONTEXT and source in {"model", "process", "control"}:
            return False
        return True

    def _single_edits(self, analysis: _Analysis) -> list[_Edit]:
        edits: list[_Edit] = []
        ranked = self._ranked_words(analysis)
        protected = [(match.start(), match.end()) for phrase in _TECHNICAL_PHRASES
                     for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", analysis.text, re.IGNORECASE)]
        # Include editable technical words even if stopwords dominate raw saliency.
        eligible = [w for w in ranked if self.allowed_replacements is not None
                    or w.text.lower() in _REPLACEMENTS or w.text.lower() in _MODIFIED_NOUNS
                    or w.text.lower() in _OPTIONAL][:self.max_hotspots]
        for word in eligible:
            source = word.text.lower()
            if self.allowed_replacements is None and any(start < word.end and end > word.start for start, end in protected):
                continue
            if self._provider is not None:
                candidates = self._provider(analysis.text, (word.start, word.end))
            elif self.allowed_replacements is not None:
                candidates = self.allowed_replacements(word.text)
            else:
                candidates = _REPLACEMENTS.get(source, ())
            ranked_replacements = []
            for candidate in islice(candidates, 64):
                if not isinstance(candidate, str):
                    continue
                candidate = candidate.strip()
                if not candidate or candidate.casefold() == source:
                    continue
                if self.vocabulary is not None and any(m.group().lower() not in self.vocabulary for m in _WORD.finditer(candidate)):
                    continue
                replacement = self._case(word.text, candidate)
                if not self._article_ok(analysis.text, word.start, replacement):
                    continue
                if self.allowed_replacements is None and not self._default_context_ok(analysis, word, replacement):
                    continue
                prediction = self._predict(analysis, word.positions, replacement)
                if prediction is None:
                    continue
                delta, kind = prediction
                ranked_replacements.append(_Edit(word.start, word.end, replacement, "replace-word", delta, kind))
            ranked_replacements.sort(key=lambda edit: float(np.dot(analysis.residual + edit.delta, analysis.residual + edit.delta)))
            edits.extend(ranked_replacements[:self.replacements_per_word])
            if self.allow_deletions and source in _OPTIONAL and word.start > 0:
                end = word.end
                if analysis.text[end:end + 1] == " ":
                    end += 1
                delta = -np.einsum("pch,ph->c", analysis.jacobian[list(word.positions)],
                                    analysis.inputs[list(word.positions)])
                edits.append(_Edit(word.start, end, "", "delete-optional-word", delta, "deletion-zero-input-proxy"))
            if self.allow_insertions and source in _MODIFIED_NOUNS:
                preceding_words = list(_WORD.finditer(analysis.text[:word.start]))
                if preceding_words and preceding_words[-1].group().lower() in _MODIFIERS:
                    continue  # Do not grow modifier stacks over successive rounds.
                for modifier in _MODIFIERS:
                    if not self._article_ok(analysis.text, word.start, modifier):
                        continue
                    if analysis.text[:word.start].lower().rstrip().endswith(modifier):
                        continue
                    prediction = self._predict(analysis, word.positions, modifier + " " + word.text)
                    if prediction is not None:
                        delta, _ = prediction
                        edits.append(_Edit(word.start, word.start, modifier + " ", "insert-modifier",
                                           delta, "insertion-mean-input-approximation"))
        for phrase, alternatives in (_PHRASES.items() if self.allow_phrases else ()):
            for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", analysis.text, re.IGNORECASE):
                positions = tuple(i for word in analysis.words if word.start >= match.start() and word.end <= match.end()
                                  for i in word.positions)
                if not positions:
                    continue
                for alternative in alternatives:
                    replacement = self._case(match.group(), alternative)
                    prediction = self._predict(analysis, positions, replacement)
                    if prediction is not None:
                        delta, kind = prediction
                        edits.append(_Edit(match.start(), match.end(), replacement, "replace-short-phrase", delta, kind))
        # The same surface may belong to more than one family or rule.
        return list({(e.start, e.end, e.replacement): e for e in edits}.values())

    @staticmethod
    def _apply(text: str, edits: tuple[_Edit, ...]) -> str:
        for edit in sorted(edits, key=lambda item: (item.start, item.end), reverse=True):
            text = text[:edit.start] + edit.replacement + text[edit.end:]
        return text

    def propose(self, text: str, target: tuple[float, float], *, max_proposals: int = 64) -> list[TextProposal]:
        if isinstance(max_proposals, bool) or not isinstance(max_proposals, int) or max_proposals < 0:
            raise ValueError("Proposal budget must be a nonnegative integer")
        if max_proposals == 0:
            return []
        analysis = self._analyze(text, target)
        singles = self._single_edits(analysis)

        def score(edits: tuple[_Edit, ...]) -> float:
            residual = analysis.residual + sum((edit.delta for edit in edits), start=np.zeros(2))
            return float(np.dot(residual, residual))

        singles.sort(key=lambda edit: score((edit,)))
        pairs: list[tuple[_Edit, ...]] = []
        if self.max_pair_edits and max_proposals > 1:
            pool = singles[:64]
            for index, first in enumerate(pool):
                for second in pool[index + 1:]:
                    if first.end < second.start or second.end < first.start:
                        pairs.append((first, second))
            pairs.sort(key=score)
        pair_budget = min(self.max_pair_edits, max_proposals // 3, len(pairs))
        single_budget = max_proposals - pair_budget
        # Oversample mechanically screened candidates so collisions do not consume
        # the external hard-query budget. No encoder forwards occur in this loop.
        ranked_edits = [(edit,) for edit in singles[:single_budget * 3]]
        ranked_edits += pairs[:pair_budget * 3]
        ranked_edits.sort(key=score)
        proposals: list[TextProposal] = []
        seen = {analysis.ids}
        selected_pairs = selected_singles = 0
        for edits in ranked_edits:
            paired = len(edits) > 1
            if paired and selected_pairs >= pair_budget or not paired and selected_singles >= single_budget:
                continue
            candidate = self._apply(text, edits)
            candidate_ids = tuple(int(i) for i in self.tokenizer(candidate, add_special_tokens=True, truncation=False)["input_ids"])
            if candidate_ids in seen or len(candidate_ids) > self.reward.max_tokens:
                continue
            if getattr(self.tokenizer, "unk_token_id", None) in candidate_ids and getattr(self.tokenizer, "unk_token_id", None) not in analysis.ids:
                continue
            seen.add(candidate_ids)
            delta = sum((edit.delta for edit in edits), start=np.zeros(2))
            predicted = score(edits)
            provenance: dict[str, Any] = {
                "method": "canonical-input-gradient-discrete-v1", "map_id": self.bundle.map_id,
                "kind": "disjoint-pair" if paired else edits[0].kind,
                "source_spans": [[edit.start, edit.end] for edit in edits],
                "replacements": [edit.replacement for edit in edits],
                "predicted_map_delta": delta.tolist(), "predicted_squared_distance": predicted,
                "predicted_loss_change": predicted - float(np.dot(analysis.residual, analysis.residual)),
                "gradient_dot_embedding_change": float(2 * np.dot(analysis.residual, delta)),
                "source_map_location": analysis.point.tolist(), "source_gradient_norm": analysis.gradient_norm,
                "prediction_kinds": [edit.prediction for edit in edits],
                "lexical_policy": "contextual-provider" if self._provider is not None else (
                    "explicit-word-callback" if self.allowed_replacements is not None else "curated-technical-families"),
                "quality": "mechanical-constraints-only; canonical and semantic verification required",
                "canonical_verified": False,
            }
            proposals.append(TextProposal(candidate, (min(e.start for e in edits), max(e.end for e in edits)), provenance))
            selected_pairs += int(paired)
            selected_singles += int(not paired)
            if len(proposals) >= max_proposals:
                break
        return proposals
