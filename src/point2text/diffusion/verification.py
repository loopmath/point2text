"""Canonical real-text checks and bounded ordinary encoder queries.

Mechanical validity is not a certificate of coherence, factuality, or originality.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
import re
import time
from typing import Callable, Sequence

import numpy as np

from .refinement_contracts import RefinementConfig, TextProposal, VerifiedText


class CanonicalVerifier:
    def __init__(self, bundle, target, config: RefinementConfig | None = None, *,
                 tokenizer=None, score_batch: Callable | None = None,
                 quality_callback: Callable | None = None, clock=None) -> None:
        self.bundle = bundle
        location = np.asarray(target, dtype=float)
        if location.shape != (2,) or not np.isfinite(location).all():
            raise ValueError("Target must contain two finite canonical map coordinates")
        self.target = tuple(float(v) for v in location)
        self.config = config or RefinementConfig()
        self.clock = clock or time.monotonic
        self.started = self.clock()
        encoder = getattr(bundle, "encoder", None)
        model = getattr(encoder, "model", None)
        self.tokenizer = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
        self.score_batch = score_batch
        self.quality_callback = quality_callback
        self.cache: dict[str, VerifiedText | None] = {}
        self.token_fingerprints: dict[tuple[int, ...], str] = {}
        self.counts = Counter()
        self.rejections = Counter()
        self.confirmation: dict = {"confirmed": False, "checks": [], "worst_distance": None}
        self.last_confirmed: VerifiedText | None = None

    def remaining_queries(self, *, confirmation: bool = False) -> int:
        reserve = 0 if confirmation else 2
        return max(0, self.config.max_hard_queries - self.counts["hard_queries"] - reserve)

    def budget_reason(self, *, confirmation: bool = False) -> str | None:
        if self.clock() - self.started >= self.config.max_seconds:
            return "time_budget"
        if self.remaining_queries(confirmation=confirmation) < 1:
            return "query_budget"
        return None

    @property
    def diagnostics(self) -> dict:
        return {
            **dict(self.counts), "hard_queries": self.counts["hard_queries"],
            "accepted": self.counts["accepted"], "rejected": self.counts["rejected"],
            "rejections": dict(self.rejections), "cache_size": len(self.cache),
            "elapsed_seconds": self.clock() - self.started,
            "max_seconds": self.config.max_seconds,
            "max_hard_queries": self.config.max_hard_queries,
            "reserved_confirmation_queries": 2,
            "remaining_search_queries": self.remaining_queries(),
            "budget_reason": self.budget_reason(),
            "encoder_token_fingerprint_available": self.tokenizer is not None,
            "encoder_token_limit_checked": self.tokenizer is not None or callable(
                getattr(getattr(self.bundle, "encoder", None), "token_lengths", None)),
            "quality_validation_state": "mechanical_checks_and_external_callback"
            if self.quality_callback else "mechanical_checks_only",
            "coherence_certificate": False,
            "epsilon_confirmation": dict(self.confirmation),
        }

    def _reject(self, reason: str) -> None:
        self.counts["rejected"] += 1
        self.rejections[reason] += 1

    def _mechanical(self, text: str) -> tuple[bool, str, int]:
        if not isinstance(text, str) or not text.strip():
            return False, "empty", 0
        words = len(text.split())
        if not self.config.min_words <= words <= self.config.max_words:
            return False, "word_limit", words
        if ("\ufffd" in text or "##" in text or "```" in text
                or re.search(r"<\|[^>]*\|>|\[(?:/?INST|CLS|SEP|MASK|PAD)\]|</?(?:think|analysis|start\w*|end\w*|bos|eos|pad|mask)\b", text, re.I)
                or re.match(r"\s*(?:#{1,6}\s|(?:assistant|system|user|analysis|thought)\s*:)", text, re.I)):
            return False, "special_header_or_artifact", words
        if not re.search(r"[.!?][\s\"'’”\)\]\}]*$", text):
            return False, "unfinished_sentence", words
        if re.search(r"[.!?](?:\s*[.!?])+", text):
            return False, "punctuation_artifact", words
        # Whitespace-adjacent words only: "Data. Data analysis" is not a repeat.
        alphabetic = list(re.finditer(r"\b[^\W\d_]+\b", text, re.UNICODE))
        repeated_function_words = {
            "a", "an", "the", "of", "to", "and", "or", "in", "on", "for",
            "with", "by", "as", "at", "from", "is", "are", "be", "was",
            "were", "it", "its", "this", "these", "those", "we", "our", "data",
        }
        run = 1
        for previous, current in zip(alphabetic, alphabetic[1:]):
            same = previous.group().casefold() == current.group().casefold()
            adjacent = text[previous.end():current.start()].isspace()
            run = run + 1 if same and adjacent else 1
            if run >= 3 or run == 2 and current.group().casefold() in repeated_function_words:
                return False, "repeated_word", words
        if self.quality_callback is not None:
            try:
                verdict = self.quality_callback(text)
                if isinstance(verdict, dict):
                    accepted = bool(verdict.get("accepted", verdict.get("valid", False)))
                elif isinstance(verdict, tuple):
                    accepted = bool(verdict[0])
                else:
                    accepted = bool(verdict)
                if not accepted:
                    return False, "quality_callback", words
            except Exception:
                return False, "quality_callback_error", words
        return True, "mechanical_pass", words

    def _token_ids(self, texts: Sequence[str]) -> list[tuple[int, ...]]:
        if self.tokenizer is None:
            # Fixtures may supply only a scorer. This explicitly lacks ID dedup.
            lengths = getattr(getattr(self.bundle, "encoder", None), "token_lengths", None)
            if lengths is not None:
                counts = lengths(texts)
                if len(counts) != len(texts):
                    raise ValueError("Tokenizer returned an unexpected batch length")
                return [tuple([-1] * int(count)) for count in counts]
            return [() for _ in texts]
        self.counts["tokenizer_calls"] += 1
        encoded = self.tokenizer(list(texts), add_special_tokens=True,
                                 truncation=False, padding=False,
                                 return_attention_mask=False)
        rows = encoded["input_ids"] if isinstance(encoded, dict) or hasattr(encoded, "keys") else encoded
        if len(rows) != len(texts):
            raise ValueError("Tokenizer returned an unexpected batch length")
        return [tuple(int(token) for token in row) for row in rows]

    def _ordinary_scores(self, texts: Sequence[str]):
        if self.score_batch is not None:
            rows = self.score_batch(list(texts), self.target)
        elif hasattr(self.bundle, "encoder") and hasattr(self.bundle, "project"):
            vectors = self.bundle.encoder.encode(list(texts))
            points = np.asarray(self.bundle.project(vectors), dtype=float)
            if points.shape != (len(texts), 2):
                raise ValueError("Canonical projection returned unexpected dimensions")
            distances = np.linalg.norm(points - np.asarray(self.target), axis=1)
            rows = [(tuple(point), float(distance)) for point, distance in zip(points, distances)]
        else:
            rows = [self.bundle.score(text, self.target) for text in texts]
        rows = list(rows)
        if len(rows) != len(texts):
            raise ValueError("Canonical scorer returned an unexpected batch length")
        output = []
        for (x, y), distance in rows:
            values = tuple(float(v) for v in (x, y, distance))
            if not all(math.isfinite(v) for v in values) or values[2] < 0:
                raise ValueError("Canonical scorer returned a non-finite or negative score")
            # Compute the metric from coordinates rather than trusting a squared loss.
            measured = math.hypot(values[0] - self.target[0], values[1] - self.target[1])
            output.append((values[0], values[1], measured))
        return output

    def _query(self, texts: Sequence[str], *, confirmation: bool = False):
        if self.budget_reason(confirmation=confirmation) or len(texts) > self.remaining_queries(confirmation=confirmation):
            return None
        self.counts["hard_queries"] += len(texts)
        self.counts["batch_calls"] += 1
        if len(texts) == 1:
            self.counts["singleton_calls"] += 1
        if confirmation:
            self.counts["confirmation_queries"] += len(texts)
        return self._ordinary_scores(texts)

    def verify_batch(self, proposals: Sequence[TextProposal | str]) -> list[VerifiedText]:
        ready = []
        for proposal in proposals:
            self.counts["proposals"] += 1
            text = proposal.text if isinstance(proposal, TextProposal) else proposal
            ok, reason, words = self._mechanical(text)
            if not ok:
                self._reject(reason)
                continue
            text = text.strip()
            key = " ".join(text.split())
            if key in self.cache:
                self.counts["cache_hits"] += 1
                self._reject("duplicate_text")
                continue
            self.cache[key] = None
            provenance = proposal.provenance if isinstance(proposal, TextProposal) else {}
            ready.append((text, key, words, provenance))
        if not ready:
            return []
        try:
            token_rows = self._token_ids([row[0] for row in ready])
        except Exception:
            token_rows = []
            for text, *_ in ready:
                try:
                    token_rows.append(self._token_ids([text])[0])
                except Exception:
                    token_rows.append(None)
        selected = []
        for row, ids in zip(ready, token_rows):
            if ids is None:
                self._reject("tokenizer_error")
                continue
            if len(ids) > self.config.max_encoder_tokens:
                self._reject("encoder_token_limit")
                continue
            if self.tokenizer is not None and ids in self.token_fingerprints:
                self._reject("encoder_token_collision")
                continue
            if len(selected) >= self.remaining_queries() or self.budget_reason():
                self._reject("verification_budget")
                continue
            if self.tokenizer is not None:
                self.token_fingerprints[ids] = row[0]
            selected.append((*row, ids))
        if not selected:
            return []
        try:
            scores = self._query([row[0] for row in selected])
        except Exception:
            self.counts["backend_errors"] += 1
            scores = []
            for row in selected:
                try:
                    single = self._query([row[0]])
                    scores.append(single[0] if single else None)
                except Exception:
                    self.counts["backend_errors"] += 1
                    scores.append(None)
        if scores is None:
            scores = [None] * len(selected)
        verified = []
        for (text, key, words, provenance, ids), score in zip(selected, scores):
            if score is None:
                self._reject("canonical_scoring_error_or_budget")
                continue
            x, y, distance = score
            result = VerifiedText(text, x, y, distance,
                                  ids if self.tokenizer is not None else (),
                                  {"word_count": words, "encoder_token_count": len(ids) or None,
                                   "validation": "mechanical_only" if not self.quality_callback
                                   else "mechanical_and_external_callback",
                                   "coherence_certified": False, "provenance": provenance})
            self.cache[key] = result
            verified.append(result)
            self.counts["accepted"] += 1
        return verified

    score_many = verify_batch

    def confirm_epsilon(self, candidate: VerifiedText) -> bool:
        """Require two fresh singleton ordinary encodes; never use batch/cache values."""
        self.last_confirmed = None
        self.confirmation = {"confirmed": False, "checks": [], "worst_distance": None}
        if candidate.distance > self.config.epsilon:
            self.confirmation["reason"] = "candidate_outside_epsilon"
            return False
        if self.remaining_queries(confirmation=True) < 2 or self.budget_reason(confirmation=True):
            self.confirmation["reason"] = "confirmation_budget"
            return False
        ok, reason, _ = self._mechanical(candidate.text)
        if not ok:
            self.confirmation["reason"] = reason
            return False
        try:
            checks = []
            for _ in range(2):
                result = self._query([candidate.text], confirmation=True)
                if not result:
                    self.confirmation["reason"] = "confirmation_budget"
                    return False
                checks.append(result[0])
                self.confirmation["checks"].append({"x": result[0][0], "y": result[0][1],
                                                     "distance": result[0][2],
                                                     "cached": False, "batch_size": 1})
            worst = max(checks, key=lambda row: row[2])
            success = all(row[2] <= self.config.epsilon for row in checks)
            self.confirmation.update(confirmed=success, worst_distance=worst[2],
                                     reason="epsilon_reached" if success else "fresh_check_outside_epsilon")
            if success:
                self.last_confirmed = replace(candidate, x=worst[0], y=worst[1], distance=worst[2],
                                              diagnostics={**candidate.diagnostics,
                                                           "epsilon_confirmation": dict(self.confirmation)})
            return success
        except Exception as exc:
            self.counts["backend_errors"] += 1
            self.confirmation["reason"] = f"confirmation_error: {exc}"
            return False
