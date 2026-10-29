"""Optional frozen-denoiser vocabulary screen for canonical word edits."""

from __future__ import annotations

from hashlib import sha256
import re
from threading import Lock
from time import perf_counter
from typing import Any, Callable

from .contracts import DiffusionRequest


class NativeReplacementProvider:
    """Cache one-token contextual alternatives, without claiming fluent output.

    Unsupported spans and exhausted per-text budgets return an empty sequence.
    Callers must fail closed, rather than falling back to unrestricted edits.
    """

    def __init__(self, runtime: Any, request: DiffusionRequest,
                 top_k: int = 32, max_spans: int = 4,
                 should_stop: Callable[[], bool] | None = None) -> None:
        if type(top_k) is not int or not 1 <= top_k <= 256:
            raise ValueError("top_k must be in [1, 256]")
        if type(max_spans) is not int or max_spans < 1:
            raise ValueError("max_spans must be positive")
        self.runtime, self.request = runtime, request
        self.top_k, self.max_spans = top_k, max_spans
        self.should_stop = should_stop
        self._cache: dict[tuple, tuple[str, ...]] = {}
        self._used: dict[tuple, int] = {}
        self._lock = Lock()
        self._calls = self._hits = self._forwards = 0
        self._seconds = 0.0
        self._entries: list[dict[str, Any]] = []

    @property
    def trace(self) -> dict[str, Any]:
        return {
            "method": "one-token-native-denoiser-vocabulary-screen-v1",
            "runtime": self.runtime.identity,
            "calls": self._calls, "cache_hits": self._hits,
            "denoising_forwards": self._forwards, "elapsed_seconds": self._seconds,
            "top_k": self.top_k, "max_spans_per_text": self.max_spans,
            "entries": list(self._entries),
            "quality_guarantee": False, "sequence_likelihood": False,
        }

    def __call__(self, text: str, span: tuple[int, int]) -> tuple[str, ...]:
        return self.replacements(text, span)

    def replacements(self, text: str, span: tuple[int, int]) -> tuple[str, ...]:
        with self._lock:
            self._calls += 1
            span = tuple(span)
            group = (self.runtime.identity, self.request.prompt, self.request.seed, text)
            key = (*group, span, self.top_k)
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            started = perf_counter()
            if self.should_stop is not None and self.should_stop():
                words, trace = [], {"reason": "deadline_or_cancel", "denoising_forwards": 0}
            elif self._used.get(group, 0) >= self.max_spans:
                words, trace = [], {"reason": "per_text_span_budget", "denoising_forwards": 0}
            else:
                try:
                    report = self.runtime.replacement_candidates(
                        self.request, text, span, top_k=self.top_k
                    )
                    words, trace = report["replacements"], dict(report["trace"])
                    self._used[group] = self._used.get(group, 0) + int(trace.get("denoising_forwards", 0))
                except Exception as exc:
                    self._used[group] = self._used.get(group, 0) + 1
                    words, trace = [], {"reason": "native_probe_error", "error": str(exc), "denoising_forwards": 0, "failed_probe_work_unmeasured": True}
            original = text[span[0]:span[1]] if len(span) == 2 else ""
            accepted: list[str] = []
            for value in words:
                word = value.strip() if isinstance(value, str) else ""
                if not re.fullmatch(r"[A-Za-z]+", word):
                    continue
                if original.islower() and not word.islower():
                    continue
                if original.istitle() and not word.istitle():
                    continue
                if original.isupper() and not word.isupper():
                    continue
                if word not in accepted:
                    accepted.append(word)
            result = tuple(accepted)
            self._cache[key] = result
            seconds = perf_counter() - started
            self._seconds += seconds
            self._forwards += int(trace.get("denoising_forwards", 0))
            self._entries.append({"text_hash": sha256(text.encode()).hexdigest()[:16],
                                  "span": list(span), "accepted_words": len(result),
                                  "seconds": seconds, **trace})
            return result
