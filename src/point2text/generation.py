"""An Ollama text-generation adapter; generation never installs or pulls models."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import re
import threading
from typing import Sequence

import httpx

from .contracts import CorpusRecord
from .profile import PromptPolicy


class GenerationError(RuntimeError):
    """An actionable failure from the configured generation backend."""


class OllamaGenerator:
    def __init__(self, base_url: str = "http://127.0.0.1:11434",
                 model: str = "gemma4:12b", *,
                 prompt_policy: PromptPolicy | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt_policy = prompt_policy or PromptPolicy()
        self._digest: str | None = None

    @property
    def identity(self) -> str:
        suffix = f"#{self._digest}" if self._digest else ""
        policy = "" if self.prompt_policy.is_default else f";{self.prompt_policy.identity}"
        return f"ollama:{self.model}@{self.base_url}{suffix}{policy}"

    def available(self) -> dict:
        """Check the daemon and its installed models without loading model weights."""
        try:
            with httpx.Client(timeout=5.0, trust_env=False) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                expected = {self.model}
                if ":" not in self.model:
                    expected.add(f"{self.model}:latest")
                installed = next((entry for entry in models
                                  if isinstance(entry, dict)
                                  and any(entry.get(key) in expected
                                          for key in ("name", "model"))), None)
                if installed is None:
                    self._digest = None
                    return {"available": False, "model": self.model,
                            "error": self._missing_model()}
                digest = installed.get("digest")
                self._digest = digest if isinstance(digest, str) else None
        except (httpx.HTTPError, ValueError, AttributeError, TypeError) as exc:
            return {"available": False, "model": self.model,
                    "error": self._connection_error(exc)}
        return {"available": True, "model": self.model, "digest": self._digest}

    def generate(self, examples: Sequence[CorpusRecord], *, seed: int,
                 cancel_event: threading.Event) -> str:
        """Produce a complete paragraph, or an empty string when cancelled.

        This synchronous interface belongs in a job worker. Internally an async
        stream lets cancellation interrupt model prefill as well as token reads.
        """
        if cancel_event.is_set():
            return ""
        payload = {
            "model": self.model,
            "system": self.prompt_policy.system_prompt(),
            "prompt": self.prompt_policy.user_prompt(examples),
            "stream": True,
            "think": False,
            "keep_alive": "5m",
            "options": {"seed": seed, "temperature": 0.8, "top_p": 0.9,
                        "num_predict": 220, "num_ctx": 4096},
        }
        try:
            return asyncio.run(self._generate(payload, cancel_event))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise GenerationError(self._missing_model()) from exc
            detail = exc.response.text[:300]
            raise GenerationError(
                f"Ollama rejected generation (HTTP {exc.response.status_code}): "
                f"{detail}. Check the configured model and Ollama version."
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationError(self._connection_error(exc)) from exc

    async def _generate(self, payload: dict,
                        cancel_event: threading.Event) -> str:
        timeout = httpx.Timeout(connect=5.0, read=180.0, write=10.0, pool=5.0)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            task = asyncio.create_task(self._stream(client, payload))
            try:
                while not task.done():
                    if cancel_event.is_set():
                        return ""
                    await asyncio.wait({task}, timeout=0.05)
                if cancel_event.is_set():
                    return ""
                return task.result()
            finally:
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
                elif not task.cancelled():
                    # Retrieve a coincident error even if cancellation won the race.
                    task.exception()

    async def _stream(self, client: httpx.AsyncClient, payload: dict) -> str:
        # Older Ollama/model combinations may reject explicit thinking control.
        # Retry only that capability error; no fallback model and no auto pull.
        for attempt in range(2):
            async with client.stream("POST", f"{self.base_url}/api/generate",
                                     json=payload) as response:
                if response.is_error:
                    await response.aread()
                    detail = response.text.lower()
                    if (attempt == 0 and response.status_code == 400
                            and "think" in detail
                            and any(word in detail for word in
                                    ("support", "unknown", "invalid"))):
                        payload = {key: value for key, value in payload.items()
                                   if key != "think"}
                        continue
                    response.raise_for_status()
                parts: list[str] = []
                done = False
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise GenerationError(
                            "Ollama returned an invalid response stream. Check "
                            "the server logs and retry."
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise GenerationError("Ollama returned an unexpected stream format.")
                    if chunk.get("error"):
                        raise GenerationError(f"Ollama generation failed: {chunk['error']}")
                    # Thinking is a separate field and is deliberately not shown.
                    text = chunk.get("response", "")
                    if not isinstance(text, str):
                        raise GenerationError("Ollama returned a non-text response.")
                    parts.append(text)
                    if chunk.get("done"):
                        done = True
                        if chunk.get("done_reason") == "length":
                            raise GenerationError(
                                "The model reached the paragraph token limit before "
                                "finishing. Retry or choose a model that follows "
                                "the requested paragraph length."
                            )
                        break
                if not done:
                    raise GenerationError(
                        "Ollama disconnected before completing the paragraph. Retry "
                        "and check the server logs if this continues."
                    )
                return self._clean_output("".join(parts))
        raise GenerationError("Ollama could not generate with this model configuration.")

    @staticmethod
    def _prompt(examples: Sequence[CorpusRecord]) -> str:
        return PromptPolicy().user_prompt(examples)

    @staticmethod
    def _clean_output(text: str) -> str:
        # Defensive support for servers that inline reasoning despite think=false.
        for tag in ("think", "analysis"):
            text = re.sub(rf"<{tag}\b[^>]*>.*?(?:</{tag}>|$)", "", text,
                          flags=re.IGNORECASE | re.DOTALL)
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = re.sub(r"^```[^\n]*\n", "", text)[:-3].strip()
        lines = text.splitlines()
        if lines and re.match(r"^#{1,6}\s", lines[0]):
            lines = lines[1:]
        return " ".join(" ".join(lines).split())

    def _missing_model(self) -> str:
        return (f"Model '{self.model}' is not installed in the Ollama server at "
                f"{self.base_url}. Choose an installed model in the application "
                "configuration, or install that model with Ollama explicitly.")

    def _connection_error(self, exc: Exception) -> str:
        if isinstance(exc, httpx.TimeoutException):
            return (f"Ollama at {self.base_url} did not respond in time. Check "
                    "the server and available memory, then retry.")
        return (f"Cannot use Ollama at {self.base_url}: {exc}. Start the Ollama "
                "application or run 'ollama serve', and check the configured "
                "server URL.")
