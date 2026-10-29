import asyncio
import json
import threading
import time

import httpx
import pytest

from point2text.contracts import CorpusRecord
from point2text.generation import GenerationError, OllamaGenerator


EXAMPLES = [CorpusRecord("1", "retrieval", "Vector search",
                         "Embedding search compares semantic vectors.")]


def bind_http(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    async_client, client = httpx.AsyncClient, httpx.Client
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs:
                        async_client(transport=transport, **kwargs))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs:
                        client(transport=transport, **kwargs))


def ndjson(*chunks):
    return "\n".join(json.dumps(chunk) for chunk in chunks) + "\n"


def test_streamed_generation_is_bounded_seeded_and_has_no_reasoning(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        payload = json.loads(request.content)
        assert request.url.path == "/api/generate"
        assert payload["model"] == "installed:test"
        assert payload["stream"] is True
        assert payload["think"] is False
        assert payload["options"]["seed"] == 23
        assert payload["options"]["num_predict"] <= 300
        assert "60–100 words" in payload["system"]
        assert "Vector search" in payload["prompt"]
        return httpx.Response(200, text=ndjson(
            {"thinking": "private chain of thought", "response": ""},
            {"response": "<think>hidden</think>\n# Heading\nSemantic "},
            {"response": "search compares meanings.", "done": True},
        ))

    bind_http(monkeypatch, handler)
    generator = OllamaGenerator("http://example.test/", "installed:test")
    result = generator.generate(EXAMPLES, seed=23, cancel_event=threading.Event())
    assert result == "Semantic search compares meanings."
    assert len(requests) == 1  # No implicit pull, show, or alternate model request.


def test_availability_records_installed_digest(monkeypatch):
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [
            {"name": "other:latest"},
            {"name": "installed:test", "digest": "abc123"},
        ]})

    bind_http(monkeypatch, handler)
    generator = OllamaGenerator(model="installed:test")
    assert generator.available() == {"available": True, "model": "installed:test",
                                     "digest": "abc123"}
    assert "abc123" in generator.identity


def test_missing_model_reports_configuration_without_pulling(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(200, json={"models": []})

    bind_http(monkeypatch, handler)
    status = OllamaGenerator(model="missing:test").available()
    assert status["available"] is False
    assert "not installed" in status["error"]
    assert requests == ["/api/tags"]


def test_missing_daemon_has_actionable_error(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("Connection refused", request=request)

    bind_http(monkeypatch, handler)
    generator = OllamaGenerator()
    assert "ollama serve" in generator.available()["error"]
    with pytest.raises(GenerationError, match="ollama serve"):
        generator.generate(EXAMPLES, seed=0, cancel_event=threading.Event())


def test_unsupported_thinking_control_retries_same_model(monkeypatch):
    payloads = []

    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if "think" in payload:
            return httpx.Response(400, json={"error": "model does not support thinking"})
        return httpx.Response(200, text=ndjson({"response": "A paragraph.", "done": True}))

    bind_http(monkeypatch, handler)
    result = OllamaGenerator().generate(EXAMPLES, seed=0, cancel_event=threading.Event())
    assert result == "A paragraph."
    assert len(payloads) == 2
    assert payloads[0]["model"] == payloads[1]["model"]
    assert "think" not in payloads[1]


@pytest.mark.parametrize("response,error", [
    (httpx.Response(404, json={"error": "model not found"}), "not installed"),
    (httpx.Response(200, text='{"response":"unfinished"}\n'), "before completing"),
    (httpx.Response(200, text="not json\n"), "invalid response stream"),
    (httpx.Response(200, text=ndjson({"error": "out of memory"})), "out of memory"),
    (httpx.Response(200, text=ndjson({"response": "Cut off", "done": True,
                                    "done_reason": "length"})), "token limit"),
])
def test_unusable_streams_are_not_accepted(monkeypatch, response, error):
    bind_http(monkeypatch, lambda request: response)
    with pytest.raises(GenerationError, match=error):
        OllamaGenerator().generate(EXAMPLES, seed=0, cancel_event=threading.Event())


def test_precancelled_generation_makes_no_request(monkeypatch):
    def handler(request):
        pytest.fail("Cancelled requests must not reach the model")

    bind_http(monkeypatch, handler)
    event = threading.Event()
    event.set()
    assert OllamaGenerator().generate(EXAMPLES, seed=0, cancel_event=event) == ""


def test_cancellation_interrupts_a_stalled_token_stream(monkeypatch):
    closed = threading.Event()
    cancel = threading.Event()

    class StalledStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b'{"response":"too late","done":true}\n'

        async def aclose(self):
            closed.set()

    bind_http(monkeypatch, lambda request: httpx.Response(200, stream=StalledStream()))
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        assert OllamaGenerator().generate(EXAMPLES, seed=0, cancel_event=cancel) == ""
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2
    assert closed.is_set()
