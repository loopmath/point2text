"""Exercise the real background search/API without downloading a model."""

import json
import threading
import time

from fastapi.testclient import TestClient

from point2text.contracts import CorpusRecord
from point2text.contracts import Candidate
from types import SimpleNamespace
from point2text.server import create_app


class FakeBundle:
    map_id = "fixed-test-map"

    def payload(self):
        return {"map_id": self.map_id, "encoder": {"id": "fake-encoder"}}

    def nearest(self, target, k=4):
        return [CorpusRecord("1", "systems", "Replication", "A local fixture.")]

    def score(self, text, target):
        point = (float(len(text)), 0.0)
        return point, abs(point[0] - target[0])


class FakeGenerator:
    identity = "fake-generator-v1"

    def __init__(self, blocking=False):
        self.calls = 0
        self.entered = threading.Event()
        self.blocking = blocking

    def available(self):
        return {"available": True, "model": self.identity, "digest": "fixture-digest"}

    def generate(self, examples, *, seed, cancel_event):
        self.calls += 1
        self.entered.set()
        if self.blocking:
            cancel_event.wait(timeout=3)
            return ""
        return ["A short technical description.", "A longer technical paragraph about replication."][self.calls % 2]


def wait_run(client, run_id):
    for _ in range(150):
        result = client.get(f"/api/runs/{run_id}").json()
        if result["status"] != "running":
            return result
        time.sleep(0.01)
    raise AssertionError("Run did not finish")


def test_run_scores_actual_text_picks_best_and_saves_metadata(tmp_path):
    app = create_app(bundle=FakeBundle(), generator=FakeGenerator(), cache_dir=tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"map_id": "fixed-test-map", "target": [30, 0], "candidates": 2})
        assert response.status_code == 202
        run_id = response.json()["id"]
        result = wait_run(client, run_id)
        assert result["status"] == "complete"
        assert len(result["candidates"]) == 2
        assert all(c["x"] == len(c["text"]) for c in result["candidates"])
        assert result["best"]["distance"] == min(c["distance"] for c in result["candidates"])
        assert result["map_id"] == "fixed-test-map"
        assert result["generator_metadata"]["digest"] == "fixture-digest"
        # Completion is published before disk persistence; wait for the atomic record.
        path = tmp_path / "runs" / f"{run_id}.json"
        for _ in range(100):
            if path.exists():
                break
            time.sleep(0.01)
        assert json.loads(path.read_text())["best"] == result["best"]


def test_wrong_map_and_invalid_budgets_are_rejected(tmp_path):
    app = create_app(bundle=FakeBundle(), generator=FakeGenerator(), cache_dir=tmp_path)
    with TestClient(app) as client:
        assert client.post("/api/runs", json={"map_id": "old", "target": [0, 0]}).status_code == 409
        for target, count in [(["NaN", 0], 2), ([0, 0], 0), ([0, 0], 7), ([0], 2)]:
            assert client.post("/api/runs", json={"map_id": "fixed-test-map", "target": target, "candidates": count}).status_code == 422
        for updates in [0, 17, 2]:
            # The baseline has no encoder-guidance capability, even for a valid count.
            assert client.post("/api/runs", json={"map_id": "fixed-test-map", "target": [0, 0], "guidance_updates": updates}).status_code == 422


def test_cancellation_releases_single_worker_and_unknown_run_is_404(tmp_path):
    generator = FakeGenerator(blocking=True)
    app = create_app(bundle=FakeBundle(), generator=generator, cache_dir=tmp_path)
    with TestClient(app) as client:
        request = {"map_id": "fixed-test-map", "target": [0, 0], "candidates": 1}
        run_id = client.post("/api/runs", json=request).json()["id"]
        assert generator.entered.wait(1)
        assert client.post("/api/runs", json=request).status_code == 409
        assert client.delete(f"/api/runs/{run_id}").status_code == 200
        assert wait_run(client, run_id)["status"] == "cancelled"
        assert client.get("/api/runs/missing").status_code == 404


def test_unavailable_local_model_is_actionable(tmp_path):
    generator = FakeGenerator()
    generator.available = lambda: {"available": False, "error": "Start Ollama and select an installed model."}
    app = create_app(bundle=FakeBundle(), generator=generator, cache_dir=tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"map_id": "fixed-test-map", "target": [0, 0]})
        assert response.status_code == 503
        assert "Start Ollama" in response.json()["detail"]


def test_custom_strategy_records_native_trace_and_preserves_partial_error(tmp_path):
    class Strategy:
        def run(self, bundle, target, **kwargs):
            kwargs["on_candidate"](Candidate("Verified custom paragraph.", 1, 2, .3, .2, 42))
            return SimpleNamespace(trace={"strategy": "custom-diffusion", "status": "error",
                                          "error": "Second trajectory failed.", "passes": 9})
    app = create_app(bundle=FakeBundle(), generator=FakeGenerator(), cache_dir=tmp_path,
                     search_factory=lambda bundle: Strategy())
    with TestClient(app) as client:
        run_id = client.post("/api/runs", json={"map_id": "fixed-test-map", "target": [0, 0]}).json()["id"]
        result = wait_run(client, run_id)
        assert result["status"] == "error"
        assert result["best"]["text"] == "Verified custom paragraph."
        assert result["strategy"] == "custom-diffusion"
        assert result["trace"]["passes"] == 9
        assert result["error"] == "Second trajectory failed."
