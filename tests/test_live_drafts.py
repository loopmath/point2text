"""Native progress remains separate from canonically scored candidates."""

import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from point2text.contracts import Candidate
from point2text.server import RunRequest, create_app


class Bundle:
    map_id = "fixed-draft-map"

    def payload(self):
        return {"map_id": self.map_id, "encoder": {"id": "fixed-encoder"}}


class Generator:
    identity = "native-fixture"

    def __init__(self, streaming=True, refinement=True):
        self.streaming, self.refinement = streaming, refinement

    def available(self):
        return {"available": True, "guided": True, "refinement": self.refinement,
                "draft_streaming": self.streaming}


def test_native_drafts_are_bounded_and_not_promoted_to_candidates(tmp_path):
    entered, release = threading.Event(), threading.Event()
    received = {}

    class Search:
        def run(self, bundle, target, **options):
            received.update(options)
            callback = options["on_draft"]
            callback({"text": "A first evolving draft.", "step": 0, "num_steps": 8,
                      "phase": "coarse", "terminal": False})
            callback({"text": "A revised evolving draft." + "x" * 25000, "step": 1,
                      "num_steps": 8, "phase": "coarse", "terminal": False})
            callback({"text": None})
            entered.set()
            release.wait(2)
            callback({"text": "The finished native draft.", "step": 7, "num_steps": 8,
                      "phase": "coarse", "terminal": True})
            options["on_candidate"](Candidate("The measured paragraph.", 1, 2, 3, .1, 42))
            return SimpleNamespace(trace={"strategy": "canonical-fixture"})

    app = create_app(bundle=Bundle(), generator=Generator(), cache_dir=tmp_path,
                     search_factory=lambda bundle: Search())
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"map_id": Bundle.map_id, "target": [0, 0],
                               "candidates": 64, "guidance_updates": 64,
                               "refinement_epsilon": .01, "refinement_unlimited": True})
        assert response.status_code == 202
        run_id = response.json()["id"]
        assert entered.wait(1)
        running = client.get(f"/api/runs/{run_id}").json()
        assert running["draft_sequence"] == 2
        assert len(running["draft"]["text"]) == 24000
        assert running["draft"]["step"] == 1 and running["draft"]["num_steps"] == 8
        assert running["draft"]["received_at"] and not running["draft"]["terminal"]
        assert running["candidates"] == [] and running["best"] is None
        assert running["refinement_unlimited"] is True
        assert running["refinement_seconds"] is None
        assert received["refinement_unlimited"] is True
        assert received["max_candidates"] == 64 and received["guidance_updates"] == 64
        release.set()
        for _ in range(100):
            finished = client.get(f"/api/runs/{run_id}").json()
            if finished["status"] != "running":
                break
            time.sleep(.01)
        assert finished["status"] == "complete"
        assert finished["draft_sequence"] == 3 and finished["draft"]["terminal"]
        assert finished["draft"]["text"] != finished["best"]["text"]
        assert finished["best"]["text"] == "The measured paragraph."


def test_progress_callback_only_for_advertised_native_backend_and_limits(tmp_path):
    received = {}

    class Search:
        def run(self, bundle, target, **options):
            received.update(options)
            options["on_candidate"](Candidate("A measured paragraph.", 1, 2, 3, .1, 42))
            return SimpleNamespace(trace={})

    app = create_app(bundle=Bundle(), generator=Generator(streaming=False, refinement=False),
                     cache_dir=tmp_path, search_factory=lambda bundle: Search())
    request = {"map_id": Bundle.map_id, "target": [0, 0]}
    assert RunRequest(**request).candidates == 6
    with TestClient(app) as client:
        assert client.post("/api/runs", json={**request, "refinement_unlimited": True}).status_code == 422
        assert client.post("/api/runs", json={**request, "candidates": 65}).status_code == 422
        assert client.post("/api/runs", json={**request, "guidance_updates": 65}).status_code == 422
        run_id = client.post("/api/runs", json=request).json()["id"]
        for _ in range(100):
            result = client.get(f"/api/runs/{run_id}").json()
            if result["status"] != "running":
                break
            time.sleep(.01)
        assert "on_draft" not in received
        assert received["max_candidates"] == 6
        assert "refinement_unlimited" not in received
        assert result["draft"] is None and result["draft_sequence"] == 0
