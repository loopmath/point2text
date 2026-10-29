"""Saved records use server scores and survive restart without changing the map."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from point2text.contracts import Candidate
from point2text.server import create_app
from point2text.storage import ParagraphLibrary


class Bundle:
    def __init__(self, map_id="fixed-map-a"):
        self.map_id = map_id

    def payload(self):
        return {"map_id": self.map_id, "encoder": {"id": "frozen-encoder"},
                "points": [{"id": "corpus-1", "text": "Fixed corpus.", "x": 0, "y": 0}]}


class Generator:
    identity = "fixture-generator"

    def available(self):
        return {"available": True, "model": self.identity, "digest": "fixed-digest"}


class Search:
    def run(self, bundle, target, **options):
        options["on_candidate"](Candidate("An exact verified paragraph.\nWith its original spacing.",
                                          1.25, -2.5, 3.2, .5, 42))
        return SimpleNamespace(trace={"strategy": "fixture-search", "passes": 2})


def app_at(cache_dir, map_id="fixed-map-a", search=Search):
    return create_app(bundle=Bundle(map_id), generator=Generator(), cache_dir=cache_dir,
                      search_factory=lambda bundle: search())


def completed_run(client, cache_dir, map_id="fixed-map-a"):
    run_id = client.post("/api/runs", json={"map_id": map_id, "target": [.25, .75], "candidates": 1}).json()["id"]
    for _ in range(200):
        result = client.get(f"/api/runs/{run_id}").json()
        if result["status"] != "running" and (cache_dir / "runs" / f"{run_id}.json").exists():
            return result
        time.sleep(.01)
    raise AssertionError("Fixture run did not finish and persist")


def test_save_uses_server_candidate_and_survives_restart(tmp_path):
    with TestClient(app_at(tmp_path)) as client:
        original_map = client.get("/api/map").json()
        run = completed_run(client, tmp_path)
        request = {"map_id": run["map_id"], "run_id": run["id"], "candidate_index": 0,
                   "title": "Keep this paragraph"}
        response = client.post("/api/saved", json=request)
        assert response.status_code == 201
        saved = response.json()
        assert saved["text"] == run["candidates"][0]["text"]
        assert [saved["x"], saved["y"]] == [1.25, -2.5]
        assert saved["target"] == [.25, .75]
        assert saved["encoder"] == "frozen-encoder"
        assert saved["title"] == request["title"]
        assert saved["source"]["run_id"] == run["id"]
        assert saved["source"]["generator_metadata"]["digest"] == "fixed-digest"
        assert saved["source"]["strategy"] == "fixture-search"
        assert saved["source"]["trace"]["passes"] == 2
        assert saved["created_at"] and saved["source"]["created_at"]
        assert client.post("/api/saved", json=request).json() == saved
        assert client.get("/api/saved").json()["paragraphs"] == [saved]
        assert client.get("/api/map").json() == original_map
        assert client.post("/api/saved", json={**request, "x": 999}).status_code == 422
        assert client.post("/api/saved", json={**request, "candidate_index": 50}).status_code == 404
        assert client.post("/api/saved", json={**request, "run_id": "../runs/record"}).status_code == 422

    # No in-memory run state is carried across app creation.
    with TestClient(app_at(tmp_path)) as restarted:
        assert restarted.get("/api/saved").json()["paragraphs"] == [saved]
        assert restarted.get(f"/api/runs/{run['id']}").status_code == 404
        assert restarted.post("/api/saved", json=request).json() == saved
        assert restarted.get("/api/map").json() == original_map
        assert restarted.delete(f"/api/saved/{saved['id']}?map_id=fixed-map-a").status_code == 200
        assert restarted.get("/api/saved").json()["paragraphs"] == []
    with TestClient(app_at(tmp_path)) as restarted:
        assert restarted.get("/api/saved").json()["paragraphs"] == []


def test_map_libraries_are_retained_but_never_mixed(tmp_path):
    with TestClient(app_at(tmp_path)) as first:
        run = completed_run(first, tmp_path)
        request = {"map_id": run["map_id"], "run_id": run["id"], "candidate_index": 0}
        saved_a = first.post("/api/saved", json=request).json()
    with TestClient(app_at(tmp_path, "fixed-map-b")) as second:
        assert second.get("/api/saved").json() == {"map_id": "fixed-map-b", "paragraphs": []}
        assert second.get("/api/saved?map_id=fixed-map-a").status_code == 409
        assert second.delete(f"/api/saved/{saved_a['id']}").status_code == 404
        assert second.post("/api/saved", json=request).status_code == 409
        assert second.post("/api/saved", json={**request, "map_id": "fixed-map-b"}).status_code == 409
        run_b = completed_run(second, tmp_path, "fixed-map-b")
        saved_b = second.post("/api/saved", json={"map_id": "fixed-map-b", "run_id": run_b["id"], "candidate_index": 0}).json()
        assert second.get("/api/saved").json()["paragraphs"] == [saved_b]
    with TestClient(app_at(tmp_path)) as first_again:
        assert first_again.get("/api/saved").json()["paragraphs"] == [saved_a]


def test_running_run_is_not_saved_until_its_provenance_is_final(tmp_path):
    entered = threading.Event()

    class BlockingSearch:
        def run(self, bundle, target, **options):
            entered.set()
            options["cancel_event"].wait(2)
            return SimpleNamespace(trace={})

    with TestClient(app_at(tmp_path, search=BlockingSearch)) as client:
        run = client.post("/api/runs", json={"map_id": "fixed-map-a", "target": [0, 0]}).json()
        assert entered.wait(1)
        response = client.post("/api/saved", json={"map_id": "fixed-map-a", "run_id": run["id"], "candidate_index": 0})
        assert response.status_code == 409
        assert "finish" in response.json()["detail"]
        client.delete(f"/api/runs/{run['id']}")


def test_concurrent_saves_are_atomic_and_idempotent(tmp_path):
    library = ParagraphLibrary(tmp_path / "library.sqlite3")
    run = {"id": "1" * 32, "map_id": "fixed-map-a", "encoder": "fixed-encoder",
           "target": [0, 0], "created_at": "2026-10-29T00:00:00+00:00",
           "generator": "fixture", "generator_metadata": {}, "strategy": "test",
           "status": "complete", "candidates": [Candidate("Exact text.", 1, 2, 3, .1, 42).payload()]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(lambda _: library.save(run, 0), range(12)))
    assert all(record == records[0] for record in records)
    assert library.list("fixed-map-a") == [records[0]]
    assert json.loads(json.dumps(records[0]))["x"] == 1
