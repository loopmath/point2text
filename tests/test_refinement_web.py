"""HTTP-to-controller lifecycle checks without optional models or native tensors."""

import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from point2text import server
from point2text.contracts import CorpusRecord
from point2text.diffusion.contracts import DiffusionDraft, DiffusionResult
from point2text.diffusion.refinement_contracts import RefinementConfig, TextProposal
from point2text.diffusion.verification import CanonicalVerifier
from point2text.diffusion.web import create_diffusion_app


class Tokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[ord(c) for c in text.lower()] for text in texts]}


class Bundle:
    map_id = "refinement-web-fixture"

    def __init__(self):
        self.encoder = self
        self.model = SimpleNamespace(tokenizer=Tokenizer())
        self.calls = []

    def nearest(self, target, k):
        return [CorpusRecord("context", "topic", "Context", "Independent neighborhood.")]

    def payload(self):
        return {"map_id": self.map_id, "encoder": {"id": "fixture-encoder"}}

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([[float(text.split()[-1]), 0] for text in texts])

    def project(self, values):
        return values


def app_fixture(monkeypatch, tmp_path, *, fail_edits=False, deadline_clock=None):
    bundle = Bundle()
    original_create = server.create_app
    monkeypatch.setattr(server, "create_app", lambda **kwargs:
                        original_create(bundle=bundle, **kwargs))

    class Runtime:
        identity = "refinement-web-native-fixture"
        revision = "fixture"

        def __init__(self):
            self.requests = []
            self.inner_updates = []

        def generate(self, request, *, observer, cancel_event, hook=None, **kwargs):
            self.requests.append(request)
            self.inner_updates.append(hook.config.inner_updates if hook else None)
            observer(DiffusionDraft("Paragraph 0.1", [], [], 0, request.steps))
            if deadline_clock is not None:
                deadline_clock[0] = 2.0
                assert cancel_event.is_set()
                return DiffusionResult("", [], {"cancelled": True, "passes": 1})
            return DiffusionResult("Paragraph 0.1", [], {"passes": 1})

    class Provider:
        trace = {"quality_guarantee": False, "native_forwards": 0}

        def __init__(self, runtime, request, **kwargs):
            self.runtime, self.request = runtime, request

    class Proposer:
        def __init__(self, bundle, *, allowed_replacements=None, **kwargs):
            self.allowed_replacements = allowed_replacements

        def propose(self, text, target, *, max_proposals):
            if fail_edits:
                raise RuntimeError("edit pipeline failed")
            return [TextProposal("Paragraph 0.00001")]

        def hotspots(self, *args, **kwargs):
            return []

    class Reward:
        def __init__(self, *args):
            pass

    for name, values in {
        "point2text.diffusion.plausibility": {"NativeReplacementProvider": Provider},
        "point2text.diffusion.discrete": {"CanonicalEditProposer": Proposer},
        "point2text.diffusion.encoder": {"TorchEmbeddingReward": Reward},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    if deadline_clock is not None:
        def verifier_factory(*args, **kwargs):
            return CanonicalVerifier(*args, **kwargs, clock=lambda: deadline_clock[0])
        monkeypatch.setattr("point2text.diffusion.refinement.CanonicalVerifier", verifier_factory)
    runtime = Runtime()
    settings = RefinementConfig(min_words=2, max_words=10, max_rounds=1,
                                max_hard_queries=24,
                                max_seconds=1 if deadline_clock is not None else 120)
    app = create_diffusion_app(cache_dir=tmp_path, guided=True, runtime=runtime,
                               refine=True, refinement_config=settings)
    return app, bundle, runtime


def finished(client, run_id):
    for _ in range(150):
        result = client.get(f"/api/runs/{run_id}").json()
        if result["status"] != "running":
            return result
        time.sleep(0.01)
    raise AssertionError("Fixture worker did not finish")


def test_http_refinement_overrides_reach_controller_and_preserve_confirmed_best(monkeypatch, tmp_path):
    app, bundle, runtime = app_fixture(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post("/api/runs", json={
            "map_id": bundle.map_id, "target": [0, 0], "candidates": 2,
            "guidance_updates": 5, "refinement_epsilon": 0.001,
            "refinement_seconds": 5,
        })
        assert response.status_code == 202
        result = finished(client, response.json()["id"])
    assert result["status"] == "complete"
    assert result["strategy"] == "canonical-feedback-refinement-v1"
    assert result["trace"]["epsilon_confirmed"] is True
    assert result["trace"]["epsilon"] == 0.001
    assert result["trace"]["config"]["max_seconds"] == 5
    assert runtime.inner_updates == [5, 5]
    assert [request.seed for request in runtime.requests] == [42, 43]
    assert result["best"] == result["trace"]["best_candidate"]
    assert bundle.calls[-2:] == [[result["best"]["text"]], [result["best"]["text"]]]
    distances = [value["distance"] for value in result["candidates"]]
    assert all(after < before for before, after in zip(distances, distances[1:]))


def test_http_worker_reports_controller_error_without_losing_verified_draft(monkeypatch, tmp_path):
    app, bundle, _ = app_fixture(monkeypatch, tmp_path, fail_edits=True)
    with TestClient(app) as client:
        response = client.post("/api/runs", json={
            "map_id": bundle.map_id, "target": [0, 0], "candidates": 1})
        result = finished(client, response.json()["id"])
    assert result["status"] == "error"
    assert result["error"] == "edit pipeline failed"
    assert result["best"]["text"] == "Paragraph 0.1"
    assert result["trace"]["best_candidate"] == result["best"]


def test_http_deadline_is_a_budget_stop_with_best_retained_not_user_cancellation(monkeypatch, tmp_path):
    app, bundle, _ = app_fixture(monkeypatch, tmp_path, deadline_clock=[0.0])
    with TestClient(app) as client:
        response = client.post("/api/runs", json={
            "map_id": bundle.map_id, "target": [0, 0], "candidates": 1})
        result = finished(client, response.json()["id"])
    assert result["status"] == "complete"
    assert result["trace"]["stop_reason"] == "budget_exhausted"
    assert result["trace"]["budget_kind"] == "time_budget"
    assert result["trace"]["epsilon_confirmed"] is False
    assert result["best"]["text"] == "Paragraph 0.1"
