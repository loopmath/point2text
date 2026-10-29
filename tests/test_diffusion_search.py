import json
import threading

import numpy as np
import pytest

from point2text.contracts import CorpusRecord
from point2text.diffusion.contracts import DiffusionRequest, DiffusionResult, DiffusionStep
from point2text.diffusion.guidance import GuidanceConfig
from point2text.diffusion.search import DiffusionSearch


class Bundle:
    map_id = "frozen-fixture-map"

    def __init__(self):
        self.scored = []

    def nearest(self, target, k):
        assert k == 4
        return [CorpusRecord("1", "geometry", "Projection", "Example context.")]

    def score(self, text, target):
        self.scored.append(text)
        distance = 0.1 if text == "Near paragraph." else 2.0
        return (distance, 0.0), distance


class Reward:
    def loss_and_gradient(self, logits, ids, *args, **kwargs):
        # Favor token1, which the tiny sampler below maps to the near paragraph.
        return 1.0, np.where(ids == 1, -1.0, 1.0), {}


class TinySampler:
    identity = "tiny-diffusion-fixture"

    def __init__(self):
        self.requests = []

    def generate(self, request, *, hook=None, cancel_event=None):
        self.requests.append(request)
        logits = np.array([[[1.0, 0.8, -2.0]]], dtype=np.float32)
        state = DiffusionStep(logits, np.array([[0]]), np.array([[True]]),
                              0, 1, object())
        if hook is not None:
            logits = hook(state)
        token = int(logits[0, 0].argmax())
        return DiffusionResult("Near paragraph." if token == 1 else "Far paragraph.",
                               [token], {"sampler": "tiny-native-fixture"})


def test_guided_and_unguided_use_same_request_but_rank_only_hard_text():
    bundle = Bundle()
    runtime = TinySampler()
    config = GuidanceConfig(top_k=2, guided_steps=1, start_fraction=0,
                            step_size=2.0, max_delta=1.0)
    search = DiffusionSearch(runtime, Reward(), config)
    request = DiffusionRequest("Same prompt.", seed=123, max_tokens=1, steps=1)
    plain = search.run(bundle, (0, 0), request=request, guided=False)
    guided = search.run(bundle, (0, 0), request=request, guided=True)
    assert runtime.requests == [request, request]
    assert bundle.scored == ["Far paragraph.", "Near paragraph."]
    assert guided.trace["best_candidate"]["distance"] < plain.trace["best_candidate"]["distance"]
    assert guided.text == "Near paragraph."
    assert guided.trace["trajectories"][0]["guidance"]["guided_step_count"] == 1
    json.dumps(guided.trace, allow_nan=False)


def test_all_verified_candidates_survive_later_runtime_error():
    class FailingRuntime(TinySampler):
        def generate(self, request, **kwargs):
            if self.requests:
                raise RuntimeError("backend failed")
            return super().generate(request, **kwargs)

    observed = []
    result = DiffusionSearch(FailingRuntime()).run(
        Bundle(), (0, 0), guided=False, max_candidates=2,
        on_candidate=observed.append)
    assert len(result.trace["candidates"]) == len(observed) == 1
    assert result.trace["status"] == "error"
    assert result.trace["error"] == "backend failed"
    assert result.text == observed[0].text


def test_cancellation_preserves_previous_verified_result():
    event = threading.Event()
    runtime = TinySampler()
    result = DiffusionSearch(runtime).run(
        Bundle(), (0, 0), guided=False, max_candidates=2,
        cancel_event=event, on_candidate=lambda candidate: event.set())
    assert result.trace["status"] == "cancelled"
    assert len(result.trace["candidates"]) == 1
    assert len(runtime.requests) == 1


def test_precancelled_search_does_not_generate_or_score():
    event = threading.Event()
    event.set()
    runtime = TinySampler()
    bundle = Bundle()
    result = DiffusionSearch(runtime).run(bundle, (0, 0), guided=False,
                                           cancel_event=event)
    assert result.text == ""
    assert runtime.requests == bundle.scored == []
    assert result.trace["status"] == "cancelled"


def test_empty_and_duplicate_outputs_do_not_become_extra_candidates():
    runtime = TinySampler()
    result = DiffusionSearch(runtime).run(Bundle(), (0, 0), max_candidates=3,
                                           guided=False)
    assert len(runtime.requests) == 3
    assert [request.seed for request in runtime.requests] == [42, 43, 44]
    assert len(result.trace["candidates"]) == 1


def test_guidance_without_reward_is_an_actionable_error():
    with pytest.raises(ValueError, match="encoder reward"):
        DiffusionSearch(TinySampler()).run(Bundle(), (0, 0))
