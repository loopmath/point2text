"""The browser adapter delegates workers without loading optional models."""

from dataclasses import dataclass
import sys
from types import ModuleType, SimpleNamespace

import pytest

from point2text.diffusion import web


@dataclass
class FakeConfig:
    inner_updates: int = 1


class FakeRuntime:
    identity = "fake-native-diffusion"
    revision = "fixture-revision"


@pytest.fixture
def components(monkeypatch):
    captured = {}
    rewards = []
    searches = []

    def create_app(**kwargs):
        captured.update(kwargs)
        return "injected-app"

    class Reward:
        def __init__(self, encoder, bundle):
            self.encoder, self.bundle = encoder, bundle
            rewards.append(self)

    class Search:
        def __init__(self, runtime, reward, config):
            self.runtime, self.reward, self.config = runtime, reward, config
            searches.append(self)

        def run(self, bundle, target, **kwargs):
            return {"bundle": bundle, "target": target, **kwargs}

    for name, values in {
        "point2text.diffusion.guidance": {"GuidanceConfig": FakeConfig},
        "point2text.diffusion.encoder": {"TorchEmbeddingReward": Reward},
        "point2text.diffusion.search": {"DiffusionSearch": Search},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("point2text.server.create_app", create_app)
    return captured, rewards, searches


def test_factory_defers_and_caches_reward_per_bundle(components, tmp_path):
    captured, rewards, searches = components
    runtime = FakeRuntime()
    assert web.create_diffusion_app(tmp_path, runtime=runtime) == "injected-app"
    assert not rewards and not searches
    assert captured["cache_dir"] == tmp_path
    assert captured["generator"].identity == runtime.identity
    assert captured["generator"].available()["model"] == "DiffusionGemma · guided"
    bundle = SimpleNamespace(encoder=object())
    factory = captured["search_factory"]
    first, second = factory(bundle), factory(bundle)
    assert len(rewards) == 1
    assert searches[0].reward is searches[1].reward is rewards[0]
    assert searches[0].config.inner_updates == 16
    assert first.run(bundle, (0.1, 0.2), seed=42)["guided"] is True
    first.run(bundle, (0.1, 0.2), seed=42, guidance_updates=8)
    assert searches[0].config.inner_updates == 8
    assert searches[1].config.inner_updates == 16
    factory(SimpleNamespace(encoder=object()))
    assert len(rewards) == 2


def test_unguided_factory_never_builds_reward_and_forwards_flag(components):
    captured, rewards, searches = components
    config = FakeConfig(inner_updates=3)
    web.create_diffusion_app(guided=False, config=config, runtime=FakeRuntime())
    bundle = SimpleNamespace(encoder=object())
    search = captured["search_factory"](bundle)
    assert not rewards
    assert searches[0].reward is None and searches[0].config is config
    result = search.run(bundle, (0, 0), seed=7, max_candidates=2)
    assert result["guided"] is False and result["max_candidates"] == 2
    assert captured["generator"].available()["model"] == "DiffusionGemma · unguided"


def test_dependency_error_is_actionable_without_model_loading(monkeypatch):
    monkeypatch.setattr(web, "_dependency_error", lambda guided: "Install optional MLX dependencies.")
    status = web.BackendInfo(FakeRuntime(), guided=True).available()
    assert not status["available"]
    assert "Install" in status["error"]
    assert status["runtime"] == FakeRuntime.identity
    assert status["revision"] == FakeRuntime.revision


def test_injected_runtime_needs_no_native_dependency_probe(monkeypatch):
    def unexpected_probe(guided):
        raise AssertionError("injected runtime should avoid native package probing")

    monkeypatch.setattr(web, "_dependency_error", unexpected_probe)
    assert web.BackendInfo(FakeRuntime(), guided=False, injected=True).available()["available"]
