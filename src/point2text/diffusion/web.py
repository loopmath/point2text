"""Use the existing browser/API with a real frozen diffusion search worker."""

from __future__ import annotations

from dataclasses import replace

from importlib import util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
from threading import Lock
from typing import Any, TYPE_CHECKING

from .runtime import MLXDiffusionRuntime, MLX_VLM_VERSION

if TYPE_CHECKING:
    from .guidance import GuidanceConfig


def _dependency_error(guided: bool) -> str | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return "The MLX DiffusionGemma backend requires an Apple Silicon Mac."
    required = ["mlx", "mlx_vlm"] + (["torch"] if guided else [])
    missing = [name for name in required if util.find_spec(name) is None]
    if missing:
        return (
            f"Missing optional diffusion dependencies: {', '.join(missing)}. "
            f"Install mlx-vlm=={MLX_VLM_VERSION}"
            + (" and torch." if guided else ".")
        )
    try:
        installed = version("mlx-vlm")
    except PackageNotFoundError:
        installed = "unknown"
    if installed != MLX_VLM_VERSION:
        return f"Install audited mlx-vlm=={MLX_VLM_VERSION}; found {installed}."
    return None


class BackendInfo:
    """Lightweight readiness metadata; querying status never loads weights."""

    def __init__(self, runtime: Any, *, guided: bool, injected: bool = False):
        self.runtime = runtime
        self.guided = guided
        self.injected = injected

    @property
    def identity(self) -> str:
        return self.runtime.identity

    def available(self) -> dict[str, Any]:
        # Custom runtimes are an explicit injection seam for embeddings/tests.
        error = None if self.injected else _dependency_error(self.guided)
        result = {
            "available": error is None,
            "model": f"DiffusionGemma · {'guided' if self.guided else 'unguided'}",
            "runtime": self.identity,
            "revision": getattr(self.runtime, "revision", None),
            "guided": self.guided,
        }
        if error:
            result["error"] = error
        return result


class _ConfiguredSearch:
    def __init__(self, search: Any, guided: bool):
        self.search = search
        self.guided = guided

    def run(self, bundle: Any, target: Any, **kwargs: Any) -> Any:
        updates = kwargs.pop("guidance_updates", None)
        if updates is not None:
            self.search.config = replace(self.search.config, inner_updates=updates)
        return self.search.run(bundle, target, guided=self.guided, **kwargs)


def create_diffusion_app(
    cache_dir: Path | None = None,
    guided: bool = True,
    config: GuidanceConfig | None = None,
    runtime: Any = None,
) -> Any:
    """Build the standard local UI with diffusion behind its search factory.

    The base app initializes the canonical encoder/map asynchronously. Reward
    construction is deferred until its first worker and cached by exact bundle
    identity. Only DiffusionSearch performs the canonical hard-text scoring.
    """
    from ..server import create_app
    from .guidance import GuidanceConfig

    injected = runtime is not None and not isinstance(runtime, MLXDiffusionRuntime)
    runtime = runtime if runtime is not None else MLXDiffusionRuntime()
    config = config if config is not None else GuidanceConfig(inner_updates=2)
    backend = BackendInfo(runtime, guided=guided, injected=injected)
    rewards: dict[int, tuple[Any, Any]] = {}
    reward_lock = Lock()

    def search_factory(bundle: Any) -> _ConfiguredSearch:
        from .search import DiffusionSearch

        reward = None
        if guided:
            with reward_lock:
                entry = rewards.get(id(bundle))
                if entry is None or entry[0] is not bundle:
                    # Import Torch only in a guided generation worker.
                    from .encoder import TorchEmbeddingReward

                    reward = TorchEmbeddingReward(bundle.encoder, bundle)
                    rewards[id(bundle)] = (bundle, reward)
                else:
                    reward = entry[1]
        return _ConfiguredSearch(DiffusionSearch(runtime, reward, config), guided)

    return create_app(
        cache_dir=cache_dir, generator=backend, search_factory=search_factory
    )
