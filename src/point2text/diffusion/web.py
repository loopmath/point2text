"""Use the existing browser/API with a real frozen diffusion search worker."""

from __future__ import annotations

from dataclasses import replace

from importlib import util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
from threading import Lock
import time
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

    def __init__(self, runtime: Any, *, guided: bool, injected: bool = False,
                 refinement: bool = False, epsilon: float | None = None):
        self.runtime = runtime
        self.guided = guided
        self.injected = injected
        self.refinement = refinement
        self.epsilon = epsilon

    @property
    def identity(self) -> str:
        return self.runtime.identity

    def available(self) -> dict[str, Any]:
        # Custom runtimes are an explicit injection seam for embeddings/tests.
        error = None if self.injected else _dependency_error(self.guided)
        result = {
            "available": error is None,
            "model": "DiffusionGemma · canonical refinement" if self.refinement else f"DiffusionGemma · {'guided' if self.guided else 'unguided'}",
            "runtime": self.identity,
            "revision": getattr(self.runtime, "revision", None),
            "guided": self.guided,
            "refinement": self.refinement,
            "epsilon": self.epsilon,
            "draft_streaming": True,
        }
        if error:
            result["error"] = error
        return result


class _ConfiguredSearch:
    def __init__(self, search: Any, guided: bool, proposer_factory=None):
        self.search = search
        self.guided = guided
        self.proposer_factory = proposer_factory

    def run(self, bundle: Any, target: Any, **kwargs: Any) -> Any:
        updates = kwargs.pop("guidance_updates", None)
        if updates is not None:
            if hasattr(self.search, "guidance_config"):
                self.search.guidance_config = replace(self.search.guidance_config, inner_updates=updates)
            else:
                self.search.config = replace(self.search.config, inner_updates=updates)
        epsilon = kwargs.pop("refinement_epsilon", None)
        seconds = kwargs.pop("refinement_seconds", None)
        unlimited = kwargs.pop("refinement_unlimited", False)
        restarts = kwargs.pop("refinement_restarts", None)
        restart_candidates = kwargs.pop("refinement_restart_candidates", None)
        if epsilon is not None or seconds is not None or unlimited or restarts is not None or restart_candidates is not None:
            settings = {}
            if epsilon is not None:
                settings["epsilon"] = epsilon
            if seconds is not None:
                settings["max_seconds"] = seconds
            if unlimited:
                settings["max_seconds"] = None
            if restarts is not None:
                settings["max_restarts"] = restarts
            if restart_candidates is not None:
                settings["restart_candidates"] = restart_candidates
            self.search.config = replace(self.search.config, **settings)
        if self.proposer_factory is not None:
            self.search.edit_proposer, kwargs["request"] = self.proposer_factory(
                bundle, target, self.search.config, kwargs)
        return self.search.run(bundle, target, guided=self.guided, **kwargs)


def create_diffusion_app(
    cache_dir: Path | None = None,
    guided: bool = True,
    config: GuidanceConfig | None = None,
    runtime: Any = None,
    refine: bool = False,
    refinement_config: Any = None,
    setup_profile: Any = None,
) -> Any:
    """Build the standard local UI with diffusion behind its search factory.

    The base app initializes the canonical encoder/map asynchronously. Reward
    construction is deferred until its first worker and cached by exact bundle
    identity. Refinement uses fresh per-run contextual proposal caches.
    """
    from ..server import create_app
    from .guidance import GuidanceConfig

    injected = runtime is not None and not isinstance(runtime, MLXDiffusionRuntime)
    runtime = runtime if runtime is not None else MLXDiffusionRuntime()
    config = config if config is not None else GuidanceConfig(inner_updates=16)
    prompt_policy = setup_profile.prompt_policy if setup_profile is not None else None
    if refine:
        from .refinement_contracts import RefinementConfig
        refinement_config = refinement_config or RefinementConfig(epsilon=0.01)
        if setup_profile is not None:
            refinement_config = replace(refinement_config,
                                        min_words=prompt_policy.min_words,
                                        max_words=prompt_policy.max_words,
                                        max_encoder_tokens=setup_profile.encoder.max_length)
    backend = BackendInfo(runtime, guided=guided, injected=injected,
                          refinement=refine, epsilon=refinement_config.epsilon if refine else None)
    rewards: dict[int, tuple[Any, Any]] = {}
    reward_lock = Lock()

    def proposer_factory(bundle, target, settings, options):
        from .contracts import DiffusionRequest
        from .discrete import CanonicalEditProposer
        from .plausibility import NativeReplacementProvider
        from .search import technical_prompt

        deadline = time.monotonic() + settings.max_seconds if settings.max_seconds is not None else float("inf")
        cancel = options.get("cancel_event")
        request = options.get("request") or DiffusionRequest(
            prompt=technical_prompt(bundle.nearest(target, k=getattr(prompt_policy, "neighbors", 4)), prompt_policy),
            seed=options.get("seed", 42), max_tokens=192)
        provider = NativeReplacementProvider(
            runtime, request, max_spans=4, top_k=32,
            should_stop=lambda: time.monotonic() >= deadline or
            (cancel is not None and cancel.is_set()))
        proposer = CanonicalEditProposer(bundle, allowed_replacements=provider,
                                        max_hotspots=4, allow_insertions=False,
                                        allow_deletions=False, allow_phrases=False)
        return proposer, request

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
        if refine:
            from .refinement import CanonicalRefinementSearch
            search = CanonicalRefinementSearch(runtime, reward=reward, guidance_config=config,
                                               config=refinement_config, prompt_policy=prompt_policy)
            return _ConfiguredSearch(search, guided, proposer_factory=proposer_factory)
        search = DiffusionSearch(runtime, reward, config, prompt_policy=prompt_policy) if prompt_policy is not None else DiffusionSearch(runtime, reward, config)
        return _ConfiguredSearch(search, guided)

    return create_app(
        cache_dir=cache_dir, generator=backend, search_factory=search_factory,
        setup_profile=setup_profile,
    )
