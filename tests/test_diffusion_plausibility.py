"""Contextual vocabulary-screen cache and strict surface filtering."""

from point2text.diffusion.contracts import DiffusionRequest
from point2text.diffusion.plausibility import NativeReplacementProvider


class FakeRuntime:
    identity = "fake-native@revision"

    def __init__(self):
        self.calls = []

    def replacement_candidates(self, request, text, span, top_k):
        self.calls.append((request.prompt, text, span, top_k))
        return {"replacements": ["rates", " rates ", "Upper", "two words", "##suffix", "<eos>"],
                "trace": {"supported": True, "denoising_forwards": 1, "probe_seconds": .01}}


def test_provider_is_contextual_cached_and_surface_conservative():
    runtime = FakeRuntime()
    provider = NativeReplacementProvider(runtime, DiffusionRequest("A fixed prompt"))
    assert provider.replacements("The model estimates costs.", (10, 19)) == ("rates",)
    assert provider("The model estimates costs.", (10, 19)) == ("rates",)
    assert len(runtime.calls) == 1
    assert runtime.calls[0][2] == (10, 19)
    assert provider.trace["cache_hits"] == 1
    assert provider.trace["denoising_forwards"] == 1
    assert not provider.trace["quality_guarantee"]


def test_per_text_probe_budget_fails_closed_and_new_text_has_new_budget():
    runtime = FakeRuntime()
    provider = NativeReplacementProvider(runtime, DiffusionRequest("Prompt"), max_spans=1)
    provider("A model estimates costs.", (8, 17))
    assert provider("A model estimates costs.", (18, 23)) == ()
    assert len(runtime.calls) == 1
    assert provider.trace["entries"][-1]["reason"] == "per_text_span_budget"
    provider("The model estimates costs.", (10, 19))
    assert len(runtime.calls) == 2


def test_native_errors_fail_closed_and_are_recorded():
    runtime = FakeRuntime()

    def fail(*args, **kwargs):
        raise RuntimeError("Fixture runtime unavailable")

    runtime.replacement_candidates = fail
    provider = NativeReplacementProvider(runtime, DiffusionRequest("Prompt"))
    assert provider("A model estimates costs.", (8, 17)) == ()
    assert provider.trace["entries"][-1]["reason"] == "native_probe_error"


def test_runtime_identity_is_part_of_cache_key():
    runtime = FakeRuntime()
    provider = NativeReplacementProvider(runtime, DiffusionRequest("Prompt"))
    provider("A model estimates costs.", (8, 17))
    runtime.identity = "fake-native@new-revision"
    provider("A model estimates costs.", (8, 17))
    assert len(runtime.calls) == 2


def test_deadline_or_cancellation_prevents_uncached_native_probe():
    runtime = FakeRuntime()
    provider = NativeReplacementProvider(runtime, DiffusionRequest("Prompt"), should_stop=lambda: True)
    assert provider("A model estimates costs.", (8, 17)) == ()
    assert not runtime.calls
    assert provider.trace["entries"][-1]["reason"] == "deadline_or_cancel"
    assert provider.trace["denoising_forwards"] == 0
