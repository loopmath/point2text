"""Opt-in real checkpoint test; never runs in the ordinary test suite.

Run with PYTHONPATH=src .venv/bin/python tests/smoke_diffusion_runtime.py
The first invocation downloads the pinned public checkpoint if necessary.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from point2text.diffusion.contracts import DiffusionRequest
from point2text.diffusion.runtime import MLXDiffusionRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/runtime-smoke.json"))
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    import mlx.core as mx

    runtime = MLXDiffusionRuntime()
    request = DiffusionRequest(
        "Write one standalone technical research paragraph explaining how sparse "
        "mixture-of-experts routing trades computation against memory. "
        "Use 80 to 110 words. Output only the paragraph.",
        seed=42, max_tokens=args.tokens, steps=args.steps,
    )
    print("Generating unguided paragraph…", flush=True)
    baseline = runtime.generate(request)
    print("Generating with identity hook…", flush=True)
    identity = runtime.generate(request, hook=lambda step: step.logits)
    identity_parity = baseline.token_ids == identity.token_ids
    if not identity_parity:
        raise AssertionError("Identity hook changed seeded token IDs")

    print("Comparing with unmodified native mlx-vlm sampler…", flush=True)
    context = runtime._prepare_prompt(request.prompt)
    mx.random.seed(request.seed)
    native_ids = runtime._model.generate(
        context,
        processor=runtime._processor,
        tokenizer=runtime.tokenizer,
        temperature=request.temperature,
        steps=request.steps,
        gen_length=request.max_tokens,
        diffusion_sampler=runtime.sampler,
        diffusion_min_canvas_length=request.max_tokens,
        diffusion_max_canvas_length=request.max_tokens,
        skip_special_token_ids=runtime.tokenizer.all_special_ids,
    )[0].tolist()
    eos = set(runtime._model.config.generation_config.get("eos_token_id", []))
    for index, token_id in enumerate(native_ids):
        if token_id in eos:
            native_ids = native_ids[:index]
            break
    native_ids, _ = runtime._strip_channel_header(native_ids)
    native_parity = baseline.token_ids == native_ids
    if not native_parity:
        raise AssertionError("Adapter changed seeded native sampler token IDs")

    print("Proving a controlled hook reaches real sampler inputs…", flush=True)
    native = importlib.import_module("mlx_vlm.generate.diffusion")
    original_sampler = native._diffusion_sample_canvas
    observed: list[int] = []
    forced_position = min(4, request.max_tokens - 1)
    forced_id = runtime.tokenizer.encode("Quantum", add_special_tokens=False)[0]

    def intervene(step):
        vocab = mx.arange(step.logits.shape[-1])
        position = mx.arange(step.logits.shape[1])[None, :, None]
        forced = mx.where(vocab == forced_id, 1000.0, -1000.0)
        return mx.where(position == forced_position, forced, step.logits)

    def observe(logits, *sampler_args):
        observed.append(int(mx.argmax(logits[0, forced_position]).item()))
        return original_sampler(logits, *sampler_args)

    native._diffusion_sample_canvas = observe
    try:
        edited = runtime.generate(request, hook=intervene)
    finally:
        native._diffusion_sample_canvas = original_sampler
    sampler_intercepted = bool(observed) and all(token_id == forced_id for token_id in observed)
    output_changed = baseline.token_ids != edited.token_ids
    if not sampler_intercepted or not output_changed:
        raise AssertionError("Controlled intervention did not reach sampler/output")
    report = {
        "identity_parity": identity_parity,
        "native_sampler_parity": native_parity,
        "controlled_hook_sampler_intercepted": sampler_intercepted,
        "controlled_hook_output_changed": output_changed,
        "forced_token_id": forced_id,
        "baseline": asdict(baseline),
        "identity": asdict(identity),
        "controlled_intervention": asdict(edited),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if isinstance(value, (bool, int))}, indent=2))
    print(f"Baseline: {baseline.text}")
    print(f"Evidence: {args.output.resolve()}")


if __name__ == "__main__":
    main()
