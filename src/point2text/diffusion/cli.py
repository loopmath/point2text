"""Run or serve the experimental frozen diffusion path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time
from threading import Event
from uuid import uuid4

from ..server import tailnet_ipv4
from .contracts import DiffusionRequest
from .guidance import GuidanceConfig
from .runtime import DEFAULT_MODEL_ID, MLXDiffusionRuntime
from .refinement_contracts import RefinementConfig


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default=DEFAULT_MODEL_ID)
    result.add_argument("--revision", default=None)
    result.add_argument("--cache-dir", type=Path, default=Path(".cache/point2text"))
    target = result.add_mutually_exclusive_group()
    target.add_argument("--target", nargs=2, type=float, metavar=("X", "Y"))
    target.add_argument("--target-record", help="Use a bundled corpus paragraph's projected position.")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--unguided", action="store_true")
    mode.add_argument("--paired", action="store_true", help="Run the same request with guidance off and on.")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--candidates", type=int, choices=range(1, 7), default=1)
    result.add_argument("--max-tokens", type=int, choices=range(1, 255), default=None, metavar="1..254", help="Canvas length; default 192 for refinement, otherwise 128.")
    result.add_argument("--steps", type=int, choices=range(1, 257), default=32, metavar="1..256")
    result.add_argument("--temperature", type=float, default=.8)
    result.add_argument("--top-k", type=int, default=32)
    result.add_argument("--guided-steps", type=int, default=8)
    result.add_argument("--inner-updates", type=int, default=2)
    result.add_argument("--strength", type=float, default=1)
    result.add_argument("--step-size", type=float, default=4)
    result.add_argument("--start-fraction", type=float, default=0)
    result.add_argument("--end-fraction", type=float, default=.9)
    result.add_argument("--max-delta", type=float, default=2)
    result.add_argument("--output", type=Path)
    result.add_argument("--refine", action="store_true", help="Use canonical feedback and local text refinement.")
    result.add_argument("--epsilon", type=float, default=1e-4)
    result.add_argument("--refinement-seconds", type=float, default=120)
    result.add_argument("--refinement-rounds", type=int, default=20)
    result.add_argument("--max-hard-queries", type=int, default=1024)
    result.add_argument("--beam-width", type=int, default=4)
    result.add_argument("--proposals-per-round", type=int, default=64)
    result.add_argument("--repair-steps", type=int, default=8)
    result.add_argument("--initial-text-file", type=Path, help="Local refinement diagnostic seed; not a hidden-target generation test.")
    result.add_argument("--local-only", action="store_true", help="Refine an initial text without loading the diffusion model.")
    result.add_argument("--serve", action="store_true", help="Serve the experimental path using the web interface.")
    binding = result.add_mutually_exclusive_group()
    binding.add_argument("--host", default="127.0.0.1")
    binding.add_argument("--tailnet", action="store_true")
    result.add_argument("--port", type=int)
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    args.max_tokens = args.max_tokens if args.max_tokens is not None else (192 if args.refine else 128)
    try:
        config = GuidanceConfig(strength=args.strength, top_k=args.top_k,
                                guided_steps=args.guided_steps,
                                inner_updates=args.inner_updates,
                                step_size=args.step_size,
                                start_fraction=args.start_fraction,
                                end_fraction=args.end_fraction,
                                max_delta=args.max_delta)
    except ValueError as exc:
        arguments.error(str(exc))
    try:
        refinement_config = RefinementConfig(
            epsilon=args.epsilon, max_seconds=args.refinement_seconds,
            max_hard_queries=args.max_hard_queries, max_rounds=args.refinement_rounds,
            beam_width=args.beam_width, proposals_per_round=args.proposals_per_round,
            repair_steps=args.repair_steps,
        )
    except ValueError as exc:
        arguments.error(str(exc))
    if args.local_only and (not args.refine or not args.initial_text_file):
        arguments.error("--local-only requires --refine and --initial-text-file.")
    if args.initial_text_file and not args.refine:
        arguments.error("--initial-text-file requires --refine.")
    if args.local_only and args.paired:
        arguments.error("--paired needs a diffusion runtime; omit it for local-only diagnostics.")
    runtime = None if args.local_only else MLXDiffusionRuntime(args.model, args.revision)
    if args.serve:
        if args.initial_text_file or args.local_only:
            arguments.error("Initial text files and local-only diagnostics are CLI-only.")
        if args.paired:
            arguments.error("--paired is a CLI experiment; omit it when serving.")
        expected_tokens = 192 if args.refine else 128
        if args.max_tokens != expected_tokens or args.steps != 32 or args.temperature != .8:
            arguments.error(f"This browser backend uses {expected_tokens} tokens, 32 steps, temperature 0.8; custom settings are CLI-only.")
        if args.tailnet:
            try:
                args.host = tailnet_ipv4()
            except RuntimeError as exc:
                arguments.error(str(exc))
        from .web import create_diffusion_app
        import uvicorn
        uvicorn.run(create_diffusion_app(cache_dir=args.cache_dir, guided=not args.unguided,
                                         config=config, runtime=runtime,
                                         refine=args.refine, refinement_config=refinement_config),
                    host=args.host, port=args.port or (8479 if args.refine else 8478))
        return

    from ..encoder import SentenceEncoder
    from ..geometry import MapBundle
    from .encoder import TorchEmbeddingReward
    from .search import DiffusionSearch, technical_prompt

    encoder = SentenceEncoder()
    bundle = MapBundle.build(encoder, args.cache_dir / "map")
    data = bundle.payload()
    if args.target:
        target = tuple(args.target)
    else:
        points = data["points"]
        record = next((p for p in points if p["id"] == args.target_record), None) if args.target_record else points[0]
        if record is None:
            arguments.error(f"Unknown corpus record: {args.target_record}")
        target = (record["x"], record["y"])
    request = DiffusionRequest(prompt=technical_prompt(bundle.nearest(target, k=4)),
                               seed=args.seed, max_tokens=args.max_tokens,
                               steps=args.steps, temperature=args.temperature)
    reward = TorchEmbeddingReward(encoder, bundle) if not args.unguided and runtime is not None else None
    initial_text = args.initial_text_file.read_text().strip() if args.initial_text_file else None
    proposer = None
    if args.refine:
        from .discrete import CanonicalEditProposer
        from .refinement import CanonicalRefinementSearch
        if runtime is None:
            proposer = CanonicalEditProposer(bundle)
    cancel = Event()
    previous = signal.signal(signal.SIGINT, lambda *unused: cancel.set())
    record = {"created_at": datetime.now(timezone.utc).isoformat(),
              "map_id": bundle.map_id, "target": list(target),
              "guidance_config": asdict(config), "runs": {},
              "refinement_config": asdict(refinement_config) if args.refine else None,
              "experiment_mode": "local_repair_diagnostic" if args.local_only else "canonical_refinement" if args.refine else "independent_diffusion",
              "comparison_note": "Paired requests are a plumbing probe, not an equal-wall-clock quality benchmark."}
    try:
        modes = [False, True] if args.paired else [not args.unguided and runtime is not None]
        for guided in modes:
            if cancel.is_set():
                break
            if args.refine and runtime is not None:
                from .plausibility import NativeReplacementProvider
                deadline = time.monotonic() + refinement_config.max_seconds
                provider = NativeReplacementProvider(
                    runtime, request, max_spans=4, top_k=32,
                    should_stop=lambda: cancel.is_set() or time.monotonic() >= deadline)
                proposer = CanonicalEditProposer(bundle, allowed_replacements=provider,
                                                 max_hotspots=4)
            strategy = CanonicalRefinementSearch(runtime, reward=reward, guidance_config=config,
                                                edit_proposer=proposer, config=refinement_config) if args.refine else DiffusionSearch(runtime, reward, config)
            options = {"initial_text": initial_text} if args.refine else {}
            result = strategy.run(
                bundle, target, max_candidates=args.candidates, seed=args.seed,
                cancel_event=cancel, request=request, guided=guided,
                **options,
            )
            name = "guided" if guided else "unguided"
            record["runs"][name] = {"text": result.text, "token_ids": result.token_ids,
                                   "trace": result.trace}
            print(json.dumps({"mode": name, "text": result.text,
                              "best_candidate": result.trace["best_candidate"],
                              "stop_reason": result.trace.get("stop_reason"),
                              "epsilon_confirmed": result.trace.get("epsilon_confirmed"),
                              "elapsed_seconds": result.trace["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    finally:
        signal.signal(signal.SIGINT, previous)
        output = args.output or args.cache_dir / "diffusion-runs" / f"{uuid4().hex}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(f"Experiment record: {output}", flush=True)


if __name__ == "__main__":
    main()
