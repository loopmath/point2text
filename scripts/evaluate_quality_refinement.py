"""Stage target-only native drafts, then compare quality-constrained refiners.

Witness text defines coordinates only; it never enters generation or edit prompts.
The public development panel is not a hidden generalization benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from point2text.diffusion.refinement_contracts import RefinementConfig, TextProposal
from point2text.encoder import SentenceEncoder
from point2text.geometry import MapBundle


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def prepare(args, bundle):
    from point2text.diffusion.contracts import DiffusionRequest
    from point2text.diffusion.encoder import TorchEmbeddingReward
    from point2text.diffusion.guidance import GuidanceConfig, SparseMapGuidance
    from point2text.diffusion.runtime import MLXDiffusionRuntime
    from point2text.diffusion.search import technical_prompt
    from point2text.diffusion.verification import CanonicalVerifier
    from point2text.diffusion.work import native_work_summary

    fixtures = [json.loads(line) for line in args.fixtures.read_text().splitlines() if line]
    if args.case:
        unknown = set(args.case) - {row["id"] for row in fixtures}
        if unknown:
            raise ValueError(f"Unknown fixture IDs: {sorted(unknown)}")
        fixtures = [row for row in fixtures if row["id"] in args.case]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "map_id": bundle.map_id,
              "mode": "target_only_native_coarse_pool", "cases": []}
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text())
        if report["map_id"] != bundle.map_id:
            raise ValueError("Cannot resume on a different map")
    runtime = MLXDiffusionRuntime()
    reward = TorchEmbeddingReward(bundle.encoder, bundle)
    guidance = GuidanceConfig(inner_updates=args.inner_updates,
                              start_fraction=args.start_fraction,
                              end_fraction=args.end_fraction,
                              guided_steps=args.guided_steps)
    report["guidance_config"] = asdict(guidance)
    report["attempts_per_target"] = args.attempts
    for fixture in fixtures:
        if any(row["id"] == fixture["id"] for row in report["cases"]):
            continue
        # The optimizer receives only this point, never the witness paragraph.
        target = tuple(float(v) for v in bundle.project(bundle.encoder.encode([fixture["text"]]))[0])
        verifier = CanonicalVerifier(bundle, target, RefinementConfig(max_seconds=args.seconds))
        candidates, native = [], []

        def observe(draft):
            for value in verifier.verify_batch([TextProposal(draft.text)]):
                candidates.append(asdict(value))
            return False

        for attempt in range(args.attempts):
            request = DiffusionRequest(technical_prompt(bundle.nearest(target, k=4)),
                                       seed=args.seed + attempt, max_tokens=192)
            hook = SparseMapGuidance(reward, target, guidance)
            result = runtime.generate(request, hook=hook, observer=observe, observe_every=3)
            for value in verifier.verify_batch([TextProposal(result.text)]):
                candidates.append(asdict(value))
            native.append({"request": asdict(request), "runtime": result.trace,
                           "guidance": hook.trace})
        item = {"id": fixture["id"], "topic": fixture["topic"], "target": list(target),
                "candidates": sorted(candidates, key=lambda value: value["distance"]),
                "native": native, "verification": verifier.diagnostics}
        item["executed_work"] = native_work_summary(native)
        report["cases"].append(item)
        save(args.output, report)
        print(json.dumps({"id": fixture["id"], "valid_drafts": len(candidates),
                          "best_distance": item["candidates"][0]["distance"] if candidates else None}), flush=True)


def refine(args, bundle):
    from point2text.diffusion.combinations import MeasuredCombinationPlanner
    from point2text.diffusion.paraphrases import OllamaParaphraseProvider, ParaphraseEditProposer
    from point2text.diffusion.quality import TechnicalQualityGate
    from point2text.diffusion.refinement import CanonicalRefinementSearch

    pool = json.loads(args.pool.read_text())
    if pool["map_id"] != bundle.map_id:
        raise ValueError("Coarse pool uses a different map")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "map_id": bundle.map_id,
              "mode": "quality_constrained_target_only_refinement", "cases": [],
              "quality_note": "Target-blind local model judgments are heuristic; independent review is separate."}
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text())
        if report["map_id"] != bundle.map_id:
            raise ValueError("Cannot resume on a different map")
    for case in pool["cases"]:
        if args.case and case["id"] not in args.case:
            continue
        if any(row["id"] == case["id"] for row in report["cases"]):
            continue
        deadline = time.monotonic() + args.seconds
        stop = lambda: time.monotonic() >= deadline
        gate = TechnicalQualityGate(should_stop=stop)
        anchor = None
        anchor_checks = []
        for candidate in case["candidates"]:
            verdict = gate(candidate["text"])
            anchor_checks.append({"text": candidate["text"], "verdict": verdict})
            if verdict["accepted"]:
                anchor = candidate
                break
        item = {"id": case["id"], "topic": case["topic"], "target": case["target"],
                "anchor": anchor, "anchor_checks": anchor_checks}
        if anchor is not None:
            gate.reference_text = anchor["text"]
            config = RefinementConfig(max_seconds=max(1, deadline-time.monotonic()),
                                      max_hard_queries=args.queries, max_rounds=args.rounds,
                                      proposals_per_round=128, beam_width=1, stall_rounds=6)
            provider = OllamaParaphraseProvider(
                model=args.proposal_model, should_stop=stop, max_calls=args.bank_calls,
                max_suggestions=24, max_tokens=1600, max_total_seconds=args.seconds)
            proposer = ParaphraseEditProposer(provider=provider, should_stop=stop)
            planner = MeasuredCombinationPlanner(max_edits=args.max_edits)
            result = CanonicalRefinementSearch(
                None, edit_proposer=proposer, config=config, quality_callback=gate,
                late_quality=True, combination_planner=planner,
            ).run(bundle, tuple(case["target"]), guided=False, initial_text=anchor["text"])
            item.update(text=result.text, trace=result.trace, quality=gate.trace,
                        paraphrases=provider.diagnostics)
        else:
            item.update(text="", trace={"status": "error", "stop_reason": "no_quality_anchor"},
                        quality=gate.trace)
        report["cases"].append(item)
        save(args.output, report)
        best = item["trace"].get("best_candidate")
        print(json.dumps({"id": item["id"], "initial": anchor["distance"] if anchor else None,
                          "best": best["distance"] if best else None,
                          "reason": item["trace"]["stop_reason"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "refine"))
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parents[1]/"evaluation/epsilon_witnesses.jsonl")
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/point2text"))
    parser.add_argument("--case", action="append")
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--inner-updates", type=int, default=16)
    parser.add_argument("--start-fraction", type=float, default=0,
                        help="Guidance window start; 0 matches the browser/CLI. Older pools used .2.")
    parser.add_argument("--end-fraction", type=float, default=.9)
    parser.add_argument("--guided-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--bank-calls", type=int, default=3)
    parser.add_argument("--proposal-model", default="gemma4:e2b")
    parser.add_argument("--max-edits", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "refine" and args.pool is None:
        parser.error("refine needs --pool from the prepare phase")
    bundle = MapBundle.build(SentenceEncoder(), args.cache_dir/"map")
    (prepare if args.phase == "prepare" else refine)(args, bundle)


if __name__ == "__main__":
    main()
