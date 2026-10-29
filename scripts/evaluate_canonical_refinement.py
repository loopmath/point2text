"""Reproduce generation or explicitly labeled local-repair diagnostics.

The witness paragraphs are evaluation fixtures, never retrieval/prompt examples.
Generation receives only their coordinates. Local repair intentionally receives a
one-word perturbation; recovering a witness is not novel target-to-text generation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from point2text.diffusion.discrete import CanonicalEditProposer
from point2text.diffusion.refinement import CanonicalRefinementSearch
from point2text.diffusion.refinement_contracts import RefinementConfig
from point2text.encoder import SentenceEncoder
from point2text.geometry import MapBundle


FIXTURES = Path(__file__).resolve().parents[1] / "evaluation" / "epsilon_witnesses.jsonl"
# Fixed before optimization. No witness-derived replacement list is supplied.
MUTATIONS = (("model", "system"), ("service", "system"),
             ("procedure", "method"), ("simulation", "model"),
             ("policy", "strategy"), ("signal", "measurement"),
             ("material", "system"), ("inference", "analysis"))


def perturb(text):
    import re
    for original, replacement in MUTATIONS:
        match = re.search(r"\b" + original + r"\b", text)
        if match:
            return text[:match.start()] + replacement + text[match.end():], {
                "original": original, "replacement": replacement,
                "span": [match.start(), match.end()],
            }
    raise ValueError("Fixture has no predeclared mutation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generation", "local-repair"), required=True)
    parser.add_argument("--case", action="append", help="Fixture ID; repeat to select several.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/point2text"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = RefinementConfig(max_seconds=args.seconds, max_hard_queries=args.queries,
                              max_rounds=args.rounds)
    fixtures = [json.loads(line) for line in FIXTURES.read_text().splitlines() if line]
    if args.case:
        unknown = set(args.case) - {row["id"] for row in fixtures}
        if unknown:
            parser.error(f"Unknown case IDs: {sorted(unknown)}")
        fixtures = [row for row in fixtures if row["id"] in args.case]
    bundle = MapBundle.build(SentenceEncoder(), args.cache_dir / "map")
    runtime = reward = None
    if args.mode == "generation":
        from point2text.diffusion.encoder import TorchEmbeddingReward
        from point2text.diffusion.runtime import MLXDiffusionRuntime
        runtime = MLXDiffusionRuntime()
        reward = TorchEmbeddingReward(bundle.encoder, bundle)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "mode": "local_repair_diagnostic" if args.mode == "local-repair" else "hidden_witness_generation",
              "map_id": bundle.map_id, "config": asdict(config),
              "constraints": "60-100 whitespace words; <=256 canonical encoder tokens",
              "quality_note": "Mechanical checks only; inspect text for coherence and scientific meaning.",
              "novelty_note": "Witness recovery in local-repair mode is not novel generation.",
              "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for fixture in fixtures:
        # This is the only use of witness text in generation mode.
        point = bundle.project(bundle.encoder.encode([fixture["text"]]))[0]
        target = tuple(float(v) for v in point)
        initial = mutation = None
        if args.mode == "local-repair":
            initial, mutation = perturb(fixture["text"])
        proposer = CanonicalEditProposer(bundle)
        if runtime is not None:
            from point2text.diffusion.contracts import DiffusionRequest
            from point2text.diffusion.plausibility import NativeReplacementProvider
            from point2text.diffusion.search import technical_prompt
            deadline = time.monotonic() + config.max_seconds
            provider = NativeReplacementProvider(
                runtime, DiffusionRequest(technical_prompt(bundle.nearest(target, k=4)), seed=args.seed, max_tokens=192),
                max_spans=4, top_k=8, should_stop=lambda: time.monotonic() >= deadline)
            proposer = CanonicalEditProposer(bundle, allowed_replacements=provider, max_hotspots=4)
        strategy = CanonicalRefinementSearch(runtime, reward=reward, edit_proposer=proposer,
                                             config=config)
        result = strategy.run(bundle, target, seed=args.seed, initial_text=initial,
                              guided=runtime is not None)
        item = {"id": fixture["id"], "target": list(target),
                "initial_text": initial, "mutation": mutation,
                "text": result.text, "trace": result.trace,
                "equals_witness": result.text == fixture["text"]}
        report["cases"].append(item)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"id": fixture["id"], "stop_reason": result.trace["stop_reason"],
                          "best": result.trace["best_candidate"],
                          "epsilon_confirmed": result.trace["epsilon_confirmed"],
                          "hard_queries": result.trace["hard_queries"]}), flush=True)


if __name__ == "__main__":
    main()
