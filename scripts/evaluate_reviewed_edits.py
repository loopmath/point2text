"""Compare atomic and measured-combination search from independently reviewed drafts.

Only reviewed SHA-256 anchors are eligible. Search is a numerical component
experiment; its final texts need a separate blinded review before quality claims.
No generator or local judge calls occur in this script.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path

from point2text.diffusion.combinations import MeasuredCombinationPlanner
from point2text.diffusion.paraphrases import ParaphraseEditProposer
from point2text.diffusion.refinement import CanonicalRefinementSearch
from point2text.diffusion.refinement_contracts import RefinementConfig
from point2text.encoder import SentenceEncoder
from point2text.geometry import MapBundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/point2text"))
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--queries", type=int, default=4096)
    args = parser.parse_args()
    pool = json.loads(gzip.decompress(args.pool.read_bytes())) if args.pool.suffix == ".gz" else json.loads(args.pool.read_text())
    review = json.loads(args.reviews.read_text())
    approved = {row["candidate_id"] for case in review["cases"] for row in case["candidates"]
                if row.get("accepted") is True}
    bundle = MapBundle.build(SentenceEncoder(), args.cache_dir/"map")
    if bundle.map_id != pool["map_id"]:
        raise ValueError("Reviewed draft coordinates do not match this frozen map")
    config = RefinementConfig(epsilon=1e-4, max_seconds=args.seconds,
                              max_hard_queries=args.queries, max_rounds=30,
                              beam_width=4, proposals_per_round=256, stall_rounds=6)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "map_id": bundle.map_id,
              "mode": "reviewed_anchor_curated_component_ablation",
              "quality_status": "Anchors independently reviewed; final edits pending independent review.",
              "config": asdict(config), "generator_calls": 0, "local_judge_calls": 0,
              "review_sha256": hashlib.sha256(args.reviews.read_bytes()).hexdigest(), "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in pool["cases"]:
        anchors = [row for row in case["candidates"]
                   if hashlib.sha256(row["text"].encode()).hexdigest() in approved]
        item = {"id": case["id"], "topic": case["topic"], "target": case["target"]}
        if not anchors:
            item.update(status="no_reviewed_anchor", methods={})
        else:
            anchor = min(anchors, key=lambda row: row["distance"])
            item.update(status="searched", anchor=anchor, methods={})
            for method in ("atomic", "measured_combinations"):
                planner = MeasuredCombinationPlanner(max_edits=6) if method != "atomic" else None
                result = CanonicalRefinementSearch(
                    None, edit_proposer=ParaphraseEditProposer(), config=config,
                    combination_planner=planner,
                ).run(bundle, tuple(case["target"]), initial_text=anchor["text"], guided=False)
                item["methods"][method] = {"text": result.text, "trace": result.trace}
            print(json.dumps({"id": case["id"], "initial": anchor["distance"],
                              **{key: value["trace"]["best_candidate"]["distance"]
                                 for key, value in item["methods"].items()}}), flush=True)
        report["cases"].append(item)
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
