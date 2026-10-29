"""Choose the closest independently approved native text for each fixed click.

This is a derived selection receipt, not a new review. Local repairs must pass
the predeclared source-quality AND anchor-preservation policy; fresh restart
finals may differ from their old anchor while preserving the public source.
"""

import argparse
from hashlib import sha256
from pathlib import Path

from evaluate_fresh_basins import approved_anchors, read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, action="append", default=[])
    parser.add_argument("--refinement-review", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.refinement) != len(args.refinement_review):
        parser.error("each --refinement needs its paired --refinement-review")
    pool = read(args.pool)
    anchors = approved_anchors(pool, read(args.reviews))
    by_id = {row["id"]: row for row in anchors}
    inputs = [args.pool, args.reviews]
    changes = []
    for anchor in anchors:
        anchor["start_origin"] = {"kind": "independently_reviewed_native_coarse", "review_id": anchor.get("review_id")}
    for path, review_path in zip(args.refinement, args.refinement_review):
        report, reviews = read(path), read(review_path)
        if report["map_id"] != pool["map_id"]:
            raise ValueError("Refinement/map identity mismatch")
        verdicts = {item["review_id"]: item for item in reviews["items"]}
        inputs.extend((path, review_path))
        for run in report["runs"]:
            result = run.get("result")
            if not result:
                continue
            text = result["text"]
            digest = sha256(text.encode()).hexdigest()
            review_id = sha256((run["target_id"] + run["arm"] + digest).encode()).hexdigest()[:12]
            verdict = verdicts.get(review_id)
            if verdict is None or verdict["verdict"] != "accepted":
                continue
            if verdict["text_sha256"] != digest:
                raise ValueError("Review hash mismatch")
            if run["arm"] != "sentence-restarts" and verdict.get("anchor_claim_preservation") != "preserved":
                continue
            best = result["trace"].get("best_candidate")
            anchor = by_id[run["target_id"]]
            if best is None or (anchor["candidate"] and best["distance"] >= anchor["candidate"]["distance"]):
                continue
            if anchor["target"] != run["anchor"]["target"]:
                raise ValueError("Target coordinate mismatch")
            previous = anchor["candidate"]["distance"] if anchor["candidate"] else None
            # Public references are fixed within these native refinement runs.
            public_reference = run["anchor"].get("public_reference_text") or anchor.get("public_reference_text")
            anchor.update(candidate=best, request=result["trace"]["request"], review_id=review_id,
                          public_reference_text=public_reference,
                          start_origin={"kind": "independently_reviewed_native_refinement", "arm": run["arm"],
                                        "review_id": review_id, "receipt": str(path),
                                        "source_commit": report["provenance"]["source_commit"]})
            changes.append({"target_id": anchor["id"], "previous_distance": previous,
                            "reviewed_distance": best["distance"], "review_id": review_id})
    save(args.output, {"schema": "independently-reviewed-native-starts-v1", "map_id": pool["map_id"],
                       "anchors": anchors, "selection_changes": changes,
                       "inputs": [{"file": str(path), "sha256": sha256(path.read_bytes()).hexdigest()} for path in inputs],
                       "policy": "Canonical distance ranks only independently approved generated/refined native text; all review axes remain intact."})


if __name__ == "__main__":
    main()
