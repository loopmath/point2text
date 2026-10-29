"""Summarize archived hybrid banks and existing canonical measurements only.

This reader never loads an encoder or calls a model. SVD describes observed
2-D single-edit displacements; it says nothing about factual validity or
coverage of unobserved directions. Combination error is reported only for
verified improving combinations retained by the normal controller history.
"""

import argparse
from collections import Counter
from hashlib import sha256
import math
from pathlib import Path

import numpy as np

from evaluate_fresh_basins import read, save


def summarize(report):
    rows, all_sources = [], []
    for run in report["runs"]:
        trace = run.get("result", {}).get("trace", {})
        provider = run.get("provider_diagnostics", {})
        sources = run.get("actual_bank_sources", [])
        measurements = []
        for key, state in run.get("planner_measurements", {}).items():
            atoms = state["atoms"]
            deltas = np.array([[atom["dx"], atom["dy"]] for atom in atoms], dtype=float).reshape(-1, 2)
            singular = np.linalg.svd(deltas, compute_uv=False).tolist() if len(deltas) else []
            measurements.append({"source_sha256": key, "measured_atoms": len(atoms),
                                 "unique_edited_spans": len({(a["start"], a["end"]) for a in atoms}),
                                 "delta_singular_values": singular,
                                 "numerical_direction_rank": int(np.linalg.matrix_rank(deltas)) if len(deltas) else 0,
                                 "rank_tolerance": "numpy.linalg.matrix_rank default float64 tolerance",
                                 "measured_coordinate_count": len(state["coordinates"])})
        combo_errors = []
        for item in trace.get("history", []):
            provenance = item.get("diagnostics", {}).get("provenance", {})
            predicted = provenance.get("measured_sum_prediction")
            if predicted is not None:
                actual = [item["x"], item["y"]]
                combo_errors.append({"predicted": predicted, "verified": actual,
                    "prediction_error": math.dist(predicted, actual), "distance": item["distance"],
                    "text_sha256": sha256(item["text"].encode()).hexdigest()})
        receipts = list(provider.get("sources", {}).values())
        byte_errors = [r.get("error") for r in receipts if "byte" in str(r.get("error", "")).casefold()]
        best = trace.get("best_candidate")
        rows.append({"target_id": run["target_id"], "initial_distance": run["anchor"]["candidate"]["distance"],
            "final_mechanical_distance": best["distance"] if best else None,
            "quality_status": "unreviewed final; approved initial",
            "stop_reason": trace.get("stop_reason"), "seconds": trace.get("elapsed_seconds"),
            "hard_queries": trace.get("hard_queries"), "executed_work": trace.get("executed_work"),
            "provider_calls": provider.get("calls"), "provider_seconds": provider.get("elapsed_seconds"),
            "model_digest": provider.get("model_digest"), "raw_variants": provider.get("raw_variant_count"),
            "accepted_provider_variants": provider.get("accepted_variant_count"),
            "provider_rejections": provider.get("rejections"),
            "receipt_statuses": dict(Counter(r["status"] for r in receipts)),
            "byte_bound_errors": byte_errors,
            "token_telemetry_caveat": "unknown for oversized byte-bound failures" if byte_errors else None,
            "bank_sources": len(sources), "accepted_bank_atoms": sum(len(s["atoms"]) for s in sources),
            "unique_source_sentence_spans": len({(s["source_sha256"], tuple(a["edit_span"]))
                                                   for s in sources for a in s["atoms"]}),
            "measurement_geometry": measurements,
            "improving_combination_prediction_errors": combo_errors,
            "combination_error_coverage": "improving controller history only; other verified combinations may not be archived",
            "copy_audit": run.get("public_reference_copy_audit"), "error": run.get("error")})
        all_sources.append({"target_id": run["target_id"], "sources": sources,
                            "canonical_measurement_archive": run.get("planner_measurements", {})})
    return {"schema": "archived-sentence-bank-diagnostics-v1", "map_id": report["map_id"],
            "provenance": report["provenance"], "no_additional_encoder_or_model_queries": True,
            "rows": rows}, {"schema": "actual-sentence-bank-atoms-v1", "map_id": report["map_id"],
                            "target_blind_provider": True, "targets": all_sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atoms", type=Path, required=True)
    args = parser.parse_args()
    summary, atoms = summarize(read(args.report))
    summary["input_receipt"] = {"file": str(args.report), "sha256": sha256(args.report.read_bytes()).hexdigest()}
    save(args.output, summary)
    save(args.atoms, atoms)


if __name__ == "__main__":
    main()
