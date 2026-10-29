"""Enumerate saved disjoint sentence edits without any encoder/model queries.

The separately declared analysis compares the finite universe to controller
counts. Exact per-query text membership was not retained. Exhaustion is a
cardinality inference conditional on the frozen native-disabled controller
only querying its initial text, supplied atoms and measured disjoint unions.
"""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path

from evaluate_fresh_basins import read, save


def enumerate_run(run):
    texts = {}
    sources = []

    def add(text, origin):
        texts.setdefault(text, []).append(origin)

    add(run["anchor"]["candidate"]["text"], {"kind": "approved_initial"})
    for source in run["actual_bank_sources"]:
        groups = {}
        for atom in source["atoms"]:
            span = tuple(atom["edit_span"])
            groups.setdefault(span, set()).add(atom["provenance"]["replacements"][0])
        spans = sorted(groups)
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("Exact sentence enumeration requires disjoint source spans")
        counts = Counter()
        unique = set()
        options = [[None] + [(span, value) for value in sorted(groups[span])] for span in spans]
        for choices in product(*options):
            edits = [edit for edit in choices if edit is not None]
            if len(edits) > 4:
                continue
            text = source["text"]
            for (start, end), replacement in sorted(edits, reverse=True):
                text = text[:start] + replacement + text[end:]
            counts[len(edits)] += 1
            unique.add(text)
            add(text, {"kind": "fixed-bank-disjoint-composition", "source_sha256": source["source_sha256"],
                       "edit_count": len(edits), "spans": [list(span) for span, _ in edits],
                       "replacements": [replacement for _, replacement in edits]})
        sources.append({"source_sha256": source["source_sha256"],
                        "accepted_atoms": len(source["atoms"]), "eligible_sentence_spans": len(spans),
                        "compositions_by_edit_count": dict(counts), "distinct_texts": len(unique)})
    trace = run["result"]["trace"]
    verifier = trace["verification"]
    coverage = run["saved_bank_coverage"]
    count_equal = all(verifier[key] == len(texts) for key in ("hard_queries", "accepted", "cache_size"))
    coverage_equal = not coverage["visited_unavailable_source_hashes"] and not coverage["unvisited_saved_source_hashes"]
    structural_checks = {"hard_queries_accepted_cache_size_equal_finite_cardinality": count_equal,
                         "no_rejected_queries": verifier["rejected"] == 0,
                         "no_native_trajectories": trace["executed_work"]["trajectories"] == 0,
                         "no_epsilon_confirmation_queries": not trace["epsilon_confirmation"]["checks"],
                         "all_saved_sources_visited_without_unknown_sources": coverage_equal}
    return {"target_id": run["target_id"], "arm": run["arm"],
            "distinct_available_texts": len(texts), "actual_hard_queries": trace["hard_queries"],
            "within_original_query_budget_with_two_confirmations_reserved": len(texts) + 2 <= run["config"]["max_hard_queries"],
            "available_texts_with_multiedit_derivation": sum(any(o.get("edit_count", 0) >= 2 for o in origins)
                                                          for origins in texts.values()),
            "sources": sources, "structural_checks": structural_checks,
            "cardinality_coverage_inference": "exhausted" if all(structural_checks.values()) else "not_established",
            "direct_per_query_text_ledger_available": False,
            "inference_condition": "The frozen controller queries only the initial text, supplied atoms and disjoint combinations from these banks. Equality of unique-query count and finite-universe cardinality then implies full coverage. This is not an independent per-text membership audit.",
            "texts": [{"text": text, "text_sha256": sha256(text.encode()).hexdigest(), "origins": origins}
                      for text, origins in texts.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = read(args.report)
    save(args.output, {"schema": "exact-fixed-bank-enumeration-analysis-v1",
        "declared_analysis": "Complete textual enumeration of zero-to-four disjoint saved sentence variants; no new scoring arm.",
        "map_id": report["map_id"], "new_encoder_queries": 0, "new_model_calls": 0,
        "input": {"file": str(args.report), "sha256": sha256(args.report.read_bytes()).hexdigest()},
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [enumerate_run(run) for run in report["runs"]]})


if __name__ == "__main__":
    main()
