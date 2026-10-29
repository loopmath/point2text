"""Conservatively audit whether a saved coarse cohort hit the old cache edge.

No model forward occurs. Reconstruct every logged curated source bank, then
enumerate a SUPERSET of its atoms and up-to-four disjoint combinations. A clean
audit needs complete source recovery and no eligible coarse duplicate outside
its current basin. Potential matches are not silently declared harmless.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
from itertools import combinations
import json
from pathlib import Path
import subprocess

from evaluate_fresh_basins import read, save
from point2text.diffusion.paraphrases import ParaphraseEditProposer


def digest(text):
    return sha256(text.encode()).hexdigest()


def audit_run(run):
    trace = run["result"]["trace"]
    wanted = set(run["proposal_bank"]["sources"])
    known, born = {}, {}

    def remember(text, basin):
        text = text.strip()
        key = digest(text)
        known[key] = text
        born[key] = min(born.get(key, basin), basin)

    remember(run["anchor"]["candidate"]["text"], 0)
    for row in run["drafts"]:
        remember(row["text"], row["basin_id"])
    for row in trace["history"]:
        remember(row["text"], row["basin_id"])
    for basin in trace["basins"]:
        if basin.get("best_candidate"):
            remember(basin["best_candidate"]["text"], basin["id"])
    proposer = ParaphraseEditProposer()
    universe = {}
    processed = {}
    while todo := {key for key in wanted & known.keys() if processed.get(key) != born[key]}:
        for key in sorted(todo):
            source = known[key]
            atoms = proposer.all_proposals(source)
            variants = [atom.text for atom in atoms]
            for count in range(2, min(4, len(atoms)) + 1):
                for choice in combinations(atoms, count):
                    ordered = sorted(choice, key=lambda atom: atom.edit_span)
                    if any(left.edit_span[1] > right.edit_span[0]
                           for left, right in zip(ordered, ordered[1:])):
                        continue
                    text = source
                    for atom in reversed(ordered):
                        start, end = atom.edit_span
                        tail = len(source) - end
                        stop = len(atom.text) - tail if tail else len(atom.text)
                        replacement = atom.text[start:stop]
                        text = text[:start] + replacement + text[end:]
                    variants.append(text)
            for text in variants:
                child = digest(text)
                universe[child] = min(universe.get(child, born[key]), born[key])
                if child in wanted:
                    remember(text, born[key])
            processed[key] = born[key]
    potential, seen = [], {digest(run["anchor"]["candidate"]["text"].strip()): 0}
    for row in run["drafts"]:
        key, basin = digest(row["text"].strip()), row["basin_id"]
        eligible = row["step"] == 0 or row["terminal"] or (row["step"] + 1) % trace["config"]["observe_every"] == 0
        if row["phase"] == "restart_coarse" and eligible:
            if (key in universe and universe[key] < basin) or (key in seen and seen[key] < basin):
                potential.append({"text_sha256": key, "basin_id": basin,
                                  "step": row["step"], "terminal": row["terminal"]})
        seen.setdefault(key, basin)
    complete = set(processed) == wanted
    return {"target_id": run["target_id"], "arm": run["arm"],
            "recorded_curated_sources": len(wanted), "recovered_sources": len(processed),
            "source_recovery_complete": complete, "candidate_superset_size": len(universe),
            "potential_cross_basin_cache_events": potential,
            "raw_quality_key_fix_applicable": trace["quality_policy"] is not None,
            "conclusion": "no_trigger_in_complete_superset" if complete and not potential else "cannot_exclude_trigger"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    args = parser.parse_args()
    report = read(args.pool)
    recorded = report["provenance"]
    original = subprocess.run(["git", "-C", str(args.source_repo), "show",
                               recorded["source_commit"] + ":src/point2text/diffusion/refinement.py"],
                              capture_output=True, check=True).stdout
    proposal_original = subprocess.run(["git", "-C", str(args.source_repo), "show",
                                        recorded["source_commit"] + ":src/point2text/diffusion/paraphrases.py"],
                                       capture_output=True, check=True).stdout
    result = {"source_commit": recorded["source_commit"],
              "recorded_refinement_sha256": recorded["refinement_sha256"],
              "code_sha_matches_recorded_commit": sha256(original).hexdigest() == recorded["refinement_sha256"],
              "proposal_code_matches_recorded_commit": proposal_original == Path(inspect.getfile(ParaphraseEditProposer)).read_bytes(),
              "method": "Complete logged-source recovery; all curated atoms and up-to-four disjoint combinations form a conservative cache-candidate superset.",
              "runs": [audit_run(run) for run in report["runs"]
                       if run.get("result") and run["arm"] == "sentence-restarts"]}
    save(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
