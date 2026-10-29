"""CPU-only known-bank local reconstruction diagnostic, never novel generation.

Construct a witness with deterministic curated style edits, then give a fresh
normal refinement search only the original paragraph and the witness's canonical
coordinates. Witness construction is outside the optimizer's query budget and
its selected edits are not injected into the optimizer. Mechanical screening and
coordinate confirmation do not establish preservation of scientific meaning.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import time

from point2text.diffusion.combinations import MeasuredCombinationPlanner
from point2text.diffusion.paraphrases import ParaphraseEditProposer
from point2text.diffusion.refinement import CanonicalRefinementSearch
from point2text.diffusion.refinement_contracts import RefinementConfig, TextProposal
from point2text.diffusion.verification import CanonicalVerifier
from point2text.encoder import SentenceEncoder
from point2text.geometry import MapBundle


SELECTION_ALGORITHM = (
    "short-curated-independent-spans-v1: retain single-span curated proposals "
    "with <=5 source words, <=6 replacement words, <=2 word-count change and "
    "different complete canonical input IDs; sort by absolute word-count "
    "change, total source+replacement words, start, end, replacement; greedily "
    "take nonoverlapping spans until requested count; require at least two"
)


def select_edits(source, proposals, token_ids, *, count=3):
    """Select from an ordinary bank, without encoder coordinates or a target."""
    source_ids = tuple(token_ids([source])[0])
    ranked = []
    for proposal in proposals:
        spans = proposal.provenance.get("source_spans", [])
        replacements = proposal.provenance.get("replacements", [])
        if (not str(proposal.provenance.get("origin", "")).startswith("curated")
                or len(spans) != 1 or len(replacements) != 1):
            continue
        start, end = spans[0]
        replacement = replacements[0]
        original = source[start:end]
        old_words, new_words = len(original.split()), len(replacement.split())
        if (old_words > 5 or new_words > 6 or abs(new_words - old_words) > 2
                or source[:start] + replacement + source[end:] != proposal.text
                or tuple(token_ids([proposal.text])[0]) == source_ids):
            continue
        ranked.append((abs(new_words - old_words), old_words + new_words,
                       start, end, replacement, proposal))
    selected = []
    for *_, proposal in sorted(ranked, key=lambda row: row[:-1]):
        start, end = proposal.provenance["source_spans"][0]
        # Conservatively exclude shared endpoints for insertions as well.
        if any(not (end < row["span"][0] or row["span"][1] < start) for row in selected):
            continue
        selected.append({"span": [start, end], "original": source[start:end],
                         "replacement": proposal.provenance["replacements"][0],
                         "bank_index": proposal.provenance.get("bank_index"),
                         "origin": proposal.provenance.get("origin"),
                         "single_edit_ids_differ": True})
        if len(selected) == count:
            break
    return sorted(selected, key=lambda row: row["span"])


def apply_edits(source, selected):
    result = source
    for row in reversed(selected):
        start, end = row["span"]
        result = result[:start] + row["replacement"] + result[end:]
    return result


def load_frozen_bundle(encoder, cache_dir):
    """Use MapBundle's normal cache validation, but fail before any PCA refit."""
    class CacheOnlyEncoder:
        identity = encoder.identity

        def encode(self, texts):
            raise RuntimeError("A valid frozen map cache is required; this diagnostic never fits a map")

    return replace(MapBundle.build(CacheOnlyEncoder(), cache_dir), encoder=encoder)


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path,
                        default=Path(__file__).resolve().parents[1] / "evaluation/epsilon_witnesses.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/point2text"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--witness-edits", type=int, choices=(2, 3), default=3)
    parser.add_argument("--max-edits", type=int, choices=(4, 6), default=4)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--proposals", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    import torch
    torch.set_num_threads(args.torch_threads)
    config = RefinementConfig(epsilon=1e-4, max_seconds=args.seconds, max_hard_queries=args.queries,
                              max_rounds=args.rounds, proposals_per_round=args.proposals,
                              beam_width=1, stall_rounds=6, min_words=60, max_words=100)
    fixtures = [json.loads(line) for line in args.fixtures.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.case:
        unknown = set(args.case) - {row["id"] for row in fixtures}
        if unknown:
            parser.error(f"Unknown fixture IDs: {sorted(unknown)}")
        fixtures = [row for row in fixtures if row["id"] in args.case]
    bundle = load_frozen_bundle(SentenceEncoder(), args.cache_dir / "map")
    modules = ("point2text.encoder", "point2text.geometry", "point2text.diffusion.paraphrases",
               "point2text.diffusion.combinations", "point2text.diffusion.refinement",
               "point2text.diffusion.refinement_contracts", "point2text.diffusion.verification")
    implementation = {name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
                      for name in modules}
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "mode": "known_curated_bank_local_reconstruction_diagnostic",
              "novel_generation": False, "arbitrary_click_attainability_claim": False,
              "quality_status": "unreviewed; mechanical validity is not semantic preservation",
              "optimizer_inputs": "source text, target coordinates, seed, normal fresh curated bank and configuration only",
              "selection_algorithm": SELECTION_ALGORITHM, "selection_seed": None,
              "search_seed": args.seed, "requested_witness_edits": args.witness_edits,
              "planner_max_edits": args.max_edits, "config": asdict(config),
              "map_id": bundle.map_id, "encoder_id": bundle.encoder.identity,
              "map_policy": "validated frozen cache only; encoder is blocked during map loading to prevent refitting",
              "implementation_sha256": implementation,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
              "preparation_queries_outside_search_budget": True,
              "cases": []}
    save(args.output, report)
    for fixture in fixtures:
        started = time.monotonic()
        source = fixture["text"]
        item = {"id": fixture["id"], "topic": fixture.get("topic"), "source": source,
                "quality_status": "unreviewed", "search_seed": args.seed,
                "source_word_count": len(source.split())}
        try:
            construction = ParaphraseEditProposer()
            bank = construction.all_proposals(source)
            selected = select_edits(source, bank, bundle.encoder.token_ids, count=args.witness_edits)
            item.update(bank_count=len(bank), distinct_bank_spans=len({p.edit_span for p in bank}),
                        selection=selected, selected_edit_count=len(selected),
                        construction_bank_diagnostics=construction.diagnostics)
            if len(selected) < 2:
                item.update(status="selection_failure", failure="fewer_than_two_independent_token_changing_modest_curated_spans")
            else:
                witness = apply_edits(source, selected)
                source_ids, witness_ids = bundle.encoder.token_ids([source, witness])
                item.update(witness=witness, witness_word_count=len(witness.split()),
                            source_encoder_token_ids=source_ids, witness_encoder_token_ids=witness_ids,
                            witness_ids_differ=source_ids != witness_ids)
                target = tuple(float(v) for v in bundle.project(bundle.encoder.encode([witness]))[0])
                item["target"] = list(target)
                item["target_construction_encoder_queries"] = 1
                preparation = CanonicalVerifier(bundle, target, config)
                prepared = preparation.verify_batch([TextProposal(source), TextProposal(witness)])
                prepared_by_text = {value.text: value for value in prepared}
                witness_value = prepared_by_text.get(witness)
                feasible = witness_value is not None and preparation.confirm_epsilon(witness_value)
                item.update(witness_feasible_confirmed=feasible,
                            preparation_verification=preparation.diagnostics,
                            source_distance=prepared_by_text[source].distance if source in prepared_by_text else None)
                if not feasible or source not in prepared_by_text or source_ids == witness_ids:
                    item.update(status="preparation_failure", failure="source_or_witness_failed_canonical_preparation")
                else:
                    # Fresh normal proposer. No witness text, selected spans, or
                    # construction bank is passed into the search or planner.
                    proposer = ParaphraseEditProposer()
                    planner = MeasuredCombinationPlanner(max_edits=args.max_edits)
                    result = CanonicalRefinementSearch(
                        None, edit_proposer=proposer, combination_planner=planner, config=config,
                    ).run(bundle, target, seed=args.seed, guided=False, initial_text=source)
                    item.update(status="searched", final=result.text, trace=result.trace,
                                exact_witness_recovery=result.text == witness,
                                epsilon_confirmed=bool(result.trace["epsilon_confirmed"]),
                                fresh_confirmation_count=len(result.trace["epsilon_confirmation"]["checks"]),
                                planner_last_stats=planner.last_stats)
        except Exception as exc:
            item.update(status="error", error=f"{type(exc).__name__}: {exc}")
        item["diagnostic_elapsed_seconds"] = time.monotonic() - started
        report["cases"].append(item)
        report["summary"] = {"cases": len(report["cases"]),
                             "searched": sum(row["status"] == "searched" for row in report["cases"]),
                             "selection_failures": sum(row["status"] == "selection_failure" for row in report["cases"]),
                             "other_failures": sum(row["status"] in ("preparation_failure", "error") for row in report["cases"]),
                             "epsilon_confirmed": sum(row.get("epsilon_confirmed", False) for row in report["cases"]),
                             "exact_witness_recovery": sum(row.get("exact_witness_recovery", False) for row in report["cases"])}
        save(args.output, report)
        best = item.get("trace", {}).get("best_candidate")
        print(json.dumps({"id": item["id"], "bank": item.get("bank_count"), "selected": item.get("selected_edit_count"),
                          "status": item["status"], "source_distance": item.get("source_distance"),
                          "best_distance": best["distance"] if best else None,
                          "epsilon_confirmed": item.get("epsilon_confirmed", False),
                          "exact_recovery": item.get("exact_witness_recovery", False),
                          "hard_queries": item.get("trace", {}).get("hard_queries"),
                          "seconds": item["diagnostic_elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
