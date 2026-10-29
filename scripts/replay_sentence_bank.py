"""Replay archived unreviewed sentence variants without calling any generator.

Both arms use the same current controller, fixed supplied banks, initial texts,
map and budgets. Only the screen function is swapped for the legacy arm. The
legacy function is loaded from its exact Git blob into an isolated namespace;
its original globals are retained. New source paragraphs absent from the
archive have no proposals. Every resulting full text still needs blind review.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import inspect
from pathlib import Path
import subprocess
import sys
import time
import types

from evaluate_fresh_basins import (
    Deadline, frozen_bank_export, read, reference_copy_audit,
    refinement_packet, save,
)

SCHEMA = "point2text-sentence-bank-v1"
LEGACY_COMMIT = "a00c2d6454451910d0bad0004d5c01cb4c60417f"


def prepare(report, path):
    targets = []
    total = 0
    for run in report["runs"]:
        originals = {source["source_sha256"]: source for source in run["actual_bank_sources"]}
        grouped = {}
        for receipt_key, receipt in run["provider_diagnostics"]["sources"].items():
            key = receipt["source_sha256"]
            source = originals[key]
            if key not in grouped:
                grouped[key] = {"source_sha256": key, "text": source["text"], "sentences": []}
            start, end = receipt["span"]
            original = source["text"][start:end]
            if sha256(original.encode()).hexdigest() != receipt["source_sentence_sha256"]:
                raise ValueError("Archived source-sentence hash mismatch")
            raw = receipt["raw_variants"]
            if len(raw) != receipt["raw_variants_total"]:
                raise ValueError("This replay requires every raw variant, not a truncated subset")
            variants = [{"text": value, "review": {
                "status": "unreviewed", "canonical_verified": False, "semantic_verified": False,
                "origin": "archived-raw-model-response", "receipt_key": receipt_key,
                "raw_variant_index": i, "original_guard_check": receipt["variant_checks"][i],
            }} for i, value in enumerate(raw)]
            grouped[key]["sentences"].append({"span": [start, end], "original": original, "variants": variants})
            total += len(raw)
        targets.append({"target_id": run["target_id"], "anchor": run["anchor"],
                        "config": run["config"], "request": run["result"]["trace"]["request"],
                        "bank": {"schema": SCHEMA, "sources": list(grouped.values())},
                        "original_model_digest": run["provider_diagnostics"]["model_digest"]})
    return {"schema": "saved-raw-sentence-variants-v1", "map_id": report["map_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_receipt": {"file": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
                               "provenance": report["provenance"]},
            "raw_variant_count": total, "targets": targets,
            "policy": "Saved raw proposals are UNREVIEWED, including those previously accepted by a literal screen. No new model requests. Missing-source banks yield no proposals."}


def legacy_module(repo, commit):
    raw = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:src/point2text/diffusion/sentence_bank.py"],
                         check=True, capture_output=True).stdout
    name = "point2text.diffusion._archived_sentence_bank_replay"
    module = types.ModuleType(name)
    module.__package__ = "point2text.diffusion"
    module.__file__ = f"git:{commit}:sentence_bank.py"
    sys.modules[name] = module
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module, sha256(raw).hexdigest()


def run(args, supplied):
    from point2text.diffusion import sentence_bank as bank_module
    from point2text.diffusion.combinations import MeasuredCombinationPlanner
    from point2text.diffusion.contracts import DiffusionRequest
    from point2text.diffusion.refinement import CanonicalRefinementSearch
    from point2text.diffusion.refinement_contracts import RefinementConfig
    from point2text.encoder import SentenceEncoder
    from point2text.geometry import MapBundle
    from point2text.profile import PromptPolicy

    class RecordingPlanner(MeasuredCombinationPlanner):
        def __init__(self):
            super().__init__(max_edits=4)
            self.archive = {}

        def record(self, source, proposals, verified):
            result = super().record(source, proposals, verified)
            for text, state in self._sources.items():
                self.archive[sha256(text.encode()).hexdigest()] = {
                    "source_text": text, "point": list(state.point),
                    "atoms": [asdict(atom) for atom in state.atoms.values()],
                    "coordinates": [{"text": candidate, "point": list(point)}
                                    for candidate, point in state.coordinates.items()]}
            return result

    implementation = Path(inspect.getfile(CanonicalRefinementSearch)).resolve()
    repo = implementation.parents[3]
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            check=True, text=True, capture_output=True).stdout.strip()
    legacy, legacy_hash = legacy_module(repo, args.legacy_commit)
    corrected_screen = bank_module.sentence_screen_reason
    if sha256(Path(bank_module.__file__).read_bytes()).hexdigest() == legacy_hash:
        raise RuntimeError("The corrected guard has not arrived; replay must wait")
    report = {"schema": "fresh-approved-anchor-refinement-v1", "map_id": supplied["map_id"],
              "anchors": [target["anchor"] for target in supplied["targets"]],
              "arms": ["replay-legacy", "replay-corrected"], "runs": [],
              "quality": "All supplied variants remain unreviewed. Full-text independent review is mandatory.",
              "provenance": {"source_commit": commit,
                  "refinement_sha256": sha256(implementation.read_bytes()).hexdigest(),
                  "sentence_bank_sha256": sha256(Path(bank_module.__file__).read_bytes()).hexdigest(),
                  "legacy_sentence_bank_sha256": legacy_hash, "legacy_commit": args.legacy_commit,
                  "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                  "bank_sha256": sha256(args.bank.read_bytes()).hexdigest(),
                  "original_receipt": supplied["source_receipt"],
                  "method": "Paired fixed-bank replay; swap screening function only. Zero model calls."}}
    guard_audit = []
    for target in supplied["targets"]:
        for source in target["bank"]["sources"]:
            for sentence in source["sentences"]:
                for index, variant in enumerate(sentence["variants"]):
                    old_reason = legacy.sentence_screen_reason(sentence["original"], variant["text"])
                    if old_reason != variant["review"]["original_guard_check"]["reason"]:
                        raise ValueError("Legacy screening does not reproduce the archived decision")
                    guard_audit.append({"target_id": target["target_id"],
                        "source_sha256": source["source_sha256"], "span": sentence["span"], "index": index,
                        "legacy_reason": old_reason,
                        "corrected_reason": corrected_screen(sentence["original"], variant["text"])})
    if not any(row["legacy_reason"] != row["corrected_reason"] for row in guard_audit):
        raise RuntimeError("No screening change on the archived pool; wait for the intended guard fix")
    if len(guard_audit) != supplied["raw_variant_count"]:
        raise ValueError("Fixed-bank variant count changed")
    report["fixed_raw_variant_guard_audit"] = guard_audit
    bundle = MapBundle.build(SentenceEncoder(), args.cache_dir / "map")
    if bundle.map_id != supplied["map_id"]:
        raise ValueError("Replay map differs from the original map")
    try:
        for target in supplied["targets"]:
            for arm, screen in (("replay-legacy", legacy.sentence_screen_reason),
                                ("replay-corrected", corrected_screen)):
                bank_module.sentence_screen_reason = screen
                config = RefinementConfig(**target["config"])
                deadline = Deadline(time.monotonic() + config.max_seconds)
                bank = bank_module.SentenceParaphraseBank(
                    reviewed_bank=target["bank"], max_variants_per_sentence=6,
                    should_stop=deadline.is_set)
                planner = RecordingPlanner()
                anchor = target["anchor"]
                item = {"target_id": target["target_id"], "arm": arm, "anchor": anchor,
                        "config": asdict(config), "drafts": [], "native_backend_enabled": False,
                        "model_calls": 0, "method_note": "Archived unreviewed whole-sentence variants; canonical-only replay."}
                report["runs"].append(item)
                try:
                    result = CanonicalRefinementSearch(
                        None, edit_proposer=bank, config=config, combination_planner=planner,
                        prompt_policy=PromptPolicy(min_words=config.min_words, max_words=config.max_words),
                    ).run(bundle, tuple(anchor["target"]),
                          request=DiffusionRequest(**target["request"]),
                          initial_text=anchor["candidate"]["text"], max_candidates=2, guided=False)
                    item["result"] = asdict(result)
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                item["proposal_bank"] = bank.diagnostics
                item["actual_bank_sources"] = frozen_bank_export(bank)
                item["actual_bank_atom_count"] = sum(len(source["atoms"]) for source in item["actual_bank_sources"])
                item["planner_measurements"] = planner.archive
                available = {source["source_sha256"] for source in target["bank"]["sources"]}
                seen = {source["source_sha256"] for source in item["actual_bank_sources"]}
                item["saved_bank_coverage"] = {"available_source_count": len(available),
                    "visited_saved_source_count": len(available & seen),
                    "visited_unavailable_source_hashes": sorted(seen - available),
                    "unvisited_saved_source_hashes": sorted(available - seen)}
                item["public_reference_copy_audit"] = reference_copy_audit(
                    item.get("result", {}).get("text", ""), anchor["public_reference_text"])
                save(args.output, report)
                if args.packet:
                    refinement_packet(report, args.packet)
                best = item.get("result", {}).get("trace", {}).get("best_candidate")
                print({"target": target["target_id"], "arm": arm,
                       "atoms": item["actual_bank_atom_count"],
                       "best": best["distance"] if best else None, "error": item.get("error")}, flush=True)
    finally:
        bank_module.sentence_screen_reason = corrected_screen
    report["stop_reason"] = "requested_fixed_bank_replays_complete"
    save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--legacy-commit", default=LEGACY_COMMIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.receipt:
            parser.error("prepare requires --receipt")
        save(args.output, prepare(read(args.receipt), args.receipt))
    else:
        if not args.bank or not args.cache_dir:
            parser.error("run requires --bank and --cache-dir")
        run(args, read(args.bank))


if __name__ == "__main__":
    main()
