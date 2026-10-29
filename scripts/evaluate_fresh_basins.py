"""Fresh off-corpus clicks and serial native basins, without witness paragraphs.

Plan targets from corpus geometry only; retain every draft and mechanical
failure. The separate review packet deliberately omits distances and seeds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import inspect
from pathlib import Path
import random
import re
import subprocess
import time

def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if str(path).endswith(".gz"):
        with gzip.open(temporary, "wt") as stream:
            stream.write(encoded)
    else:
        temporary.write_text(encoded)
    temporary.replace(path)


def merge_pools(paths):
    pools = [(path, read(path)) for path in paths]
    if not pools:
        raise ValueError("Need at least one input pool")
    first = pools[0][1]
    shared = {key: value for key, value in first["settings"].items()
              if key not in {"seed", "source_rank", "attempts"}}
    result = {"schema": "merged-fresh-native-basins-v1", "map_id": first["map_id"],
              "target_plan": first["target_plan"], "settings": dict(shared),
              "cases": [], "source_pools": [], "quality": "Individual reviews required; raw pools are retained unchanged."}
    for path, pool in pools:
        comparable = {key: value for key, value in pool["settings"].items()
                      if key not in {"seed", "source_rank", "attempts"}}
        if pool["map_id"] != result["map_id"] or comparable != shared:
            raise ValueError("Merged pools need identical map and non-seed/non-source settings")
        result["source_pools"].append({"file": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
                                       "settings": pool["settings"]})
        for case in pool["cases"]:
            target = next((row for row in result["cases"] if row["id"] == case["id"]), None)
            if target is None:
                target = {**case, "attempts": []}
                result["cases"].append(target)
            if target["target"] != case["target"]:
                raise ValueError("Merged target coordinates disagree")
            seeds = {attempt["seed"] for attempt in target["attempts"]}
            if any(attempt["seed"] in seeds for attempt in case["attempts"]):
                raise ValueError("Merged pools repeat a native seed for the same target")
            target["attempts"].extend(case["attempts"])
    result["settings"].update(seed="per-attempt", source_rank="per-source-pool", attempts="per-source-pool")
    return result


def target_plan(bundle, seed=810731, count=6):
    """Convex mixtures guarantee interior clicks without defining target text."""
    import numpy as np
    points = bundle.project(bundle.embeddings)
    topics = sorted({record.topic for record in bundle.records})
    centers = {topic: points[[r.topic == topic for r in bundle.records]].mean(axis=0)
               for topic in topics}
    if count > len(topics):
        raise ValueError("Need at least one distinct corpus topic per planned click")
    center = points.mean(axis=0)
    selected = []
    for _ in range(count):
        remaining = [topic for topic in topics if topic not in selected]
        topic = max(remaining, key=lambda name: min(
            [float(np.linalg.norm(centers[name] - centers[other])) for other in selected]
            or [float(np.linalg.norm(centers[name] - center))]))
        selected.append(topic)
    rng = np.random.default_rng(seed)
    targets = []
    for index, topic in enumerate(selected):
        # All three weights are positive: no extrapolation beyond the corpus hull.
        auxiliary = points[int(rng.integers(len(points)))]
        target = .84 * centers[topic] + .12 * center + .04 * auxiliary
        nearest = np.linalg.norm(points - target, axis=1)
        if float(nearest.min()) <= 1e-6:
            raise ValueError("Planned click accidentally equals a corpus location")
        targets.append({"id": f"fresh-click-{index + 1:02d}", "topic_anchor": topic,
                        "target": target.tolist(), "nearest_corpus_distance": float(nearest.min()),
                        "nearest_corpus_id": bundle.records[int(nearest.argmin())].id})
    return {"schema": "fresh-off-corpus-clicks-v1", "map_id": bundle.map_id,
            "seed": seed, "method": "farthest topic centroids; convex .84 topic + .12 global + .04 seeded corpus point",
            "witness_text_used": False, "targets": targets}


def packet(pool, output):
    rows, keys = [], []
    for case in pool["cases"]:
        for attempt in case["attempts"]:
            valid = attempt.get("candidates", [])
            selected = min(valid, key=lambda row: row["distance"]) if valid else None
            text = selected["text"] if selected else attempt.get("result", {}).get("text", "")
            if not text and attempt.get("drafts"):
                text = attempt["drafts"][-1]["text"]
            if not text:
                continue
            digest = sha256(text.encode()).hexdigest()
            review_id = sha256((case["id"] + str(attempt["seed"]) + digest).encode()).hexdigest()[:12]
            row = {"review_id": review_id, "text_sha256": digest, "text": text,
                   "verdict": None, "reason": None}
            if attempt.get("public_reference"):
                row["reference_text"] = attempt["public_reference"]["text"]
            rows.append(row)
            keys.append({"review_id": review_id, "target_id": case["id"], "seed": attempt["seed"],
                         "mechanically_valid": selected is not None,
                         "distance": selected["distance"] if selected else None})
    random.Random(512091).shuffle(rows)
    save(output, {"schema": "distance-blinded-fresh-text-review-v1",
                  "instructions": "Review grammar, scientific accuracy, scope/qualifications and coherence. Mark accepted/rejected/uncertain with reasons. Distances and target labels are withheld. This is individual review, not a judge-accuracy estimate.",
                  "items": rows})
    save(output.with_name(output.stem + "-key.json"), {"items": keys})


def source_packet(bundle, plan, output, ranks=(1,), cases=None):
    items, keys = [], []
    for case in plan["targets"]:
        if cases and case["id"] not in cases:
            continue
        neighbors = bundle.nearest(tuple(case["target"]), k=max(ranks))
        for rank in ranks:
            record = neighbors[rank - 1]
            review_id = sha256(record.text.encode()).hexdigest()[:12]
            if not any(item["review_id"] == review_id for item in items):
                items.append({"review_id": review_id,
                              "text_sha256": sha256(record.text.encode()).hexdigest(),
                              "public_corpus_id": record.id, "text": record.text,
                              "verdict": None, "reason": None})
            keys.append({"review_id": review_id, "target_id": case["id"], "source_rank": rank})
    random.Random(70913).shuffle(items)
    save(output, {"schema": "public-source-quality-review-v1",
                  "instructions": "Review only scientific accuracy, scope, qualifications and coherence of these public corpus references. These are not generated outputs. No target coordinates/distances are supplied.", "items": items})
    save(output.with_name(output.stem + "-key.json"), {"items": keys})


def summary(pool, paired=None, reviews=None):
    attempts = [attempt for case in pool["cases"] for attempt in case["attempts"]]
    native_passes = sum(attempt.get("result", {}).get("trace", {}).get("passes", 0) for attempt in attempts)
    guided_passes = sum(len((attempt.get("guidance") or {}).get("steps", [])) for attempt in attempts)
    updates = sum(len(step["updates"]) for attempt in attempts
                  for step in (attempt.get("guidance") or {}).get("steps", []))
    result = {"map_id": pool["map_id"], "targets": len(pool["cases"]),
              "trajectories": len(attempts), "native_passes": native_passes,
              "actual_guided_passes": guided_passes, "actual_encoder_updates": updates,
              "selection_coverage": {
                  "closest_mechanical_candidate": sum(bool(attempt["candidates"]) for attempt in attempts),
                  "raw_final_or_draft_fallback": sum(not attempt["candidates"] and bool(
                      attempt.get("result", {}).get("text") or attempt.get("drafts")) for attempt in attempts),
              },
              "cases": []}
    for case in pool["cases"]:
        values = [candidate for attempt in case["attempts"] for candidate in attempt["candidates"]]
        result["cases"].append({"id": case["id"], "topic_anchor": case["topic_anchor"],
                                "best_mechanical_distance": min((v["distance"] for v in values), default=None),
                                "quality": "Independent review required; mechanical validity alone is insufficient."})
    if paired:
        if pool["map_id"] != paired["map_id"]:
            raise ValueError("Paired arms must share the frozen map")
        comparisons = []
        indexed = {(case["id"], attempt["seed"]): (case, attempt)
                   for case in paired["cases"] for attempt in case["attempts"]}
        for case in pool["cases"]:
            for attempt in case["attempts"]:
                other_case, other = indexed[(case["id"], attempt["seed"])]
                if case["target"] != other_case["target"] or attempt["request"] != other["request"]:
                    raise ValueError("Paired arms have unequal targets/requests")
                comparisons.append({"target_id": case["id"], "seed": attempt["seed"],
                                    "identical_final_text": attempt.get("result", {}).get("text") == other.get("result", {}).get("text"),
                                    "primary_best_distance": min((v["distance"] for v in attempt["candidates"]), default=None),
                                    "paired_best_distance": min((v["distance"] for v in other["candidates"]), default=None)})
        result["pairing"] = {"identical_targets_and_requests": True, "comparisons": comparisons}
    if reviews:
        anchors = approved_anchors(pool, reviews)
        result["independent_quality"] = {
            "approved_target_count": sum(anchor["candidate"] is not None for anchor in anchors),
            "target_count": len(anchors), "best_reviewed_generated_anchors": anchors,
            "note": "Individual distance-blinded judgments; no statistical quality guarantee.",
        }
    return result


def approved_anchors(pool, reviews):
    accepted = {item["review_id"]: item for item in reviews["items"] if item["verdict"] == "accepted"}
    anchors = []
    for case in pool["cases"]:
        choices = []
        for attempt in case["attempts"]:
            if not attempt["candidates"]:
                continue
            candidate = min(attempt["candidates"], key=lambda row: row["distance"])
            digest = sha256(candidate["text"].encode()).hexdigest()
            review_id = sha256((case["id"] + str(attempt["seed"]) + digest).encode()).hexdigest()[:12]
            if review_id in accepted:
                if accepted[review_id]["text_sha256"] != digest:
                    raise ValueError("Review text hash does not match its generated candidate")
                choices.append((candidate, attempt, review_id))
        if choices:
            candidate, attempt, review_id = min(choices, key=lambda item: item[0]["distance"])
            anchors.append({"id": case["id"], "target": case["target"], "candidate": candidate,
                            "request": attempt["request"], "review_id": review_id,
                            "public_reference_text": attempt.get("public_reference", {}).get("text")})
        else:
            anchors.append({"id": case["id"], "target": case["target"], "candidate": None,
                            "reason": "no_independently_approved_generated_anchor"})
    return anchors


def refinement_packet(report, output, anchor_pool=None):
    rows, keys = [], []
    references = {}
    if anchor_pool:
        for case in anchor_pool["cases"]:
            for attempt in case["attempts"]:
                for candidate in attempt["candidates"]:
                    references[(case["id"], candidate["text"])] = attempt.get("public_reference", {}).get("text")
    for item in report["runs"]:
        if not item.get("result", {}).get("text"):
            continue
        text = item["result"]["text"]
        digest = sha256(text.encode()).hexdigest()
        review_id = sha256((item["target_id"] + item["arm"] + digest).encode()).hexdigest()[:12]
        anchor_text = item["anchor"]["candidate"]["text"]
        public_text = item["anchor"].get("public_reference_text") or references.get((item["target_id"], anchor_text))
        if not public_text:
            raise ValueError("Final review requires its original public reference; supply --anchor-pool")
        rows.append({"review_id": review_id, "text_sha256": digest,
                     "public_reference_text": public_text, "initial_anchor_text": anchor_text,
                     "text": text, "verdict": None, "reason": None,
                     "anchor_claim_preservation": None, "anchor_claim_reason": None})
        keys.append({"review_id": review_id, "target_id": item["target_id"], "arm": item["arm"],
                     "requires_anchor_preservation": item["arm"] in ("words", "sentence"),
                     "best_candidate": item["result"]["trace"].get("best_candidate")})
    random.Random(820179).shuffle(rows)
    save(output, {"schema": "distance-blinded-refinement-quality-v1",
                  "policy_declared_before_final_review": True,
                  "instructions": "For ALL items, mark verdict accepted/rejected/uncertain for grammar, scientific accuracy, coherence and fidelity to the PUBLIC reference. Separately mark anchor_claim_preservation preserved/drift/uncertain and explain any change versus the INITIAL generated anchor. Arms, targets and distances are hidden. The predeclared key additionally requires anchor preservation for local-repair arms; fresh-restart outputs may differ from the old anchor while remaining faithful to the public source.", "items": rows})
    save(output.with_name(output.stem + "-key.json"), {"items": keys})


def refinement_summary(report, reviews=None):
    verdicts = {item["review_id"]: item for item in (reviews or {}).get("items", [])}
    rows = []
    for run in report["runs"]:
        trace = run.get("result", {}).get("trace", {})
        best = trace.get("best_candidate")
        text = run.get("result", {}).get("text", "")
        digest = sha256(text.encode()).hexdigest()
        review_id = sha256((run["target_id"] + run["arm"] + digest).encode()).hexdigest()[:12]
        review = verdicts.get(review_id)
        if review and review["text_sha256"] != digest:
            raise ValueError("Post-review hash does not match final text")
        initial = run["anchor"].get("candidate")
        row = {"target_id": run["target_id"], "arm": run["arm"],
               "status": run.get("status", trace.get("status", "error")),
               "reason": run.get("reason", trace.get("stop_reason", run.get("error"))),
               "initial_reviewed_distance": initial["distance"] if initial else None,
               "final_mechanical_distance": best["distance"] if best else None,
               "post_review": review["verdict"] if review else "not_reviewed",
               "post_review_reason": review.get("reason") if review else None,
               "epsilon_confirmed": trace.get("epsilon_confirmed", False),
               "confirmation": trace.get("epsilon_confirmation"),
               "elapsed_seconds": trace.get("elapsed_seconds"),
               "hard_queries": trace.get("hard_queries"),
               "restart_count": trace.get("restart_count"),
               "executed_work": trace.get("executed_work")}
        accepted = bool(review and review["verdict"] == "accepted" and (
            run["arm"] == "sentence-restarts" or review.get("anchor_claim_preservation") == "preserved"))
        row["passes_predeclared_quality_policy"] = accepted
        row["best_reviewed_initial_or_final_distance"] = min(
            [initial["distance"]] + ([best["distance"]] if accepted and best else [])
        ) if initial else None
        rows.append(row)
    return {"schema": "fresh-refinement-summary-v1", "map_id": report["map_id"],
            "targets": len(report["anchors"]),
            "targets_with_reviewed_generated_anchor": sum(anchor["candidate"] is not None for anchor in report["anchors"]),
            "rows": rows, "note": "Unreviewed/rejected final improvements do not establish quality-preserving progress. Missing targets remain explicit."}


def refine(args, bundle):
    from dataclasses import replace
    from point2text.diffusion.combinations import MeasuredCombinationPlanner
    from point2text.diffusion.contracts import DiffusionRequest
    from point2text.diffusion.encoder import TorchEmbeddingReward
    from point2text.diffusion.guidance import GuidanceConfig
    from point2text.diffusion.paraphrases import ParaphraseEditProposer
    from point2text.diffusion.refinement import CanonicalRefinementSearch
    from point2text.diffusion.refinement_contracts import RefinementConfig
    from point2text.diffusion.runtime import MLXDiffusionRuntime
    from point2text.profile import PromptPolicy

    pool, reviews = read(args.pool), read(args.reviews)
    if pool["map_id"] != bundle.map_id:
        raise ValueError("Anchor pool belongs to a different frozen map")
    fields = RefinementConfig.__dataclass_fields__
    if not {"repair_span_mode", "max_restarts"}.issubset(fields):
        raise RuntimeError("This phase requires the restart/sentence refinement core")
    anchors = approved_anchors(pool, reviews)
    policy = PromptPolicy(**pool["settings"]["prompt_policy"])
    arms = args.arm or ["words", "sentence"]
    report = {"schema": "fresh-approved-anchor-refinement-v1", "map_id": bundle.map_id,
              "anchors": anchors, "arms": arms, "runs": [],
              "quality": "Anchors independently approved; new outputs require the separate blinded post-review."}
    implementation = Path(inspect.getfile(CanonicalRefinementSearch)).resolve()
    source_root = implementation.parents[3]
    commit = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    report["provenance"] = {
        "source_commit": commit,
        "refinement_sha256": sha256(implementation.read_bytes()).hexdigest(),
        "runtime_sha256": sha256(Path(inspect.getfile(MLXDiffusionRuntime)).read_bytes()).hexdigest(),
        "pool_sha256": sha256(args.pool.read_bytes()).hexdigest(),
        "reviews_sha256": sha256(args.reviews.read_bytes()).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.resume and args.output.exists():
        report = read(args.output)
        if report["map_id"] != bundle.map_id or report["arms"] != arms:
            raise ValueError("Resume requires identical map and arms")
    session_end = time.monotonic() + args.session_seconds
    runtime = MLXDiffusionRuntime()
    reward = TorchEmbeddingReward(bundle.encoder, bundle)
    guidance = GuidanceConfig(inner_updates=args.inner_updates, strength=args.strength,
                              max_delta=args.max_delta, guided_steps=args.guided_steps,
                              start_fraction=args.start_fraction)
    for index, anchor in enumerate(anchors):
        if anchor["candidate"] is None:
            for arm in arms:
                if not any(row["target_id"] == anchor["id"] and row["arm"] == arm for row in report["runs"]):
                    report["runs"].append({"target_id": anchor["id"], "arm": arm,
                                           "anchor": anchor, "status": "not_run",
                                           "reason": "no_independently_approved_generated_anchor"})
            save(args.output, report)
            continue
        for arm in arms:
            if any(item["target_id"] == anchor["id"] and item["arm"] == arm for item in report["runs"]):
                continue
            remaining = session_end - time.monotonic()
            if remaining <= 0:
                report["stop_reason"] = "session_budget"
                save(args.output, report)
                return
            deadline = Deadline(time.monotonic() + min(args.seconds, remaining))
            config = RefinementConfig(
                max_seconds=min(args.seconds, remaining), max_hard_queries=args.queries,
                max_rounds=args.rounds, beam_width=4, proposals_per_round=128,
                min_words=policy.min_words, max_words=policy.max_words,
                repair_every=1, stall_rounds=3,
                repair_span_mode="words" if arm == "words" else "sentence",
                max_restarts=args.restarts if arm == "sentence-restarts" else 0,
                restart_candidates=2,
            )
            proposer = ParaphraseEditProposer(bundle, should_stop=deadline.is_set)
            item = {"target_id": anchor["id"], "arm": arm, "anchor": anchor,
                    "config": asdict(config), "guidance": asdict(guidance), "drafts": []}
            report["runs"].append(item)
            request = replace(DiffusionRequest(**anchor["request"]), seed=args.seed + index * 10000)
            try:
                result = CanonicalRefinementSearch(
                    runtime, reward=reward, guidance_config=guidance, edit_proposer=proposer,
                    config=config, combination_planner=MeasuredCombinationPlanner(max_edits=4),
                    prompt_policy=policy,
                ).run(bundle, tuple(anchor["target"]), request=request,
                      initial_text=anchor["candidate"]["text"], max_candidates=2,
                      guided=not args.unguided,
                      on_draft=lambda draft: item["drafts"].append(draft))
                item["result"] = asdict(result)
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
            item["proposal_bank"] = proposer.diagnostics
            save(args.output, report)
            if args.packet:
                refinement_packet(report, args.packet)
            best = item.get("result", {}).get("trace", {}).get("best_candidate")
            print(json.dumps({"target": anchor["id"], "arm": arm,
                              "initial": anchor["candidate"]["distance"],
                              "best": best["distance"] if best else None, "error": item.get("error")}), flush=True)


class Deadline:
    def __init__(self, end):
        self.end = end

    def is_set(self):
        return time.monotonic() >= self.end


def generate(args, bundle):
    from point2text.diffusion.contracts import DiffusionRequest
    from point2text.diffusion.encoder import TorchEmbeddingReward
    from point2text.diffusion.guidance import GuidanceConfig, SparseMapGuidance
    from point2text.diffusion.refinement_contracts import RefinementConfig, TextProposal
    from point2text.diffusion.runtime import MLXDiffusionRuntime
    from point2text.diffusion.search import technical_prompt
    from point2text.diffusion.verification import CanonicalVerifier
    from point2text.profile import PromptPolicy

    plan = read(args.targets)
    if plan["map_id"] != bundle.map_id:
        raise ValueError("Target plan belongs to a different frozen map")
    policy = PromptPolicy() if not args.grounded else PromptPolicy(
        purpose="Paraphrase the provided public reference without changing its scientific claims",
        style="precise, fluent and conservative technical prose", neighbors=1,
        min_words=args.min_words, max_words=args.max_words,
        extra_instructions=(
            "Treat the single reference as the entire factual scope. Preserve every qualification, "
            "limitation and condition. Preserve distinctions among observation, association, estimation "
            "and causal evidence. Do not add examples, mechanisms, benefits, guarantees or measurements "
            "absent from the reference. Rephrase syntax and wording without copying any full reference "
            "sentence. Return a coherent, complete paragraph."))
    if args.conservative_grounded:
        if not args.grounded:
            raise ValueError("--conservative-grounded requires --grounded")
        from dataclasses import replace
        policy = replace(policy, extra_instructions=(
            "Write ONLY 2–3 coherent sentences paraphrasing the single reference's existing claims. "
            "Do not fill space with a concluding implication or recommendation. Preserve every 'may', "
            "'can' and 'helps' qualification and its scope. Never turn 'helps' or 'can' into 'prevents' "
            "or 'guarantees'. Do not use 'must', 'requires', 'prevents', 'guarantees' or 'ensures' unless "
            "the reference itself states that exact strength for that exact claim. Do not add "
            "requirements, inference conditions, purposes, causal mechanisms or conclusions. "
            "Change wording and sentence structure without copying any complete source sentence; "
            "retain the same modest factual scope. No filler."))
    settings = {"attempts": args.attempts, "seed": args.seed, "inner_updates": args.inner_updates,
                "max_tokens": args.tokens, "steps": args.steps, "prompt_policy": asdict(policy),
                "guided": not args.unguided, "grounded": args.grounded,
                "min_words": args.min_words, "max_words": args.max_words,
                "conservative_grounded": args.conservative_grounded,
                "source_rank": args.source_rank,
                "strength": args.strength, "max_delta": args.max_delta,
                "guided_steps": args.guided_steps, "start_fraction": args.start_fraction}
    report = {"schema": "fresh-native-basins-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "map_id": bundle.map_id, "target_plan": plan, "settings": settings,
              "quality": "Mechanical checks only; independent blinded review is separate.", "cases": []}
    if args.resume and args.output.exists():
        report = read(args.output)
        if report["map_id"] != bundle.map_id or report["settings"] != settings:
            raise ValueError("Resume requires identical map and experimental settings")
    started = time.monotonic()
    deadline = Deadline(started + args.seconds)
    runtime = MLXDiffusionRuntime()
    reward = TorchEmbeddingReward(bundle.encoder, bundle)
    guidance = GuidanceConfig(inner_updates=args.inner_updates, strength=args.strength,
                              max_delta=args.max_delta, guided_steps=args.guided_steps,
                              start_fraction=args.start_fraction)
    report["runtime"] = runtime.identity
    report["guidance"] = asdict(guidance)
    versions = {}
    for package in ("numpy", "torch", "sentence-transformers", "mlx", "mlx-vlm"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    report["provenance"] = {"script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                            "versions": versions, "witness_text_used": False}
    for case_index, target_info in enumerate(plan["targets"]):
        if args.case and target_info["id"] not in args.case:
            continue
        case = next((row for row in report["cases"] if row["id"] == target_info["id"]), None)
        if case is None:
            case = {**target_info, "attempts": []}
            report["cases"].append(case)
        target = tuple(case["target"])
        for attempt_index in range(args.attempts):
            seed = args.seed + case_index * 10000 + attempt_index * 101
            if any(row["seed"] == seed for row in case["attempts"]):
                continue
            if deadline.is_set():
                report["stop_reason"] = "global_time_budget"
                save(args.output, report)
                return
            item = {"seed": seed, "drafts": [], "candidates": []}
            case["attempts"].append(item)
            verifier = CanonicalVerifier(bundle, target, RefinementConfig(
                max_seconds=max(.1, deadline.end - time.monotonic()), max_hard_queries=1024,
                min_words=args.min_words, max_words=args.max_words))

            def verify(text, origin):
                if args.grounded:
                    reference = examples[0].text
                    sentences = re.split(r"(?<=[.!?])\s+", reference.strip())
                    copied = [sentence for sentence in sentences if sentence.casefold() in text.casefold()]
                    if copied:
                        return {"origin": origin, "text": text, "verified": [],
                                "source_copy_rejection": copied,
                                "verification": verifier.diagnostics}
                verified = verifier.verify_batch([TextProposal(text)])
                item["candidates"].extend(asdict(value) for value in verified)
                return {"origin": origin, "text": text,
                        "verified": [asdict(value) for value in verified],
                        "verification": verifier.diagnostics}

            def observe(draft):
                item["drafts"].append({**asdict(draft), "canonical": verify(draft.text, "draft")})
                return deadline.is_set()

            examples = bundle.nearest(target, k=max(policy.neighbors, args.source_rank))
            if args.grounded:
                examples = [examples[args.source_rank - 1]]
                item["public_reference"] = asdict(examples[0])
            request = DiffusionRequest(technical_prompt(examples, policy),
                                       seed=seed, max_tokens=args.tokens, steps=args.steps)
            item["request"] = asdict(request)
            hook = None if args.unguided else SparseMapGuidance(reward, target, guidance)
            before = time.monotonic()
            try:
                result = runtime.generate(request, hook=hook, cancel_event=deadline,
                                          observer=observe, observe_every=3)
                item["result"] = asdict(result)
                item["final_canonical"] = verify(result.text, "final")
                if args.grounded:
                    reference = examples[0].text
                    sentences = re.split(r"(?<=[.!?])\s+", reference.strip())
                    item["source_copy_audit"] = {
                        "exact_reference": " ".join(reference.split()).casefold() == " ".join(result.text.split()).casefold(),
                        "complete_reference_sentences_copied": [sentence for sentence in sentences
                            if sentence.casefold() in result.text.casefold()],
                    }
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
            item["guidance"] = hook.trace if hook else None
            item["elapsed_seconds"] = time.monotonic() - before
            item["verification"] = verifier.diagnostics
            report["elapsed_seconds_this_session"] = time.monotonic() - started
            save(args.output, report)
            if args.packet:
                packet(report, args.packet)
            best = min((value["distance"] for value in item["candidates"]), default=None)
            print(json.dumps({"target": case["id"], "seed": seed, "valid_drafts": len(item["candidates"]),
                              "best_distance": best, "seconds": item["elapsed_seconds"],
                              "error": item.get("error")}), flush=True)
    report["stop_reason"] = "requested_basins_complete"
    save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "generate", "packet", "sources", "summary", "refine", "refinement-summary", "refinement-packet", "merge"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--input", type=Path, action="append")
    parser.add_argument("--paired", type=Path)
    parser.add_argument("--anchor-pool", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument("--source-rank", type=int, default=1)
    parser.add_argument("--source-ranks", default="1", help="Comma-separated public-neighbor ranks for source-only review.")
    parser.add_argument("--arm", choices=("words", "sentence", "sentence-restarts"), action="append")
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=False)
    parser.add_argument("--plan-seed", type=int, default=810731)
    parser.add_argument("--seed", type=int, default=1973431)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--inner-updates", type=int, default=16)
    parser.add_argument("--unguided", action="store_true", help="Explicit same-seed native ablation; no logit guidance.")
    parser.add_argument("--grounded", action="store_true", help="Exploratory one-reference claim-preserving prompt; separate from the paired ablation.")
    parser.add_argument("--conservative-grounded", action="store_true",
                        help="Separate exploratory 2–3sentence source-modality-preserving prompt.")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=1.0)
    parser.add_argument("--guided-steps", type=int, default=8)
    parser.add_argument("--start-fraction", type=float, default=.2)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--min-words", type=int, default=60,
                        help="Explicit experimental word policy; grounded40is separate from product60minimum.")
    parser.add_argument("--max-words", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--session-seconds", type=float, default=1200)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "merge":
        if not args.input:
            parser.error("merge requires --input pools")
        save(args.output, merge_pools(args.input))
        return
    if args.phase == "packet":
        if not args.pool:
            parser.error("packet requires --pool")
        packet(read(args.pool), args.output)
        return
    if args.phase == "summary":
        if not args.pool:
            parser.error("summary requires --pool")
        save(args.output, summary(read(args.pool), read(args.paired) if args.paired else None,
                                  read(args.reviews) if args.reviews else None))
        return
    if args.phase == "refinement-summary":
        if not args.pool:
            parser.error("refinement-summary requires --pool")
        save(args.output, refinement_summary(read(args.pool), read(args.reviews) if args.reviews else None))
        return
    if args.phase == "refinement-packet":
        if not args.pool:
            parser.error("refinement-packet requires --pool")
        refinement_packet(read(args.pool), args.output, read(args.anchor_pool) if args.anchor_pool else None)
        return
    if args.phase in ("generate", "sources") and not args.targets:
        parser.error("generate/sources require --targets")
    if args.phase == "refine" and (not args.pool or not args.reviews):
        parser.error("refine requires --pool and --reviews")
    if not args.cache_dir:
        parser.error("plan/generate require --cache-dir pointing at the frozen map cache")
    if not 1 <= args.attempts <= 64 or not 1 <= args.tokens <= 254 or args.seconds <= 0:
        parser.error("invalid attempt, token or time budget")
    from point2text.encoder import SentenceEncoder
    from point2text.geometry import MapBundle
    bundle = MapBundle.build(SentenceEncoder(), args.cache_dir / "map")
    if args.phase == "plan":
        save(args.output, target_plan(bundle, args.plan_seed))
        print(args.output)
    elif args.phase == "sources":
        ranks = tuple(int(value) for value in args.source_ranks.split(","))
        if not ranks or min(ranks) < 1 or max(ranks) > 64:
            parser.error("source ranks must be in1..64")
        source_packet(bundle, read(args.targets), args.output, ranks, args.case)
    elif args.phase == "refine":
        refine(args, bundle)
    else:
        generate(args, bundle)


if __name__ == "__main__":
    main()
