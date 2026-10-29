"""Recheck canonical repeatability and the recorded small-edit response panel.

Uses the pinned encoder only; never calls a generator or changes model weights.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from point2text.encoder import SentenceEncoder
from point2text.geometry import MapBundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parents[1] / "docs/experiments/epsilon-geometry.json")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/point2text/map"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    fixture = json.loads(args.fixture.read_text())
    encoder = SentenceEncoder()
    bundle = MapBundle.build(encoder, args.cache_dir)
    if bundle.map_id != fixture["map_id"]:
        raise SystemExit("Fixture map identity differs; build a new response panel for this map.")
    texts = [r.text for r in bundle.records]
    batch = encoder.encode(texts)

    def maximum_displacement(left, right):
        return float(np.max(np.linalg.norm(bundle.project(left) - bundle.project(right), axis=1)))

    repeat = encoder.encode(texts)
    reordered = encoder.encode(texts[::-1])[::-1]
    singles = np.concatenate([encoder.encode([text]) for text in texts[:20]])
    drift = {"same_batch": maximum_displacement(batch, repeat),
             "reordered_batch": maximum_displacement(batch, reordered),
             "singleton_vs_batch": maximum_displacement(batch[:20], singles),
             "cached_vs_fresh": maximum_displacement(bundle.embeddings, batch)}
    points = bundle.project(batch)
    results = []
    probes = fixture["local_edits"] + fixture["tokenization_collisions"]
    probe_vectors = encoder.encode([p["text"] for p in probes])
    tokenize = lambda text: encoder.model.tokenizer(text, add_special_tokens=True)["input_ids"]
    for probe, point in zip(probes, bundle.project(probe_vectors), strict=True):
        index = probe["source_index"]
        results.append({"id": probe["id"], "kind": probe["kind"],
                        "map_displacement": float(np.linalg.norm(point - points[index])),
                        "same_encoder_tokens": tokenize(probe["text"]) == tokenize(texts[index])})
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "map_id": bundle.map_id, "epsilon_distance": 1e-4,
              "maximum_observed_map_drift": max(drift.values()),
              "repeatability": drift, "edit_responses": results,
              "scope": "CPU numerical probe; not a reachability or quality benchmark"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"map_id": bundle.map_id, "repeatability": drift,
                      "response_count": len(results)}, indent=2))


if __name__ == "__main__":
    main()
