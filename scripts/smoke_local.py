#!/usr/bin/env python3
"""Optional real-model smoke check against a running local point2text app.

Unlike the default tests, this requests real generation. It uses only stdlib.
"""

import argparse
import json
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--candidates", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    def api(path, data=None):
        request = Request(args.url.rstrip("/") + path,
                          data=json.dumps(data).encode() if data is not None else None,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.load(response)

    deadline = time.monotonic() + args.timeout
    while not api("/api/status")["ready"]:
        if time.monotonic() >= deadline:
            raise SystemExit("Map did not become ready before the timeout.")
        time.sleep(1)
    bundle = api("/api/map")
    witness = bundle["points"][0]
    run_id = api("/api/runs", {"map_id": bundle["map_id"],
                             "target": [witness["x"], witness["y"]],
                             "candidates": args.candidates, "seed": 42})["id"]
    while True:
        run = api(f"/api/runs/{run_id}")
        if run["status"] != "running":
            break
        if time.monotonic() >= deadline:
            with urlopen(Request(args.url.rstrip("/") + f"/api/runs/{run_id}",
                                 method="DELETE"), timeout=10):
                pass
            raise SystemExit("Generation timed out; cancellation requested.")
        time.sleep(0.5)
    if run["status"] != "complete" or not run["best"]:
        raise SystemExit(run.get("error") or f"Run ended: {run['status']}")
    print(json.dumps({"status": "passed", "run_id": run_id,
                      "map_id": bundle["map_id"], "target_record": witness["id"],
                      "verified_candidates": len(run["candidates"]),
                      "best_distance": run["best"]["distance"],
                      "elapsed_seconds": run["elapsed_seconds"],
                      "generator": run["generator"]}, indent=2))


if __name__ == "__main__":
    main()
