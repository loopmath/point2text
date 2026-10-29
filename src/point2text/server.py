"""Loopback web API; scientific components do not depend on the server."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import ipaddress
import math
import os
from pathlib import Path
import threading
import time
import subprocess
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .contracts import Candidate
from .storage import ParagraphLibrary


class RunRequest(BaseModel):
    map_id: str = Field(min_length=1, max_length=200)
    target: tuple[float, float]
    candidates: int = Field(default=6, ge=1, le=64)
    seed: int = Field(default=42, ge=0, le=2**31 - 7)
    guidance_updates: int | None = Field(default=None, ge=1, le=64)
    refinement_epsilon: float | None = Field(default=None, gt=0, le=1)
    refinement_seconds: float | None = Field(default=None, ge=5, le=600)
    refinement_unlimited: bool = False

    @field_validator("target")
    @classmethod
    def finite_target(cls, value):
        if not all(math.isfinite(x) for x in value):
            raise ValueError("Target coordinates must be finite.")
        return value


class SaveRequest(BaseModel):
    model_config = {"extra": "forbid"}
    map_id: str = Field(min_length=1, max_length=200)
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    candidate_index: int = Field(ge=0)
    title: str | None = Field(default=None, min_length=1, max_length=160)


@dataclass
class Run:
    id: str
    map_id: str
    target: tuple[float, float]
    seed: int
    requested_candidates: int
    generator: str
    encoder: str
    generator_metadata: dict
    strategy: str = "nearest-context-independent-candidates-v1"
    trace: dict | None = None
    guidance_updates: int | None = None
    refinement_epsilon: float | None = None
    refinement_seconds: float | None = None
    refinement_unlimited: bool = False
    draft: dict | None = None
    draft_sequence: int = 0
    status: str = "running"
    error: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: float = field(default_factory=time.monotonic)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    elapsed_seconds: float | None = None

    def snapshot(self):
        with self.lock:
            candidates = [c.payload() for c in self.candidates]
            return {
                "id": self.id, "map_id": self.map_id, "target": list(self.target),
                "seed": self.seed, "requested_candidates": self.requested_candidates,
                "status": self.status, "error": self.error,
                "candidates": candidates,
                "best": min(candidates, key=lambda c: c["distance"], default=None),
                "elapsed_seconds": self.elapsed_seconds if self.elapsed_seconds is not None
                else time.monotonic() - self.started,
                "created_at": self.created_at,
                "generator": self.generator, "encoder": self.encoder,
                "generator_metadata": self.generator_metadata,
                "strategy": self.strategy, "trace": self.trace,
                "guidance_updates": self.guidance_updates,
                "refinement_epsilon": self.refinement_epsilon,
                "refinement_seconds": self.refinement_seconds,
                "refinement_unlimited": self.refinement_unlimited,
                "draft": self.draft, "draft_sequence": self.draft_sequence,
                "distance_metric": "euclidean-unwhitened-pca",
            }


def create_app(*, bundle: Any = None, generator: Any = None,
               cache_dir: Path | None = None, search_factory: Any = None) -> FastAPI:
    """Inject a map and generator to test or embed the app without model loading."""
    cache_dir = Path(cache_dir or os.environ.get("POINT2TEXT_CACHE", ".cache/point2text"))
    state_lock = threading.Lock()
    state = {"bundle": bundle, "error": None, "active": None, "closed": False}
    runs: dict[str, Run] = {}
    library = ParagraphLibrary(cache_dir / "library.sqlite3")
    if generator is None:
        from .generation import OllamaGenerator
        generator = OllamaGenerator(
            base_url=os.environ.get("POINT2TEXT_OLLAMA_URL", "http://127.0.0.1:11434"),
            model=os.environ.get("POINT2TEXT_MODEL", "gemma4:12b"),
        )

    def initialize():
        try:
            from .encoder import SentenceEncoder
            from .geometry import MapBundle
            result = MapBundle.build(SentenceEncoder(), cache_dir / "map")
            with state_lock:
                state["bundle"] = result
        except Exception as exc:
            with state_lock:
                state["error"] = f"Could not initialize the local map: {exc}"

    @asynccontextmanager
    async def lifespan(app):
        if state["bundle"] is None:
            threading.Thread(target=initialize, name="point2text-map", daemon=True).start()
        yield
        with state_lock:
            state["closed"] = True
            for run in runs.values():
                run.cancel_event.set()

    app = FastAPI(title="point2text", version="0.1.0", lifespan=lifespan)
    static = Path(__file__).parent / "static"
    if static.exists():
        app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/status")
    def status():
        return {"ready": state["bundle"] is not None, "error": state["error"],
                "generator": generator.available()}

    def get_bundle():
        if state["error"]:
            raise HTTPException(503, state["error"])
        if state["bundle"] is None:
            raise HTTPException(503, "The local encoder and map are loading. Try again shortly.")
        return state["bundle"]

    @app.get("/api/map")
    def map_data():
        return get_bundle().payload()

    def worker(run: Run, map_bundle):
        try:
            from .search import BaselineSearch

            def on_candidate(candidate):
                with run.lock:
                    run.candidates.append(candidate)

            def on_draft(update):
                if not isinstance(update, dict) or not isinstance(update.get("text"), str):
                    return
                def count(value):
                    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
                draft = {
                    "text": update["text"][:24000],
                    "step": count(update.get("step")), "num_steps": count(update.get("num_steps")),
                    "phase": str(update.get("phase", "diffusion"))[:120],
                    "terminal": update.get("terminal") is True,
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }
                with run.lock:
                    run.draft = draft
                    run.draft_sequence += 1

            strategy = search_factory(map_bundle) if search_factory is not None else BaselineSearch(generator)
            options = {"guidance_updates": run.guidance_updates} if search_factory is not None and run.guidance_updates is not None else {}
            if search_factory is not None:
                if run.generator_metadata.get("draft_streaming"):
                    options["on_draft"] = on_draft
                if run.refinement_epsilon is not None:
                    options["refinement_epsilon"] = run.refinement_epsilon
                if run.refinement_seconds is not None:
                    options["refinement_seconds"] = run.refinement_seconds
                if run.refinement_unlimited:
                    options["refinement_unlimited"] = True
            result = strategy.run(
                map_bundle, run.target, max_candidates=run.requested_candidates,
                seed=run.seed, on_candidate=on_candidate, cancel_event=run.cancel_event,
                **options,
            )
            with run.lock:
                trace = getattr(result, "trace", None)
                if isinstance(trace, dict):
                    run.trace = trace
                    run.strategy = trace.get("strategy", run.strategy)
                if not run.candidates and not run.cancel_event.is_set():
                    raise RuntimeError("The model returned no usable paragraphs. Try another target or model.")
                run.status = "cancelled" if run.cancel_event.is_set() else "error" if trace and trace.get("status") == "error" else "complete"
                if run.status == "error":
                    run.error = trace.get("error")
        except Exception as exc:
            with run.lock:
                run.status = "cancelled" if run.cancel_event.is_set() else "error"
                run.error = None if run.cancel_event.is_set() else str(exc)
        finally:
            with run.lock:
                run.elapsed_seconds = time.monotonic() - run.started
            try:
                run_dir = cache_dir / "runs"
                run_dir.mkdir(parents=True, exist_ok=True)
                path = run_dir / f"{run.id}.json"
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(run.snapshot(), indent=2) + "\n")
                tmp.replace(path)
            except OSError:
                # A read-only run directory must not discard a verified candidate.
                pass
            finally:
                with state_lock:
                    if state["active"] == run.id:
                        state["active"] = None

    @app.post("/api/runs", status_code=202)
    def start_run(request: RunRequest):
        map_bundle = get_bundle()
        if request.map_id != map_bundle.map_id:
            raise HTTPException(409, "The map changed. Reload the page before generating.")
        metadata = generator.available()
        if not metadata.get("available"):
            raise HTTPException(503, metadata.get("error", "The local generator is unavailable."))
        if request.guidance_updates is not None and not metadata.get("guided"):
            raise HTTPException(422, "Encoder updates per pass apply only to guided diffusion.")
        if (request.refinement_epsilon is not None or request.refinement_seconds is not None or request.refinement_unlimited) and not metadata.get("refinement"):
            raise HTTPException(422, "Refinement tolerance and time budget require the canonical refinement backend.")
        with state_lock:
            if state["closed"]:
                raise HTTPException(503, "The server is shutting down.")
            if state["active"] is not None:
                raise HTTPException(409, "A generation is still finishing. Wait a moment and try again.")
            run = Run(
                id=uuid4().hex, map_id=request.map_id, target=request.target,
                seed=request.seed, requested_candidates=request.candidates,
                generator=generator.identity, encoder=map_bundle.payload()["encoder"]["id"],
                generator_metadata=metadata,
                guidance_updates=request.guidance_updates,
                refinement_epsilon=request.refinement_epsilon,
                refinement_seconds=request.refinement_seconds,
                refinement_unlimited=request.refinement_unlimited,
            )
            # Bound browser-session history; completed records remain on disk.
            while len(runs) >= 30:
                runs.pop(next(iter(runs)))
            runs[run.id] = run
            state["active"] = run.id
        threading.Thread(target=worker, args=(run, map_bundle),
                         name=f"point2text-{run.id[:8]}", daemon=True).start()
        return {"id": run.id}

    def find_run(run_id):
        with state_lock:
            run = runs.get(run_id)
        if run is None:
            raise HTTPException(404, "Run not found in this server session.")
        return run

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str):
        return find_run(run_id).snapshot()

    @app.delete("/api/runs/{run_id}")
    def cancel_run(run_id: str):
        run = find_run(run_id)
        run.cancel_event.set()
        return run.snapshot()

    def current_map_id(map_id: str | None):
        active_id = get_bundle().map_id
        if map_id is not None and map_id != active_id:
            raise HTTPException(409, "The map changed. Reload the page to view its saved paragraphs.")
        return active_id

    @app.get("/api/saved")
    def saved_paragraphs(map_id: str | None = None):
        active_id = current_map_id(map_id)
        return {"map_id": active_id, "paragraphs": library.list(active_id)}

    @app.post("/api/saved", status_code=201)
    def save_paragraph(request: SaveRequest):
        current_map_id(request.map_id)
        with state_lock:
            run = runs.get(request.run_id)
        if run is not None:
            record = run.snapshot()
        else:
            # Request validation permits UUID hex only, so this cannot escape runs/.
            try:
                record = json.loads((cache_dir / "runs" / f"{request.run_id}.json").read_text())
            except (OSError, ValueError):
                raise HTTPException(404, "The verified generation record was not found.")
        if record.get("map_id") != request.map_id:
            raise HTTPException(409, "This paragraph belongs to a different fixed map.")
        if record.get("status") == "running":
            raise HTTPException(409, "Wait for the search to finish or stop it before saving.")
        candidates = record.get("candidates", [])
        if request.candidate_index >= len(candidates):
            raise HTTPException(404, "The requested candidate was not found.")
        return library.save(record, request.candidate_index, request.title)

    @app.delete("/api/saved/{record_id}")
    def delete_saved_paragraph(record_id: str, map_id: str | None = None):
        active_id = current_map_id(map_id)
        if not library.delete(record_id, active_id):
            raise HTTPException(404, "Saved paragraph not found in this map.")
        return {"deleted": record_id, "map_id": active_id}

    return app


def tailnet_ipv4() -> str:
    """Resolve the active local Tailscale address without changing Serve routes."""
    try:
        result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                                text=True, check=True, timeout=10)
        address = ipaddress.IPv4Address(result.stdout.strip())
        if address not in ipaddress.IPv4Network("100.64.0.0/10"):
            raise ValueError("address is outside the Tailscale IPv4 range")
        return str(address)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Could not resolve this machine's Tailscale IPv4 address. Install "
            "the Tailscale CLI, connect to your tailnet, and check 'tailscale ip -4'."
        ) from exc


def main():
    parser = argparse.ArgumentParser(description="Run the local point2text demo.")
    binding = parser.add_mutually_exclusive_group()
    binding.add_argument("--host", default="127.0.0.1")
    binding.add_argument("--tailnet", action="store_true",
                         help="Bind directly to this machine's Tailscale IPv4 address.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=os.environ.get("POINT2TEXT_MODEL", "gemma4:12b"))
    args = parser.parse_args()
    if args.tailnet:
        try:
            args.host = tailnet_ipv4()
        except RuntimeError as exc:
            parser.error(str(exc))
    os.environ["POINT2TEXT_MODEL"] = args.model
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
