"""Host-local saved paragraphs; records never participate in fitting a map."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


class ParagraphLibrary:
    """SQLite transactions make saves safe across requests and server restarts."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=15)
        try:
            with connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS paragraphs (
                    id TEXT PRIMARY KEY, map_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, candidate_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL, record TEXT NOT NULL,
                    UNIQUE(map_id, run_id, candidate_index)
                )""")
                yield connection
        finally:
            connection.close()

    def list(self, map_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record FROM paragraphs WHERE map_id = ? ORDER BY created_at DESC, id",
                (map_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save(self, run: dict, candidate_index: int, title: str | None = None) -> dict:
        candidate = run["candidates"][candidate_index]
        record = {
            "id": uuid4().hex,
            "map_id": run["map_id"], "encoder": run["encoder"],
            "title": title or " ".join(candidate["text"].split()[:9]),
            "text": candidate["text"], "x": candidate["x"], "y": candidate["y"],
            "target": run["target"], "distance": candidate["distance"],
            "seed": candidate["seed"], "elapsed_seconds": candidate["elapsed_seconds"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "run_id": run["id"], "candidate_index": candidate_index,
                "created_at": run["created_at"], "generator": run["generator"],
                "generator_metadata": run["generator_metadata"],
                "strategy": run["strategy"], "trace": run.get("trace"),
                "status": run["status"],
                "settings": {key: run.get(key) for key in (
                    "seed", "requested_candidates", "guidance_updates",
                    "refinement_epsilon", "refinement_seconds", "refinement_unlimited",
                )},
            },
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO paragraphs VALUES (?, ?, ?, ?, ?, ?)",
                (record["id"], record["map_id"], run["id"], candidate_index,
                 record["created_at"], json.dumps(record, allow_nan=False)),
            )
            row = connection.execute(
                "SELECT record FROM paragraphs WHERE map_id = ? AND run_id = ? AND candidate_index = ?",
                (run["map_id"], run["id"], candidate_index),
            ).fetchone()
        return json.loads(row[0])

    def delete(self, record_id: str, map_id: str) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                "DELETE FROM paragraphs WHERE id = ? AND map_id = ?", (record_id, map_id)
            )
        return result.rowcount > 0
