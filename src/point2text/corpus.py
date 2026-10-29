"""The authored demo corpus; not a research benchmark or retrieved evidence."""

import json
from importlib.resources import files
from pathlib import Path
from typing import Sequence

from .contracts import CorpusRecord


def validate_records(records: Sequence[CorpusRecord]) -> list[CorpusRecord]:
    records = list(records)
    if any(not isinstance(record, CorpusRecord) for record in records):
        raise ValueError("Corpus must contain CorpusRecord values")
    if any(not all(isinstance(value, str) and value.strip()
                   for value in (r.id, r.topic, r.title, r.text)) for r in records):
        raise ValueError("Corpus records must have nonempty fields")
    if len({record.id for record in records}) != len(records):
        raise ValueError("Corpus IDs must be unique")
    return records


def load_corpus(path: str | Path | None = None) -> list[CorpusRecord]:
    """Read the bundled demo or explicit JSONL; never retrieve outside evidence."""
    source = Path(path) if path is not None else files("point2text").joinpath("assets/corpus.jsonl")
    records = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(CorpusRecord(**json.loads(line)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid corpus record at {source}:{number}: {exc}") from exc
    return validate_records(records)
