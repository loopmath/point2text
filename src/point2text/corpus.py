"""The authored demo corpus; not a research benchmark or retrieved evidence."""

import json
from importlib.resources import files

from .contracts import CorpusRecord


def load_corpus() -> list[CorpusRecord]:
    path = files("point2text").joinpath("assets/corpus.jsonl")
    records = [CorpusRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({record.id for record in records}) != len(records):
        raise ValueError("Corpus IDs must be unique")
    if any(not all((r.id, r.topic, r.title, r.text.strip())) for r in records):
        raise ValueError("Corpus records must have nonempty fields")
    return records
