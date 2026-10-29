import json
from types import SimpleNamespace

import numpy as np
import pytest

from point2text.contracts import CorpusRecord
from point2text.corpus import load_corpus
from point2text.encoder import MAX_TOKENS, SentenceEncoder
from point2text.geometry import MapBundle
import point2text.geometry as geometry


class TinyEncoder:
    identity = "tiny-encoder-v1"

    def __init__(self):
        self.calls = []
        self.vectors = {
            "alpha": [1.0, 0.0, 0.0, 0.0],
            "beta": [0.0, 2.0, 0.0, 0.0],
            "gamma": [0.0, 0.0, 3.0, 0.0],
            "delta": [-1.0, 0.0, 0.0, 1.0],
            "new paragraph": [0.2, 0.4, 0.5, 0.6],
        }

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts])


@pytest.fixture
def records(monkeypatch):
    values = [CorpusRecord(str(i), "topic", text.title(), text)
              for i, text in enumerate(("alpha", "beta", "gamma", "delta"))]
    monkeypatch.setattr(geometry, "load_corpus", lambda: values)
    return values


def test_cache_roundtrip_avoids_reencoding(tmp_path, records):
    original = MapBundle.build(TinyEncoder(), tmp_path)
    encoder = TinyEncoder()
    restored = MapBundle.build(encoder, tmp_path)
    assert encoder.calls == []
    assert original.map_id == restored.map_id
    assert original.payload() == restored.payload()
    assert len(list(tmp_path.glob("*.npz"))) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1
    np.testing.assert_allclose(restored.components @ restored.components.T, np.eye(2), atol=1e-12)
    with pytest.raises(ValueError):
        restored.components[0, 0] = 100


def test_map_id_changes_with_encoder_corpus_or_embedding_values(tmp_path, records):
    first = MapBundle.build(TinyEncoder(), tmp_path / "first")
    changed_encoder = TinyEncoder()
    changed_encoder.identity = "tiny-encoder-v2"
    second = MapBundle.build(changed_encoder, tmp_path / "second")
    changed_vectors = TinyEncoder()
    changed_vectors.vectors["alpha"][0] = 5.0
    third = MapBundle.build(changed_vectors, tmp_path / "third")
    records[0] = CorpusRecord("0", "another topic", "Alpha", "alpha")
    fourth = MapBundle.build(TinyEncoder(), tmp_path / "fourth")
    assert len({first.map_id, second.map_id, third.map_id, fourth.map_id}) == 4


def test_score_reembeds_actual_text(tmp_path, records):
    encoder = TinyEncoder()
    bundle = MapBundle.build(encoder, tmp_path)
    expected = (np.asarray(encoder.vectors["new paragraph"]) - bundle.mean) @ bundle.components.T
    point, distance = bundle.score("new paragraph", (0.1, -0.2))
    assert encoder.calls[-1] == ["new paragraph"]
    np.testing.assert_allclose(point, expected)
    assert distance == pytest.approx(np.linalg.norm(expected - [0.1, -0.2]))
    _, exact_distance = bundle.score("new paragraph", tuple(point))
    assert exact_distance == pytest.approx(0.0, abs=1e-12)


def test_omitted_residual_preserves_projection(tmp_path, records):
    bundle = MapBundle.build(TinyEncoder(), tmp_path)
    direction = np.asarray([0.2, -0.7, 1.2, 2.0])
    residual = direction - bundle.components.T @ (bundle.components @ direction)
    assert np.linalg.norm(residual) > 0.1
    np.testing.assert_allclose(bundle.project(bundle.embeddings[0] + residual),
                               bundle.project(bundle.embeddings[0]), atol=1e-12)


def test_payload_and_nearest_use_plain_records(tmp_path, records):
    bundle = MapBundle.build(TinyEncoder(), tmp_path)
    point = tuple(bundle.project(bundle.embeddings[2]))
    assert bundle.nearest(point, k=1) == [records[2]]
    assert len(bundle.nearest(point, k=100)) == len(records)
    payload = bundle.payload()
    json.dumps(payload, allow_nan=False)
    assert payload["corpus_size"] == 4
    assert payload["encoder"]["dimensions"] == 4
    for item in payload["points"]:
        assert payload["bounds"]["xmin"] <= item["x"] <= payload["bounds"]["xmax"]
        assert payload["bounds"]["ymin"] <= item["y"] <= payload["bounds"]["ymax"]


def test_invalid_values_and_corrupt_cache_are_handled(tmp_path, records):
    encoder = TinyEncoder()
    bundle = MapBundle.build(encoder, tmp_path)
    with pytest.raises(ValueError, match="finite"):
        bundle.score("new paragraph", (float("nan"), 0))
    with pytest.raises(ValueError, match="positive"):
        bundle.nearest((0, 0), k=0)
    encoder.vectors["new paragraph"] = [float("nan"), 0, 0, 0]
    with pytest.raises(ValueError, match="nonfinite"):
        bundle.score("new paragraph", (0, 0))
    next(tmp_path.glob("*.npz")).write_bytes(b"broken cache")
    restored = MapBundle.build(TinyEncoder(), tmp_path)
    assert restored.map_id == bundle.map_id


def test_curated_corpus_is_original_demo_shape():
    records = load_corpus()
    assert 100 <= len(records) <= 150
    assert len({record.topic for record in records}) == 10
    assert len({record.id for record in records}) == len(records)
    assert len({record.text for record in records}) == len(records)
    assert all(len(record.text.split()) <= 60 for record in records)


def test_encoder_rejects_overlength_before_scoring():
    encoder = object.__new__(SentenceEncoder)
    encoder.dimensions = 4
    encoder.model = SimpleNamespace(tokenizer=lambda texts, **kwargs: {
        "input_ids": [[1] * (len(text.split()) + 2) for text in texts]
    })
    with pytest.raises(ValueError, match=f"maximum is {MAX_TOKENS}"):
        encoder.encode(["word " * MAX_TOKENS])
