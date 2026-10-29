"""Frozen ordinary PCA and actual-text scoring, independent of generation."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

from .contracts import CorpusRecord, TextEncoder
from .corpus import load_corpus

SCHEMA = "point2text-pca-v1"
METRIC = "euclidean-unwhitened-pca"


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(metadata: dict, arrays: tuple[np.ndarray, ...] = ()) -> str:
    digest = hashlib.sha256(_json(metadata).encode())
    for array in arrays:
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(_json(list(canonical.shape)).encode())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _embeddings(value: np.ndarray, rows: int, dimensions: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != rows or array.shape[1] < 2:
        raise ValueError("Encoder must return a [texts, dimensions>=2] matrix")
    if dimensions is not None and array.shape[1] != dimensions:
        raise ValueError("Encoder dimensions do not match this map")
    if not np.isfinite(array).all():
        raise ValueError("Encoder returned nonfinite embeddings")
    return array


def _target(target: tuple[float, float]) -> np.ndarray:
    value = np.asarray(target, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("Target must be two finite PCA coordinates")
    return value


@dataclass(frozen=True)
class MapBundle:
    encoder: TextEncoder
    records: list[CorpusRecord]
    embeddings: np.ndarray
    mean: np.ndarray
    components: np.ndarray  # [2, dimensions], orthonormal rows
    explained_variance: np.ndarray
    map_id: str

    def __post_init__(self) -> None:
        for array in (self.embeddings, self.mean, self.components, self.explained_variance):
            array.setflags(write=False)

    @classmethod
    def build(cls, encoder: TextEncoder, cache_dir: Path) -> "MapBundle":
        records = load_corpus()
        if len(records) < 3:
            raise ValueError("A PCA map needs at least three corpus records")
        corpus_id = _digest({"records": [asdict(record) for record in records]})
        metadata = {"schema": SCHEMA, "encoder_id": encoder.identity,
                    "corpus_id": corpus_id, "metric": METRIC}
        key = _digest(metadata)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        array_path, manifest_path = cache_dir / f"{key}.npz", cache_dir / f"{key}.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(manifest.get(k) != v for k, v in metadata.items()):
                raise ValueError("Cache metadata mismatch")
            with np.load(array_path, allow_pickle=False) as saved:
                embeddings = _embeddings(saved["embeddings"], len(records))
                mean = np.asarray(saved["mean"], dtype=np.float64)
                components = np.asarray(saved["components"], dtype=np.float64)
                variance = np.asarray(saved["explained_variance"], dtype=np.float64)
            dimensions = embeddings.shape[1]
            if (mean.shape != (dimensions,) or components.shape != (2, dimensions)
                    or variance.shape != (2,) or not all(np.isfinite(a).all() for a in (mean, components, variance))
                    or not np.allclose(components @ components.T, np.eye(2), atol=1e-7)):
                raise ValueError("Invalid cached PCA arrays")
            map_id = _digest(metadata, (embeddings, mean, components, variance))
            if map_id != manifest.get("map_id"):
                raise ValueError("Cache fingerprint mismatch")
            return cls(encoder, records, embeddings, mean, components, variance, map_id)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            # Partial/corrupt/stale caches are safely replaced; model caches are separate.
            pass

        embeddings = _embeddings(encoder.encode([r.text for r in records]), len(records))
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
        components = directions[:2].copy()
        for axis in components:
            if axis[np.argmax(np.abs(axis))] < 0:
                axis *= -1
        total = float(np.sum(singular_values**2))
        variance = singular_values[:2] ** 2 / total if total > 0 else np.zeros(2)
        map_id = _digest(metadata, (embeddings, mean, components, variance))
        manifest = dict(metadata, map_id=map_id, dimensions=embeddings.shape[1], corpus_size=len(records))
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, embeddings=embeddings, mean=mean,
                                components=components, explained_variance=variance)
        os.replace(temporary_path, array_path)
        with tempfile.NamedTemporaryFile(dir=cache_dir, mode="w", encoding="utf-8", suffix=".json", delete=False) as temporary:
            temporary.write(_json(manifest))
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, manifest_path)
        return cls(encoder, records, embeddings, mean, components, variance, map_id)

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        value = np.asarray(embeddings, dtype=np.float64)
        if value.ndim not in (1, 2) or value.shape[-1] != self.mean.size or not np.isfinite(value).all():
            raise ValueError("Invalid embedding dimensions or values")
        return (value - self.mean) @ self.components.T

    def payload(self) -> dict:
        projected = self.project(self.embeddings)
        minimum, maximum = projected.min(axis=0), projected.max(axis=0)
        padding = max(float(np.max(maximum - minimum)), 0.1) * 0.1
        return {
            "map_id": self.map_id,
            "encoder": {"id": self.encoder.identity, "dimensions": int(self.mean.size)},
            "points": [dict(asdict(record), x=float(point[0]), y=float(point[1]))
                       for record, point in zip(self.records, projected, strict=True)],
            "bounds": {"xmin": float(minimum[0] - padding), "xmax": float(maximum[0] + padding),
                       "ymin": float(minimum[1] - padding), "ymax": float(maximum[1] + padding)},
            "explained_variance": self.explained_variance.tolist(),
            "corpus_size": len(self.records),
            "topics": sorted({record.topic for record in self.records}),
        }

    def nearest(self, target: tuple[float, float], k: int = 4) -> list[CorpusRecord]:
        if k < 1:
            raise ValueError("k must be positive")
        distances = np.linalg.norm(self.project(self.embeddings) - _target(target), axis=1)
        return [self.records[int(i)] for i in np.argsort(distances, kind="stable")[:k]]

    def score(self, text: str, target: tuple[float, float]) -> tuple[tuple[float, float], float]:
        location = _target(target)
        embedding = _embeddings(self.encoder.encode([text]), 1, self.mean.size)[0]
        point = self.project(embedding)
        return (float(point[0]), float(point[1])), float(np.linalg.norm(point - location))
