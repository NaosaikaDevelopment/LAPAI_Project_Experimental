from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")


VECTORS = {
    "apple memory": np.array([1, 0, 0], dtype=np.float32),
    "banana memory": np.array([0, 1, 0], dtype=np.float32),
    "computer memory": np.array([0, 0, 1], dtype=np.float32),
    "query apple": np.array([1, 0, 0], dtype=np.float32),
    "query banana": np.array([0, 1, 0], dtype=np.float32),
}


def _patch_deterministic_embeddings(monkeypatch, module, dim: int = 3):
    def fake_embedding(text: str, embedding_type="passage"):
        key = text.replace("query: ", "").replace("passage: ", "")
        vec = VECTORS.get(key, np.array([1, 1, 0], dtype=np.float32))
        vec = vec / np.linalg.norm(vec)
        if dim != vec.shape[0]:
            vec = np.resize(vec, dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
        return vec.astype(np.float32)

    monkeypatch.setattr(module, "generate_embedding", fake_embedding)
    for name in ("add_to_faiss", "recall_from_faiss", "generate_embedding"):
        obj = getattr(module, name, None)
        globals_dict = getattr(obj, "__globals__", None)
        if globals_dict is not None and "generate_embedding" in globals_dict:
            monkeypatch.setitem(globals_dict, "generate_embedding", fake_embedding)


def _fresh_index(monkeypatch, module, dim: int = 3):
    index = faiss.IndexFlatIP(dim)
    mapping: dict = {}
    monkeypatch.setattr(module, "faiss_index", index, raising=False)
    monkeypatch.setattr(module, "id_map", mapping, raising=False)

    cache = getattr(module, "cache", None)
    if cache is not None:
        monkeypatch.setattr(cache, "faiss_index", index, raising=False)
        monkeypatch.setattr(cache, "id_map", mapping, raising=False)
    return index, mapping


def test_faiss_round_trip_and_map_consistency(isolated_core, monkeypatch):
    core = isolated_core
    _patch_deterministic_embeddings(monkeypatch, core)
    index, mapping = _fresh_index(monkeypatch, core)

    index, mapping = core.add_to_faiss(index, mapping, "apple memory", "u1", importance=1.0)
    index, mapping = core.add_to_faiss(index, mapping, "banana memory", "u2", importance=2.0)

    assert index.ntotal == 2
    assert len(mapping) == 2

    results = core.recall_from_faiss(
        core.generate_embedding("query apple", embedding_type="query"),
        topk=2,
        threshold=0.9,
    )
    assert results
    assert results[0]["content"] == "apple memory"
    assert results[0]["similarity"] >= 0.9

    saved_map = json.loads(Path(core.FAISS_MAP).read_text(encoding="utf-8"))
    saved_index = faiss.read_index(core.FAISS_INDEX)
    assert saved_index.ntotal == len(saved_map)


def test_faiss_threshold_filters_dissimilar_results(isolated_core, monkeypatch):
    core = isolated_core
    _patch_deterministic_embeddings(monkeypatch, core)
    index, mapping = _fresh_index(monkeypatch, core)
    core.add_to_faiss(index, mapping, "apple memory", "u1", importance=1.0)

    far = core.recall_from_faiss(
        core.generate_embedding("query banana", embedding_type="query"),
        topk=5,
        threshold=0.5,
    )
    assert far == []


def test_faiss_rejects_dimension_mismatch(isolated_core, monkeypatch):
    core = isolated_core
    _patch_deterministic_embeddings(monkeypatch, core, dim=4)
    index, mapping = _fresh_index(monkeypatch, core, dim=3)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        core.add_to_faiss(index, mapping, "bad dimension", "u1")
    assert index.ntotal == 0


def test_faiss_empty_index_returns_empty(isolated_core, monkeypatch):
    core = isolated_core
    _fresh_index(monkeypatch, core)
    result = core.recall_from_faiss(
        np.array([1, 0, 0], dtype=np.float32),
        topk=5,
        threshold=0.5,
    )
    assert result == []
