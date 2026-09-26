from __future__ import annotations

import importlib.util
import sqlite3
import sys
import types
from pathlib import Path


def _load_memorystore(tmp_path):
    base = Path(__file__).parents[1]
    maincore = types.ModuleType("MainCore")
    maincore.__path__ = [str(base / "MainCore")]
    core = types.ModuleType("MainCore.core")
    core.__path__ = [str(base / "MainCore" / "core")]
    state = types.ModuleType("MainCore.core.state")
    state.cache = types.SimpleNamespace(DB_FILE=str(tmp_path / "memory.db"))

    sys.modules.setdefault("MainCore", maincore)
    sys.modules.setdefault("MainCore.core", core)
    sys.modules["MainCore.core.state"] = state

    name = "MainCore.core.memorystore_testisolated"
    spec = importlib.util.spec_from_file_location(
        name,
        base / "MainCore" / "core" / "memorystore.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, state.cache


def _commit(ms, **kwargs):
    defaults = {
        "session_id": 1,
        "confidence": 0.95,
        "created_at": "2026-09-25T10:00:00+00:00",
    }
    defaults.update(kwargs)
    return ms.commit_memory(**defaults)


def test_question_logged_but_never_stored(tmp_path):
    ms, cache = _load_memorystore(tmp_path)
    result = _commit(
        ms,
        user_text="What is my favorite word?",
        kind="question",
        category="preference",
    )

    assert result["success"] is True
    assert result["stored"] is False

    conn = sqlite3.connect(cache.DB_FILE)
    try:
        assert conn.execute("SELECT COUNT(*) FROM input_classifications").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM long_term_memories").fetchone()[0] == 0
    finally:
        conn.close()


def test_preference_preserves_raw_evidence(tmp_path):
    ms, cache = _load_memorystore(tmp_path)
    text = "My fav word is The fallen sky bring the truth of the lies"
    result = _commit(
        ms,
        user_text=text,
        kind="preference",
        category="favorite_word",
    )

    assert result["stored"] is True

    conn = sqlite3.connect(cache.DB_FILE)
    try:
        row = conn.execute(
            "SELECT text, normalized_text, kind, category, confidence, support_count "
            "FROM long_term_memories"
        ).fetchone()
    finally:
        conn.close()

    assert row == (text, text.casefold(), "preference", "favorite_word", 0.95, 1)


def test_exact_duplicate_reinforces_single_record(tmp_path):
    ms, cache = _load_memorystore(tmp_path)
    kwargs = {
        "user_text": "I like bakso.",
        "kind": "preference",
        "category": "food",
        "session_id": 1,
    }

    first = _commit(ms, **kwargs, confidence=0.80, created_at="2026-09-25T10:00:00+00:00")
    second = _commit(ms, **kwargs, confidence=0.92, created_at="2026-09-25T11:00:00+00:00")

    assert first["updated"] is False
    assert second["updated"] is True
    assert second["support_count"] == 2
    assert second["confidence"] == 0.92

    conn = sqlite3.connect(cache.DB_FILE)
    try:
        assert conn.execute("SELECT COUNT(*) FROM long_term_memories").fetchone()[0] == 1
        assert conn.execute("SELECT support_count FROM long_term_memories").fetchone()[0] == 2
    finally:
        conn.close()


def test_low_confidence_is_rejected_from_long_term_memory(tmp_path):
    ms, cache = _load_memorystore(tmp_path)
    result = _commit(
        ms,
        user_text="Maybe I like sushi.",
        kind="preference",
        category="food",
        confidence=0.69,
    )

    assert result["stored"] is False

    conn = sqlite3.connect(cache.DB_FILE)
    try:
        assert conn.execute("SELECT COUNT(*) FROM input_classifications").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM long_term_memories").fetchone()[0] == 0
    finally:
        conn.close()


def test_invalid_kind_fails_closed(tmp_path):
    ms, cache = _load_memorystore(tmp_path)
    result = _commit(
        ms,
        user_text="bad classification",
        kind="made_up_kind",
        category="x",
    )

    assert result["success"] is False
    assert result["stored"] is False

    conn = sqlite3.connect(cache.DB_FILE)
    try:
        assert conn.execute("SELECT COUNT(*) FROM input_classifications").fetchone()[0] == 0
    finally:
        conn.close()


def test_retrieval_excludes_questions(tmp_path):
    ms, _ = _load_memorystore(tmp_path)
    _commit(ms, user_text="My favorite word is eclipse.", kind="preference", category="favorite_word")
    _commit(ms, user_text="What is my favorite word?", kind="question", category="favorite_word", created_at="2026-09-25T10:01:00+00:00")

    result = ms.recall_long_term_memory("favorite word", limit=5)

    assert len(result) == 1
    assert result[0]["content"] == "My favorite word is eclipse."
    assert result[0]["kind"] == "preference"


def test_relation_decay_is_newest_first_inside_category(tmp_path):
    ms, _ = _load_memorystore(tmp_path)
    for text, ts in [
        ("old food fact", "2026-09-25T10:00:00+00:00"),
        ("middle food fact", "2026-09-25T11:00:00+00:00"),
        ("new food fact", "2026-09-25T12:00:00+00:00"),
    ]:
        _commit(ms, user_text=text, kind="fact", category="food", created_at=ts)

    result = ms.recall_long_term_memory("food fact", limit=10)
    assert [r["relation_score"] for r in result] == [1.0, 0.995, 0.995 ** 2]
