from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def test_initialize_storage_is_idempotent(isolated_core):
    core = isolated_core

    core.init_db()
    core.init_db()
    core.init_memory_state()
    core.init_memory_state()
    core.init_learning_db()
    core.init_learning_db()

    conn = sqlite3.connect(core.DB_FILE)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        conn.close()

    assert "sessions" in tables
    assert "messages" in tables
    assert "memory_state" in tables


def test_create_session_sanitizes_filename_and_persists(isolated_core, clean_workspace):
    core = isolated_core
    core.init_db()

    sid, json_file = core.create_session("../../evil title?!")
    assert isinstance(sid, int)

    # The old check (`startswith(chats_dir)`) passed even for "chats/../../evil",
    # so it never caught path traversal. Resolve the path and compare properly.
    chats = (clean_workspace / "chats").resolve()
    resolved = Path(json_file).resolve()
    assert ".." not in Path(json_file).parts, f"path traversal left in {json_file!r}"
    assert chats in resolved.parents, f"{resolved} escapes {chats}"
    assert not any(ch in Path(json_file).name for ch in '?!<>:"|*'), "unsafe characters in filename"

    conn = sqlite3.connect(core.DB_FILE)
    try:
        row = conn.execute("SELECT id, title, json_file FROM sessions WHERE id = ?", (sid,)).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == sid
    assert row[1] == "../../evil title?!"


def test_memory_state_round_trip(isolated_core):
    core = isolated_core
    core.init_db()
    session_id = 42

    core.save_memory_state(session_id, "summary A", "2026-09-24 19:00:00", 7)
    first = core.get_memory_state(session_id)
    assert first == {
        "summary": "summary A",
        "summarized_until": "2026-09-24 19:00:00",
        "summarized_until_rowid": 7,
    }

    core.save_memory_state(session_id, "summary B", "2026-09-24 20:00:00", 12)
    second = core.get_memory_state(session_id)
    assert second["summary"] == "summary B"
    assert second["summarized_until_rowid"] == 12


def test_memory_state_sessions_are_isolated(isolated_core):
    core = isolated_core
    core.init_db()

    core.save_memory_state(1, "session one", "t1", 1)
    core.save_memory_state(2, "session two", "t2", 2)

    assert core.get_memory_state(1)["summary"] == "session one"
    assert core.get_memory_state(2)["summary"] == "session two"


def test_search_memory_uses_parameter_binding(isolated_core):
    core = isolated_core
    core.init_db()

    conn = sqlite3.connect(core.DB_FILE)
    try:
        conn.execute("INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)", (1, "user", "alpha beta"))
        conn.commit()
    finally:
        conn.close()

    results = core.search_memory("alpha", limit=5)
    assert results
    assert results[0][3] == "alpha beta"


@pytest.mark.parametrize(
    "query",
    ['a"b', "a:b", "a AND (b OR c)", "'; DROP TABLE messages; --", "😀 日本語", "*", "(", ""],
)
def test_search_memory_does_not_crash_on_sql_special_characters(isolated_core, query):
    core = isolated_core
    core.init_db()

    # Put a row in first so a real MATCH is executed against non-empty data.
    conn = sqlite3.connect(core.DB_FILE)
    try:
        conn.execute("INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)", (1, "user", "a b c"))
        conn.commit()
    finally:
        conn.close()

    try:
        core.search_memory(query, limit=5)
    except sqlite3.OperationalError as exc:
        pytest.fail(f"search_memory leaked raw FTS syntax failure for query={query!r}: {exc}")

    # The table must still exist afterwards (the injection attempt must be harmless).
    conn = sqlite3.connect(core.DB_FILE)
    try:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        conn.close()
