from __future__ import annotations

import json
from pathlib import Path

import pytest


CASES_PATH = Path(__file__).with_name("regression_cases.json")
REAL_CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))


def test_regression_case_fixtures_are_nonempty():
    assert REAL_CASES
    for case in REAL_CASES:
        assert case["id"]
        assert case["text"]
        assert case["query"]


@pytest.fixture
def no_faiss(isolated_core, monkeypatch):
    """Disable FAISS through the actual globals used by append_message."""
    if getattr(isolated_core, "add_to_faiss", None) is None:
        pytest.fail("add_to_faiss is missing; memory persistence contract is incomplete")

    noop = lambda *args, **kwargs: None
    monkeypatch.setattr(isolated_core, "add_to_faiss", noop)

    append = getattr(isolated_core, "append_message", None)
    globals_dict = getattr(append, "__globals__", None)
    if globals_dict is not None and "add_to_faiss" in globals_dict:
        monkeypatch.setitem(globals_dict, "add_to_faiss", noop)

    return isolated_core


# Every case in regression_cases.json is exercised (before, only case [0] was used,
# so the unicode and long-text cases in the JSON file were never actually tested).
@pytest.mark.parametrize("case", REAL_CASES, ids=[c["id"] for c in REAL_CASES])
def test_memory_regression_case_persists_in_fts(no_faiss, case):
    core = no_faiss
    core.init_db()

    sid, session_file = core.create_session("regression-memory")
    core.append_message(sid, session_file, "user", case["text"])
    rows = core.search_memory(case["query"], limit=5)
    assert any(row[3] == case["text"] for row in rows), (
        f"[{case['id']}] {case['goal']} - stored memory could not be retrieved by FTS"
    )


def test_recent_memory_is_session_isolated(no_faiss):
    core = no_faiss
    core.init_db()

    sid1, file1 = core.create_session("session-one")
    sid2, file2 = core.create_session("session-two")
    core.append_message(sid1, file1, "user", "ONLY-SESSION-ONE")
    core.append_message(sid2, file2, "user", "ONLY-SESSION-TWO")

    first = core.recall_recent_memory(sid1, minutes=5)
    second = core.recall_recent_memory(sid2, minutes=5)
    assert any(x["content"] == "ONLY-SESSION-ONE" for x in first)
    assert not any(x["content"] == "ONLY-SESSION-TWO" for x in first)
    assert any(x["content"] == "ONLY-SESSION-TWO" for x in second)
    assert not any(x["content"] == "ONLY-SESSION-ONE" for x in second)


def test_persona_extensionless_path_is_supported_or_explicitly_documented(isolated_core, clean_workspace):
    """Current LAPAI design uses Settings/PersonaAI without requiring .txt."""
    settings = clean_workspace / "Settings"
    settings.mkdir()
    (settings / "PersonaAI").write_text("persona sentinel", encoding="utf-8")

    if not hasattr(isolated_core, "load_persona"):
        pytest.fail("load_persona is part of the current LAPAI memory/persona contract but is missing")

    # The strict contract intentionally expects the extensionless path used by 1.5.1.
    assert isolated_core.load_persona() == "persona sentinel"


def test_real_case_text_round_trips_through_json(tmp_path: Path):
    payload = [case["text"] for case in REAL_CASES]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == payload
