from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_memory_tools():
    base = Path(__file__).parents[1]
    maincore = types.ModuleType("MainCore")
    maincore.__path__ = [str(base / "MainCore")]
    core = types.ModuleType("MainCore.core")
    core.__path__ = [str(base / "MainCore" / "core")]

    statecore = types.ModuleType("MainCore.statecore")
    statecore.cache = types.SimpleNamespace(
        current_user_msg="",
        session_id=1,
        crntTime="2026-09-25T10:00:00+00:00",
        conf={"limitrrm": 5},
    )

    memorystore = types.ModuleType("MainCore.core.memorystore")
    memorystore.commit_memory = lambda **kwargs: {"success": True, "stored": False}
    memorystore.recall_long_term_memory = lambda user_input, **kwargs: []
    statecore.recall_relevant_memory = lambda **kwargs: memorystore.recall_long_term_memory(**kwargs)

    sys.modules.setdefault("MainCore", maincore)
    sys.modules.setdefault("MainCore.core", core)
    sys.modules["MainCore.statecore"] = statecore
    sys.modules["MainCore.core.memorystore"] = memorystore

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    commit_mod = load(
        base / "MainCore" / "core" / "tools" / "memory_commit.py",
        "MainCore.core.tools.memory_commit_testisolated",
    )
    recall_mod = load(
        base / "MainCore" / "core" / "tools" / "memorydecision.py",
        "MainCore.core.tools.memorydecision_testisolated",
    )
    return statecore.cache, commit_mod, recall_mod


def test_memory_commit_uses_exact_current_user_and_emits_state(monkeypatch):
    cache, mod, _ = _load_memory_tools()
    cache.current_user_msg = "What is my favorite word?"
    seen = {}

    def fake_commit_memory(**kwargs):
        seen.update(kwargs)
        return {"success": True, "stored": False}

    monkeypatch.setattr(mod, "commit_memory", fake_commit_memory)
    result = mod.memory_commit(kind="question", category="preference", confidence=0.95)

    assert seen["user_text"] == "What is my favorite word?"
    assert result["_state"] == {"memory_query": True}


def test_memory_commit_does_not_activate_query_for_event(monkeypatch):
    cache, mod, _ = _load_memory_tools()
    cache.current_user_msg = "I ate noodles today."
    monkeypatch.setattr(mod, "commit_memory", lambda **kwargs: {"success": True, "stored": True})

    result = mod.memory_commit(kind="event", category="food", confidence=0.95)
    assert result["_state"] == {"memory_query": False}


def test_recallmemory_is_zero_arg_and_uses_current_user(monkeypatch):
    cache, _, mod = _load_memory_tools()
    cache.current_user_msg = "naove, what is my fav word?"
    seen = {}

    def fake_recall(user_input, **kwargs):
        seen["user_input"] = user_input
        seen.update(kwargs)
        return [{
            "content": "My fav word is eclipse.",
            "kind": "preference",
            "category": "favorite_word",
            "score": 0.9,
            "relation_score": 1.0,
            "confidence": 0.95,
            "support_count": 1,
        }]

    monkeypatch.setattr(mod, "recall_relevant_memory", fake_recall)
    result = mod.recallmemory()

    assert seen["user_input"] == "naove, what is my fav word?"
    assert "My fav word is eclipse." in result


def test_recallmemory_returns_explicit_empty_result(monkeypatch):
    cache, _, mod = _load_memory_tools()
    cache.current_user_msg = "hello"
    monkeypatch.setattr(mod, "recall_relevant_memory", lambda user_input, **kwargs: [])
    assert mod.recallmemory() == "No relevant memory was found."
