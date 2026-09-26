from __future__ import annotations

import importlib.util
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_pm():
    base = Path(__file__).parents[1]
    maincore = types.ModuleType("MainCore")
    maincore.__path__ = [str(base / "MainCore")]
    core = types.ModuleType("MainCore.core")
    core.__path__ = [str(base / "MainCore" / "core")]
    state = types.ModuleType("MainCore.core.state")
    state.cache = types.SimpleNamespace(conf={}, tokenizer=None, prompt_manager=None)

    sys.modules.setdefault("MainCore", maincore)
    sys.modules.setdefault("MainCore.core", core)
    sys.modules["MainCore.core.state"] = state

    name = "MainCore.core.promptmanager_testisolated"
    spec = importlib.util.spec_from_file_location(
        name,
        base / "MainCore" / "core" / "promptmanager.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, state.cache


def test_relation_stays_full_for_first_three_minutes():
    pm, _ = _load_pm()
    manager = pm.PromptManager()
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    manager._now = staticmethod(lambda: now)
    assert manager._relation((now - timedelta(seconds=180)).isoformat()) == 1.0


def test_relation_decays_by_half_percent_after_three_minutes():
    pm, _ = _load_pm()
    manager = pm.PromptManager()
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    manager._now = staticmethod(lambda: now)
    assert manager._relation((now - timedelta(seconds=181)).isoformat()) == 0.995


def test_relation_reaches_zero_after_ten_hours():
    pm, _ = _load_pm()
    manager = pm.PromptManager()
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    manager._now = staticmethod(lambda: now)
    assert manager._relation((now - timedelta(hours=10)).isoformat()) == 0.0


def test_system_messages_are_protected():
    pm, _ = _load_pm()
    manager = pm.PromptManager()
    system = {"role": "system", "content": "core"}
    manager.sync([system])
    assert manager.relation_for(system) == 1.0


def test_tool_call_turn_is_atomic_when_trimmed():
    pm, _ = _load_pm()
    manager = pm.PromptManager()
    old_user = {"role": "user", "content": "old request", "_lapai_created_at": "2026-09-25T10:00:00+00:00"}
    old_assistant = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function"}], "_lapai_created_at": "2026-09-25T10:00:01+00:00"}
    old_tool = {"role": "tool", "tool_call_id": "c1", "name": "x", "content": "old result", "_lapai_created_at": "2026-09-25T10:00:02+00:00"}
    new_user = {"role": "user", "content": "new request", "_lapai_created_at": "2026-09-25T11:00:00+00:00"}
    messages = [old_user, old_assistant, old_tool, new_user]
    manager.sync(messages)
    assert manager.trim_oldest_batch(messages, target_count=1) == 3
    assert messages == [new_user]


def test_expired_turn_removes_entire_atomic_group():
    pm, _ = _load_pm()
    manager = pm.PromptManager()
    now = datetime.now(timezone.utc)
    group = [
        {"role": "user", "content": "old", "_lapai_created_at": (now - timedelta(hours=10)).isoformat()},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}], "_lapai_created_at": (now - timedelta(hours=10)).isoformat()},
        {"role": "tool", "tool_call_id": "c1", "name": "x", "content": "result", "_lapai_created_at": (now - timedelta(hours=10)).isoformat()},
    ]
    messages = list(group)
    assert manager.remove_expired(messages) == 3
    assert messages == []


def test_prepare_strips_internal_metadata():
    pm, _ = _load_pm()
    manager = pm.PromptManager()
    messages = [{"role": "user", "content": "hello", "_lapai_created_at": "2026-09-25T10:00:00+00:00", "_lapai_internal": 123}]
    prepared = manager.prepare_for_model(messages)
    assert "_lapai_created_at" not in prepared[0]
    assert "_lapai_internal" not in prepared[0]


def test_soft_limit_reserves_output_budget():
    pm, cache = _load_pm()
    cache.conf = {"promptContextLimit": 1000, "promptSoftLimitRatio": 0.80, "promptReservedOutputTokens": 200}
    assert pm.PromptManager().prompt_soft_limit == 640


def test_newest_turn_is_preserved_during_trim():
    pm, _ = _load_pm()
    manager = pm.PromptManager()
    old = [
        {"role": "user", "content": f"old-{i}", "_lapai_created_at": f"2026-09-25T10:{i:02d}:00+00:00"}
        for i in range(3)
    ]
    newest = {"role": "user", "content": "NEWEST", "_lapai_created_at": "2026-09-25T12:00:00+00:00"}
    messages = old + [newest]
    manager.sync(messages)
    manager.trim_oldest_batch(messages, target_count=20)
    assert messages == [newest]


def test_prepare_trims_when_prompt_reaches_soft_limit():
    pm, cache = _load_pm()
    cache.conf = {
        "promptContextLimit": 80,
        "promptSoftLimitRatio": 0.80,
        "promptReservedOutputTokens": 0,
        "promptTrimBatchMessages": 20,
    }
    manager = pm.PromptManager()
    messages = [
        {
            "role": "user",
            "content": "old content " * 10,
            "_lapai_created_at": f"2026-09-25T10:0{i}:00+00:00",
        }
        for i in range(3)
    ]
    messages.append({
        "role": "user",
        "content": "new",
        "_lapai_created_at": "2026-09-25T12:00:00+00:00",
    })

    prepared = manager.prepare_for_model(messages)

    assert prepared[-1]["content"] == "new"
    assert len(prepared) < 4
    assert manager.last_stats["trimmed_messages"] > 0
