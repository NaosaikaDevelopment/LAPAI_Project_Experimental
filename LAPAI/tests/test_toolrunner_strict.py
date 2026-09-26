from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_runner():
    base = Path(__file__).parents[1]
    maincore = types.ModuleType("MainCore")
    maincore.__path__ = [str(base / "MainCore")]
    core = types.ModuleType("MainCore.core")
    core.__path__ = [str(base / "MainCore" / "core")]
    state = types.ModuleType("MainCore.core.state")
    state.cache = types.SimpleNamespace(tool_registry=None)

    sys.modules.setdefault("MainCore", maincore)
    sys.modules.setdefault("MainCore.core", core)
    sys.modules["MainCore.core.state"] = state

    name = "MainCore.core.trunner_testisolated"
    spec = importlib.util.spec_from_file_location(
        name,
        base / "MainCore" / "core" / "trunner.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TR = _load_runner()


class Registry:
    def __init__(self):
        self.tools = {}


def test_runner_executes_exact_function_result():
    registry = Registry()

    def add(a: int, b: int):
        return a + b

    registry.tools["add"] = {"function": add, "requires_state": []}
    assert TR.ToolRunner(registry).execute("add", {"a": 5, "b": 7}) == 12


def test_runner_accepts_json_argument_string():
    registry = Registry()

    def add(a: int, b: int):
        return a + b

    registry.tools["add"] = {"function": add, "requires_state": []}
    assert TR.ToolRunner(registry).execute("add", '{"a": 2, "b": 3}') == 5


def test_runner_rejects_invalid_json():
    registry = Registry()

    def ping():
        return "pong"

    registry.tools["ping"] = {"function": ping, "requires_state": []}

    with pytest.raises(ValueError, match="valid JSON"):
        TR.ToolRunner(registry).execute("ping", "{bad")


def test_runner_rejects_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        TR.ToolRunner(Registry()).execute("missing", {})


def test_runner_rejects_missing_state_before_execution():
    registry = Registry()
    called = []

    def protected():
        called.append(True)
        return "should not run"

    registry.tools["protected"] = {
        "function": protected,
        "requires_state": ["ready"],
    }

    with pytest.raises(RuntimeError, match="inactive state"):
        TR.ToolRunner(registry).execute(
            "protected",
            {},
            context_state={},
        )

    assert called == []


def test_runner_allows_active_required_state():
    registry = Registry()

    def protected():
        return "ok"

    registry.tools["protected"] = {
        "function": protected,
        "requires_state": ["ready"],
    }

    assert TR.ToolRunner(registry).execute(
        "protected",
        {},
        context_state={"ready": True},
    ) == "ok"


def test_runner_enforces_signature():
    registry = Registry()

    def add(a: int, b: int):
        return a + b

    registry.tools["add"] = {"function": add, "requires_state": []}

    with pytest.raises(TypeError):
        TR.ToolRunner(registry).execute("add", {"a": 1})

    with pytest.raises(TypeError):
        TR.ToolRunner(registry).execute(
            "add",
            {"a": 1, "b": 2, "c": 3},
        )


def test_runner_rejects_non_dict_arguments():
    registry = Registry()

    def ping():
        return "pong"

    registry.tools["ping"] = {"function": ping, "requires_state": []}

    with pytest.raises(TypeError, match="dictionary"):
        TR.ToolRunner(registry).execute("ping", ["not", "dict"])
