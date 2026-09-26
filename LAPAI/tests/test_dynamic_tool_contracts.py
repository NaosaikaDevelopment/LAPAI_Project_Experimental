from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_tools_state():
    base = Path(__file__).parents[1]
    maincore = types.ModuleType("MainCore")
    maincore.__path__ = [str(base / "MainCore")]
    core = types.ModuleType("MainCore.core")
    core.__path__ = [str(base / "MainCore" / "core")]
    importlib_mod = __import__("importlib")
    inspect_mod = __import__("inspect")
    pkgutil_mod = __import__("pkgutil")
    core.importlib = importlib_mod
    core.inspect = inspect_mod
    core.pkgutil = pkgutil_mod
    sys.modules.setdefault("MainCore", maincore)
    sys.modules.setdefault("MainCore.core", core)

    module_name = "MainCore.toolsState_testisolated"
    spec = importlib.util.spec_from_file_location(
        module_name,
        base / "MainCore" / "toolsState.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


TS = _load_tools_state()


def _fn(name, doc, params=None):
    params = params or {}
    namespace = {}
    args = ", ".join(f"{p}: {t}" for p, t in params.items())
    doc_literal = repr(doc)
    src = (
        f"def {name}({args}):\n"
        f"    {doc_literal}\n"
        f"    return None\n"
    )
    exec(src, namespace)
    return namespace[name]


def _info(fn):
    return {
        "function": fn,
        "description": fn.__doc__ or "",
        **TS._parse_tool_metadata(fn.__doc__ or ""),
    }


def test_metadata_parser_is_generic_and_exact():
    doc = """
    A tool.

    @required_each_turn
    @requires_state: alpha
    @requires_state: beta
    @provides_state: gamma
    @required_when_state: delta
    """
    meta = TS._parse_tool_metadata(doc)

    assert meta["required_each_turn"] is True
    assert meta["requires_state"] == ["alpha", "beta"]
    assert meta["provides_state"] == ["gamma"]
    assert meta["required_when_state"] == ["delta"]


def test_metadata_parser_ignores_unknown_directives():
    meta = TS._parse_tool_metadata("@totally_unknown: thing\n")
    assert meta == {
        "required_each_turn": False,
        "requires_state": [],
        "provides_state": [],
        "required_when_state": [],
    }


def test_required_and_eligible_sets_change_only_from_state():
    registry = TS.ToolRegistry()
    alpha = _fn("alpha", "@required_each_turn\n@provides_state: ready")
    beta = _fn("beta", "@requires_state: ready\n@required_when_state: ready")
    registry.tools = {
        "alpha": _info(alpha),
        "beta": _info(beta),
    }

    assert TS.get_required_tool_names(registry, {}) == {"alpha"}
    assert "beta" not in TS.get_eligible_tool_names(registry, {})

    active = {"ready": True}
    assert TS.get_required_tool_names(registry, active) == {"alpha", "beta"}
    assert "beta" in TS.get_eligible_tool_names(registry, active)


def test_required_when_state_uses_all_conditions():
    registry = TS.ToolRegistry()
    multi = _fn(
        "multi",
        "@required_when_state: a\n@required_when_state: b",
    )
    registry.tools = {"multi": _info(multi)}

    assert TS.get_required_tool_names(registry, {"a": True}) == set()
    assert TS.get_required_tool_names(
        registry,
        {"a": True, "b": True},
    ) == {"multi"}


def test_schema_required_arguments_come_from_signature():
    add = _fn(
        "add",
        "Add two values.",
        {"a": "int", "b": "int"},
    )
    schema = TS.build_tool_schema("add", _info(add))

    assert schema["function"]["parameters"]["required"] == ["a", "b"]
    assert schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"


def test_discovery_rejects_duplicate_tool_names(tmp_path, monkeypatch):
    package = tmp_path / "dynamic_tools"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text(
        "def collide():\n    \'\'\'A\'\'\'\n    return 1\n",
        encoding="utf-8",
    )
    (package / "b.py").write_text(
        "def collide():\n    \'\'\'B\'\'\'\n    return 2\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = TS.ToolRegistry()
    try:
        registry.discover("dynamic_tools")
    except ValueError as exc:
        assert "Duplicate tool name 'collide'" in str(exc)
    else:
        raise AssertionError("duplicate tool names must fail discovery")
