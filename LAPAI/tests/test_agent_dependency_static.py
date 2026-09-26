from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _function_node(path, function_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    raise AssertionError(f"{function_name} not found in {path}")


def test_recallmemory_has_zero_arguments():
    node = _function_node(ROOT / "MainCore" / "core" / "tools" / "memorydecision.py", "recallmemory")
    assert len(node.args.args) == 0
    assert len(node.args.kwonlyargs) == 0


def test_recallmemory_declares_conditional_dependency():
    node = _function_node(ROOT / "MainCore" / "core" / "tools" / "memorydecision.py", "recallmemory")
    doc = ast.get_docstring(node) or ""
    assert "@requires_state: memory_query" in doc
    assert "@required_when_state: memory_query" in doc


def test_memory_commit_declares_required_provider_contract():
    node = _function_node(ROOT / "MainCore" / "core" / "tools" / "memory_commit.py", "memory_commit")
    doc = ast.get_docstring(node) or ""
    assert "@required_each_turn" in doc
    assert "@provides_state: memory_query" in doc


def test_agent_loop_has_generic_dependency_engine_and_no_memory_name_branch():
    text = (ROOT / "MainCore" / "statecore.py").read_text(encoding="utf-8")
    for forbidden in (
        'tool_name == "memory_commit"',
        'name == "memory_commit"',
        'tool_name == "recallmemory"',
    ):
        assert forbidden not in text

    assert "get_eligible_tool_names(" in text
    assert "get_required_tool_names(" in text
    assert "required_auto_" in text
    assert "inspect.signature(" in text
    assert 'request_kwargs["tool_choice"] = "required"' in text


def test_entrypoints_pass_current_user_explicitly():
    for relative in (
        "MainCore/runcorefp.py",
        "MainCore/runcoremain.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "cache.current_user_msg = user_msg" in text
        assert "current_user=user_msg" in text
