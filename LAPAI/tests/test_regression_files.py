from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_all_strict_regression_modules_exist():
    names = [
        "test_dynamic_tool_contracts.py",
        "test_toolrunner_strict.py",
        "test_memory_store_strict.py",
        "test_memory_tools_strict.py",
        "test_prompt_manager_strict.py",
        "test_agent_dependency_static.py",
        "test_agent_dependency_integration.py",
    ]
    for name in names:
        assert (ROOT / "tests" / name).is_file()


def test_generic_tests_do_not_hardcode_concrete_memory_tool_names():
    names = [
        "test_dynamic_tool_contracts.py",
        "test_toolrunner_strict.py",
        "test_agent_dependency_integration.py",
    ]
    for name in names:
        text = (ROOT / "tests" / name).read_text(encoding="utf-8")
        assert "memory_commit" not in text
        assert "recallmemory" not in text
