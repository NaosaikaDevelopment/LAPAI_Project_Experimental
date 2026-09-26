from __future__ import annotations

import ast
from pathlib import Path

import pytest


SKIP_DIRS = {
    ".git", ".hg", ".venv", "venv", "env", "LAPAI-env", "node_modules", "__pycache__",
    ".pytest_cache", ".hypothesis", ".tox", "build", "dist", "site-packages", "test-reports",
}


def _python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        # Only look at the path *inside* the repo. Checking the absolute path would skip
        # every file when the repo itself lives under a folder named e.g. "build" or "env".
        parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS or part.endswith(("-env", "_env")) for part in parts[:-1]):
            continue
        yield path


def _parse(path: Path):
    """Parse a file, or return None if it cannot be parsed (the compile test reports that)."""
    try:
        return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None


def _decorator_names(node) -> set[str]:
    names: set[str] = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def test_all_python_sources_compile(lapai_root: Path):
    py_files = list(_python_files(lapai_root))
    assert py_files, f"No Python source files found under {lapai_root}"
    problems: list[str] = []
    for path in py_files:
        try:
            # utf-8-sig tolerates the BOM that Windows editors sometimes add.
            source = path.read_text(encoding="utf-8-sig", errors="strict")
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            problems.append(f"{path}:{exc.lineno}: SyntaxError: {exc.msg}")
        except UnicodeDecodeError as exc:
            problems.append(f"{path}: not valid UTF-8 ({exc.reason})")
    assert not problems, "Files that do not compile:\n" + "\n".join(problems)


def test_no_duplicate_top_level_function_definitions(lapai_root: Path):
    problems: list[str] = []
    for path in _python_files(lapai_root):
        tree = _parse(path)
        if tree is None:
            continue
        seen: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # @overload / @x.register are legitimate re-definitions.
                if _decorator_names(node) & {"overload", "register"}:
                    continue
                if node.name in seen:
                    problems.append(f"{path}: duplicate top-level function {node.name} at line {node.lineno}")
                seen.add(node.name)
    assert not problems, "Duplicate definitions found:\n" + "\n".join(problems)


def test_no_mutable_default_arguments_in_lapai_source(lapai_root: Path):
    mutable = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
    problems: list[str] = []
    for path in _python_files(lapai_root):
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defaults = [*node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]]
            if any(isinstance(d, mutable) for d in defaults):
                problems.append(f"{path}:{node.lineno} {node.name} uses a mutable default")
    assert not problems, "Mutable default arguments found:\n" + "\n".join(problems)


def test_no_bare_except_in_lapai_source(lapai_root: Path):
    problems: list[str] = []
    for path in _python_files(lapai_root):
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                problems.append(f"{path}:{node.lineno} bare except")
    assert not problems, "Bare except blocks found:\n" + "\n".join(problems)


def test_repository_has_modular_core_contract(lapai_root: Path):
    main_core = lapai_root / "MainCore"
    if not main_core.exists():
        pytest.skip("MainCore is not present in this checkout; modular integration tests will report the missing API")
    assert (main_core / "__init__.py").exists(), "MainCore must be a Python package"
    assert (main_core / "runcorefp.py").exists(), "Public fast-response core is missing"
    assert (main_core / "statecore.py").exists(), "Shared state/cache module is missing"
