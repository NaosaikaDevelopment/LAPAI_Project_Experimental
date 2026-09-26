from pathlib import Path
import ast


def load_function_docstrings(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = ast.get_docstring(node) or ""
    return result


def test_dynamic_dependency_metadata_exists():
    tools_state = Path("MainCore/toolsState.py")
    memory_commit = Path("MainCore/core/tools/memory_commit.py")
    memorydecision = Path("MainCore/core/tools/memorydecision.py")

    assert "required_each_turn" in tools_state.read_text(encoding="utf-8")
    assert "required_when_state" in tools_state.read_text(encoding="utf-8")
    assert "requires_state" in tools_state.read_text(encoding="utf-8")

    commit_doc = load_function_docstrings(memory_commit)["memory_commit"]
    recall_doc = load_function_docstrings(memorydecision)["recallmemory"]

    assert "@required_each_turn" in commit_doc
    assert "@provides_state: memory_query" in commit_doc
    assert "@requires_state: memory_query" in recall_doc
    assert "@required_when_state: memory_query" in recall_doc


def test_recallmemory_has_no_query_argument():
    docs = load_function_docstrings(Path("MainCore/core/tools/memorydecision.py"))
    source = Path("MainCore/core/tools/memorydecision.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "recallmemory")
    assert len(fn.args.args) == 0
