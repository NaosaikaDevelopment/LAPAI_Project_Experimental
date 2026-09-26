from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(
    os.environ.get("LAPAI_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()

if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))


def _import_required(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        pytest.fail(
            f"Unable to import required LAPAI module {module_name!r}: "
            f"{type(exc).__name__}: {exc}"
        )

    module_file = getattr(module, "__file__", None)
    if module_file:
        resolved = Path(module_file).resolve()
        if ROOT not in resolved.parents and resolved != ROOT:
            pytest.fail(
                f"Imported {module_name!r} from outside repository: {resolved}"
            )

    return module


@pytest.fixture
def lapai_root() -> Path:
    return ROOT


@pytest.fixture
def clean_workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def statecore():
    return _import_required("MainCore.statecore")


@pytest.fixture
def core_module():
    return _import_required("MainCore.core")


@pytest.fixture
def fp_core():
    return _import_required("MainCore.runcorefp")


@pytest.fixture
def isolated_core(statecore, clean_workspace, monkeypatch):
    """Use the real statecore module with only runtime paths isolated."""
    cache = statecore.cache
    root = clean_workspace

    data_dir = root / "data"
    chat_dir = data_dir / "chats"
    tools_dir = data_dir / "toolsMenu"
    learning_dir = data_dir / "Learning"
    settings_dir = root / "Settings"

    for path in (data_dir, chat_dir, tools_dir, learning_dir):
        path.mkdir(parents=True, exist_ok=True)

    paths = {
        "DATA_DIR": str(data_dir),
        "persnoal_file": str(data_dir / "PersonalData.txt"),
        "CHAT_DIR": str(chat_dir),
        "DB_FILE": str(data_dir / "Raw_Memory.db"),
        "TOOLSREADY": str(data_dir / "ToolsMenu.db"),
        "TOOLS_DIR": str(tools_dir),
        "Learning_Dir": str(learning_dir),
        "Learning_DB": str(learning_dir / "Learning.db"),
        "FAISS_INDEXM": str(data_dir / "Main_Memory.bin"),
        "FAISS_MAP": str(data_dir / "memory_map.json"),
        "QUESTIONS_FILE": str(data_dir / "questions.json"),
        "pepath": str(settings_dir / "PersonaAI"),
    }

    for attr, value in paths.items():
        monkeypatch.setattr(cache, attr, value, raising=False)


    monkeypatch.setattr(cache, "faiss_index", None, raising=False)
    monkeypatch.setattr(cache, "id_map", {}, raising=False)
    monkeypatch.setattr(cache, "session_id", {}, raising=False)
    monkeypatch.setattr(cache, "session_file", None, raising=False)
    monkeypatch.setattr(cache, "msgt", [], raising=False)
    monkeypatch.setattr(cache, "prompt", [], raising=False)
    monkeypatch.setattr(cache, "current_user_msg", "", raising=False)
    monkeypatch.setattr(cache, "agent_state", {}, raising=False)


    for attr in (
        "DB_FILE", "CHAT_DIR", "TOOLSREADY", "TOOLS_DIR", "Learning_Dir",
        "Learning_DB", "FAISS_INDEXM", "FAISS_MAP", "QUESTIONS_FILE",
        "persnoal_file", "pepath",
    ):
        monkeypatch.setattr(statecore, attr, getattr(cache, attr), raising=False)

    monkeypatch.setattr(statecore, "FAISS_INDEX", cache.FAISS_INDEXM, raising=False)
    monkeypatch.setattr(statecore, "FAISS_MAP", cache.FAISS_MAP, raising=False)

    return statecore
