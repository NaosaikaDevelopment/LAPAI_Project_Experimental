from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(os.environ.get("LAPAI_REPO_ROOT", Path.cwd())).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORE_HELPERS = (
    "normalize_embedding",
    "calculate_context_budget",
    "init_db",
    "save_memory_state",
    "search_memory",
    "add_to_faiss",
)


def _candidate_modules() -> list[str]:
    configured = os.environ.get("LAPAI_TEST_MODULE")
    if configured:
        return [configured]
    return [
        "MainCore.core",
        "MainCore.statecore",
        "MainCore.runcorefp",
        "MainCore.runcoremain",
        "MainCore",
    ]


def import_first(*names: str) -> ModuleType:
    errors: list[str] = []
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    pytest.fail(
        "Could not import a required LAPAI module.\n"
        "Set LAPAI_REPO_ROOT if pytest was launched outside the repository.\n\n"
        + "\n".join(errors)
    )


@pytest.fixture(scope="session")
def lapai_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def statecore() -> ModuleType:
    return import_first("MainCore.statecore")


@pytest.fixture(scope="session")
def fp_core() -> ModuleType:
    return import_first("MainCore.runcorefp")


@pytest.fixture(scope="session")
def core_module() -> ModuleType:
    imported: list[ModuleType] = []
    errors: list[str] = []
    for name in _candidate_modules():
        try:
            imported.append(importlib.import_module(name))
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if not imported:
        pytest.fail(
            "Could not import any LAPAI core module.\n"
            "Set LAPAI_REPO_ROOT (repo folder) or LAPAI_TEST_MODULE (module name).\n\n"
            + "\n".join(errors)
        )
    scored = [(sum(hasattr(m, h) for h in CORE_HELPERS), m) for m in imported]
    score, selected = max(scored, key=lambda item: item[0])
    if score == 0:
        pytest.fail(
            "Imported LAPAI modules, but none exposes the required core helpers: "
            + ", ".join(CORE_HELPERS)
        )
    return selected


@pytest.fixture
def clean_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _get_cache(core_module: ModuleType):
    cache = getattr(core_module, "cache", None)
    if cache is not None:
        return cache
    try:
        return importlib.import_module("MainCore.core.state").cache
    except Exception as exc:
        pytest.fail(f"Could not locate LAPAI cache object: {exc}")


@pytest.fixture
def isolated_core(
    core_module: ModuleType,
    clean_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    """Isolate v1.5.1 persistence by patching the actual shared cache object."""
    cache = _get_cache(core_module)

    paths = {
        "DB_FILE": clean_workspace / "Raw_Memory.db",
        "CHAT_DIR": clean_workspace / "chats",
        "FAISS_INDEXM": clean_workspace / "Main_Memory.bin",
        "FAISS_MAP": clean_workspace / "memory_map.json",
        "Learning_DB": clean_workspace / "Learning.db",
        "Learning_Dir": clean_workspace / "Learning",
        "QUESTIONS_FILE": clean_workspace / "questions.json",
        "persnoal_file": clean_workspace / "PersonalData.txt",
    }

    monkeypatch.setattr(core_module, "cache", cache, raising=False)

    # `append_message` currently lives in MainCore.statecore.py rather than
    # MainCore.core. Expose it only on the fixture target so regression tests
    # can exercise the current persistence implementation without changing the
    # production public API.
    try:
        statecore_module = importlib.import_module("MainCore.statecore")
        if hasattr(statecore_module, "append_message"):
            monkeypatch.setattr(
                core_module,
                "append_message",
                statecore_module.append_message,
                raising=False,
            )
    except Exception:
        pass

    for name, path in paths.items():
        monkeypatch.setattr(cache, name, str(path), raising=False)

    Path(cache.CHAT_DIR).mkdir(parents=True, exist_ok=True)
    Path(cache.Learning_Dir).mkdir(parents=True, exist_ok=True)

    # Compatibility aliases only for tests that want to inspect the resolved files.
    monkeypatch.setattr(core_module, "DB_FILE", cache.DB_FILE, raising=False)
    monkeypatch.setattr(core_module, "CHAT_DIR", cache.CHAT_DIR, raising=False)
    monkeypatch.setattr(core_module, "FAISS_INDEX", cache.FAISS_INDEXM, raising=False)
    monkeypatch.setattr(core_module, "FAISS_MAP", cache.FAISS_MAP, raising=False)
    monkeypatch.setattr(core_module, "Learning_DB", cache.Learning_DB, raising=False)
    monkeypatch.setattr(core_module, "Learning_Dir", cache.Learning_Dir, raising=False)

    return core_module
