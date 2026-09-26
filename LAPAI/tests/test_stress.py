from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.mark.stress
def test_memory_state_survives_parallel_writes(isolated_core):
    core = isolated_core
    core.init_db()

    def write_one(i: int):
        core.save_memory_state(i, f"summary-{i}", f"time-{i}", i)
        return core.get_memory_state(i)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write_one, range(32)))

    assert len(results) == 32
    assert all(result["summary"].startswith("summary-") for result in results)
    assert {result["summarized_until_rowid"] for result in results} == set(range(32))


@pytest.mark.stress
@pytest.mark.parametrize("n", [0, 1, 2, 10, 100, 1000])
def test_context_budget_boundaries(core_module, n):
    value = core_module.calculate_context_budget(n, n // 2, n // 2 + 1)
    assert value == 0
