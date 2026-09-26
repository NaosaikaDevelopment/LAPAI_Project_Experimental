from __future__ import annotations


def test_core_fixture_uses_aggregate_core_package(core_module):
    for name in (
        "normalize_embedding", "calculate_context_budget", "parse_yesno",
        "extract_keywords", "compute_importance", "compute_recency",
        "build_memory_prompt", "format_items", "add_to_faiss",
    ):
        assert callable(getattr(core_module, name, None)), name


def test_isolated_core_uses_shared_real_statecore_module(statecore, isolated_core):
    assert isolated_core is statecore
    assert isolated_core.cache is statecore.cache


def test_fixture_layer_does_not_replace_production_modules(statecore, core_module):
    assert statecore.__name__ == "MainCore.statecore"
    assert core_module.__name__ == "MainCore.core"
