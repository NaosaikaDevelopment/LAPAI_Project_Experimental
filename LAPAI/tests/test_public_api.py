from __future__ import annotations

import inspect

import pytest


def test_runcorefp_exposes_initialize_core_and_main_function(fp_core):
    assert callable(getattr(fp_core, "initialize_core", None)), "initialize_core() is missing"
    assert callable(getattr(fp_core, "Main_Core_FP_Function", None)), "Main_Core_FP_Function() is missing"


def test_fast_function_has_a_small_public_contract(fp_core):
    fn = fp_core.Main_Core_FP_Function
    sig = inspect.signature(fn)
    required_positional = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    assert len(required_positional) <= 1, (
        "The modular public FP API should hide internal session/FAISS arguments; "
        f"required parameters are: {[p.name for p in required_positional]}"
    )


def test_statecore_exposes_shared_cache(statecore):
    cache = getattr(statecore, "cache", None)
    assert cache is not None, "MainCore.statecore.cache is missing"
    assert hasattr(cache, "conf"), "cache.conf is missing"


def test_statecore_cache_conf_is_mapping_like(statecore):
    conf = statecore.cache.conf
    assert hasattr(conf, "get"), "cache.conf should expose .get()"
