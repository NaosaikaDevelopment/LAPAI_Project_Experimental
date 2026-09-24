from __future__ import annotations

import math
import time

import numpy as np
import pytest


def _fn(module, name):
    fn = getattr(module, name, None)
    if fn is None:
        pytest.fail(f"Required core helper missing: {name}")
    return fn


def test_normalize_embedding_is_unit_length(core_module):
    fn = _fn(core_module, "normalize_embedding")
    vector = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    out = np.asarray(fn(vector))
    assert out.shape == vector.shape
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-6)


def test_normalize_embedding_rejects_zero_vector(core_module):
    fn = _fn(core_module, "normalize_embedding")
    with pytest.raises(ValueError):
        fn(np.zeros(8, dtype=np.float32))


def test_context_budget_never_goes_negative(core_module):
    fn = _fn(core_module, "calculate_context_budget")
    assert fn(1000, 100, 200) == 700
    assert fn(100, 80, 30) == 0
    assert fn(0, 0, 1) == 0


def test_parse_yesno_contract(core_module):
    fn = _fn(core_module, "parse_yesno")
    assert fn("YES") == "Yes"
    assert fn("no, absolutely not") == "No"
    assert fn("unclear") == "No"


def test_keyword_extraction_is_case_insensitive(core_module):
    fn = _fn(core_module, "extract_keywords")
    assert fn("Hello WORLD 123") == ["hello", "world", "123"]


def test_importance_is_finite_and_positive(core_module):
    fn = _fn(core_module, "compute_importance")
    for text in ["", "hello", "x " * 1000]:
        value = float(fn(text))
        assert math.isfinite(value)
        assert value > 0


def test_recency_is_monotonic(core_module):
    fn = _fn(core_module, "compute_recency")
    now = time.time()
    recent = float(fn(now))
    old = float(fn(now - 86400))
    assert 0 < old < recent <= 1.000001


def test_build_memory_prompt_keeps_user_message_last(core_module):
    fn = _fn(core_module, "build_memory_prompt")
    result = fn(
        user_msg="What did I say about LAPAI?",
        summary="Project is a local AI runtime.",
        active_memory=[{"role": "user", "content": "I use Python."}],
        recalled_memory=[{"role": "user", "content": "LAPAI has memory."}],
        knowledge=[{"role": "system", "content": "FAISS is enabled."}],
    )
    assert isinstance(result, list)
    assert result[-1]["role"] == "user"
    assert result[-1]["content"] == "What did I say about LAPAI?"


def test_format_items_handles_unicode(core_module):
    fn = _fn(core_module, "format_items")
    result = fn([{"content": "日本語 ✅ äöü"}])
    assert "日本語" in result
    assert "✅" in result
