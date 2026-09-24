from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

# README/runner: property-based tests belong to the heavier --stress layer.
pytestmark = pytest.mark.stress


def test_normalize_embedding_handles_nonzero_finite_vectors(core_module):
    np = pytest.importorskip("numpy")
    @given(st.lists(st.floats(min_value=-1_000, max_value=1_000, allow_nan=False, allow_infinity=False), min_size=1, max_size=32))
    def _property(values):
        vector = np.asarray(values, dtype=np.float32)
        if np.linalg.norm(vector) == 0:
            return
        out = np.asarray(core_module.normalize_embedding(vector))
        assert np.all(np.isfinite(out))
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-4)

    _property()


def test_context_budget_never_negative_under_generated_inputs(core_module):
    @given(
        model=st.integers(min_value=0, max_value=100_000),
        reserved=st.integers(min_value=0, max_value=100_000),
        used=st.integers(min_value=0, max_value=100_000),
    )
    def _property(model, reserved, used):
        result = core_module.calculate_context_budget(model, reserved, used)
        assert result >= 0
        assert result == max(0, model - reserved - used)

    _property()
