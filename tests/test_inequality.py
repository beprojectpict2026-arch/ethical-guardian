"""Tests for eguard.metrics.inequality, following docs/metric_definitions.md section 1."""

import numpy as np
import pandas as pd
import pytest

from eguard.metrics import gini, gini_delta


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([5, 5, 5, 5], 0.0),
        ([0, 0, 0, 10], 0.75),
        ([1, 2, 3, 4], 0.25),
        ([3, 1, 2], 2 / 9),
    ],
)
def test_gini_hand_computed(values, expected):
    assert gini(values) == pytest.approx(expected)


def test_gini_single_value_is_zero():
    assert gini([7]) == 0.0


def test_gini_all_zero_is_zero():
    assert gini([0, 0, 0]) == 0.0


@pytest.mark.parametrize("n", [2, 5, 100])
def test_gini_maximum_is_n_minus_one_over_n(n):
    values = [0] * (n - 1) + [1]
    assert gini(values) == pytest.approx((n - 1) / n)


def test_gini_is_permutation_invariant():
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 100, size=50)
    assert gini(values) == pytest.approx(gini(rng.permutation(values)))


def test_gini_is_scale_invariant():
    values = np.array([1.0, 4.0, 9.0, 16.0])
    assert gini(values) == pytest.approx(gini(values * 1000))


def test_gini_accepts_lists_arrays_and_series():
    values = [1, 2, 3, 4]
    assert gini(values) == gini(np.array(values)) == gini(pd.Series(values))


def test_gini_returns_python_float():
    assert isinstance(gini([1, 2, 3]), float)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "empty"),
        ([1, -2, 3], "non-negative"),
        ([1, np.nan], "NaN"),
        ([1, np.inf], "NaN or infinite"),
        ([[1, 2], [3, 4]], "one-dimensional"),
        (["a", "b"], "only numbers"),
    ],
)
def test_gini_rejects_invalid_input(values, message):
    with pytest.raises(ValueError, match=message):
        gini(values)


def test_gini_delta_hand_computed():
    assert gini_delta([5, 5, 5, 5], [0, 0, 0, 20]) == pytest.approx(0.75)


def test_gini_delta_is_negative_when_agent_reduces_inequality():
    assert gini_delta([0, 0, 0, 20], [5, 5, 5, 5]) == pytest.approx(-0.75)


def test_gini_delta_is_zero_for_identical_distributions():
    assert gini_delta([1, 2, 3], [3, 2, 1]) == pytest.approx(0.0)
