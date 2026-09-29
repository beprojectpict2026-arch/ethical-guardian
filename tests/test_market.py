"""Tests for eguard.metrics.market, following docs/metric_definitions.md section 5."""

import numpy as np
import pytest

from eguard.metrics import coefficient_of_variation


@pytest.mark.parametrize(
    ("prices", "expected"),
    [
        ([10, 10, 10], 0.0),
        ([8, 12], 20.0),
    ],
)
def test_cv_hand_computed(prices, expected):
    assert coefficient_of_variation(prices) == pytest.approx(expected)


def test_cv_uses_population_standard_deviation():
    # Sample std (ddof=1) would give about 28.28 for [8, 12]; population std gives 20.
    assert coefficient_of_variation([8, 12]) == pytest.approx(20.0)


def test_cv_is_scale_invariant():
    prices = np.array([8.0, 12.0, 10.0, 11.0])
    assert coefficient_of_variation(prices) == pytest.approx(coefficient_of_variation(prices * 100))


@pytest.mark.parametrize(
    ("prices", "message"),
    [
        ([], "empty"),
        ([10], "at least 2"),
        ([10, 0], "strictly positive"),
        ([10, -5], "strictly positive"),
        ([10, np.nan], "NaN"),
    ],
)
def test_cv_rejects_invalid_input(prices, message):
    with pytest.raises(ValueError, match=message):
        coefficient_of_variation(prices)
