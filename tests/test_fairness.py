"""Tests for eguard.metrics.fairness, following docs/metric_definitions.md sections 2 and 3."""

import numpy as np
import pytest
from fairlearn.metrics import demographic_parity_ratio as fairlearn_dpr
from fairlearn.metrics import equalized_odds_difference as fairlearn_eod

from eguard import EguardWarning
from eguard.metrics import demographic_parity_ratio, equalized_odds_difference

# ---------------------------------------------------------------- Demographic parity ratio


def test_dpr_hand_computed():
    groups = ["A"] * 10 + ["B"] * 10
    y_pred = [1] * 5 + [0] * 5 + [1] * 2 + [0] * 8
    with pytest.warns(EguardWarning, match="smaller than 30"):
        result = demographic_parity_ratio(y_pred, groups)
    assert result == pytest.approx(0.4)


def test_dpr_equal_rates_is_one():
    groups = ["A"] * 40 + ["B"] * 40
    y_pred = ([1] * 10 + [0] * 30) * 2
    assert demographic_parity_ratio(y_pred, groups) == pytest.approx(1.0)


def test_dpr_three_groups_uses_min_over_max():
    groups = ["A"] * 40 + ["B"] * 40 + ["C"] * 40
    y_pred = [1] * 20 + [0] * 20 + [1] * 10 + [0] * 30 + [1] * 40
    assert demographic_parity_ratio(y_pred, groups) == pytest.approx(0.25)


def test_dpr_accepts_numeric_group_labels():
    groups = [0] * 40 + [1] * 40
    y_pred = [1] * 20 + [0] * 20 + [1] * 10 + [0] * 30
    assert demographic_parity_ratio(y_pred, groups) == pytest.approx(0.5)


def test_dpr_accepts_boolean_predictions():
    groups = ["A"] * 40 + ["B"] * 40
    y_pred = [True] * 20 + [False] * 20 + [True] * 10 + [False] * 30
    assert demographic_parity_ratio(y_pred, groups) == pytest.approx(0.5)


def test_dpr_no_selections_is_nan_with_warning():
    groups = ["A"] * 40 + ["B"] * 40
    with pytest.warns(EguardWarning, match="no group has any selections"):
        result = demographic_parity_ratio([0] * 80, groups)
    assert np.isnan(result)


@pytest.mark.parametrize(
    ("y_pred", "groups", "message"),
    [
        ([1, 0, 1], ["A", "B"], "length"),
        ([1, 2, 0, 1], ["A", "A", "B", "B"], "only 0 and 1"),
        ([1, 0, 1, 0], ["A", "A", "A", "A"], "at least 2"),
        ([1, 0, 1, 0], ["A", None, "B", "B"], "missing"),
        ([], [], "empty"),
    ],
)
def test_dpr_rejects_invalid_input(y_pred, groups, message):
    with pytest.raises(ValueError, match=message):
        demographic_parity_ratio(y_pred, groups)


# ------------------------------------------------------------- Equalized odds difference


def test_eod_hand_computed():
    groups = ["A"] * 20 + ["B"] * 20
    y_true = ([1] * 10 + [0] * 10) * 2
    y_pred = [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8 + [1] * 5 + [0] * 5 + [1] * 1 + [0] * 9
    with pytest.warns(EguardWarning, match="smaller than 30"):
        result = equalized_odds_difference(y_true, y_pred, groups)
    assert result == pytest.approx(0.3)


def test_eod_perfect_predictions_is_zero():
    groups = ["A"] * 40 + ["B"] * 40
    y_true = ([1] * 20 + [0] * 20) * 2
    assert equalized_odds_difference(y_true, y_true, groups) == pytest.approx(0.0)


def test_eod_takes_the_larger_of_tpr_and_fpr_gaps():
    # Group A: TPR 0.9, FPR 0.1. Group B: TPR 0.8, FPR 0.5. TPR gap 0.1, FPR gap 0.4.
    groups = ["A"] * 40 + ["B"] * 40
    y_true = ([1] * 20 + [0] * 20) * 2
    y_pred = [1] * 18 + [0] * 2 + [1] * 2 + [0] * 18 + [1] * 16 + [0] * 4 + [1] * 10 + [0] * 10
    assert equalized_odds_difference(y_true, y_pred, groups) == pytest.approx(0.4)


def test_eod_group_without_negatives_is_nan_with_warning():
    groups = ["A"] * 40 + ["B"] * 40
    y_true = [1] * 20 + [0] * 20 + [1] * 40
    y_pred = [1] * 40 + [0] * 40
    with pytest.warns(EguardWarning, match="group B has no actual negatives"):
        result = equalized_odds_difference(y_true, y_pred, groups)
    assert np.isnan(result)


def test_eod_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        equalized_odds_difference([1, 0, 1], [1, 0], ["A", "B", "A"])


# --------------------------------------------------------- Cross-check against Fairlearn


@pytest.mark.parametrize("seed", range(5))
def test_matches_fairlearn_on_random_data(seed):
    rng = np.random.default_rng(seed)
    n = 300
    groups = rng.choice(["A", "B", "C"], size=n)
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)

    expected_dpr = fairlearn_dpr(y_true, y_pred, sensitive_features=groups)
    expected_eod = fairlearn_eod(y_true, y_pred, sensitive_features=groups)

    assert demographic_parity_ratio(y_pred, groups) == pytest.approx(expected_dpr)
    assert equalized_odds_difference(y_true, y_pred, groups) == pytest.approx(expected_eod)


def test_rejects_two_dimensional_groups():
    with pytest.raises(ValueError, match="one-dimensional"):
        demographic_parity_ratio([1, 0, 1, 0], [["A", "B"], ["A", "B"]])


def test_rejects_mixed_type_group_labels():
    groups = np.array(["A", 1, "B", 2], dtype=object)
    with pytest.raises(ValueError, match="comparable type"):
        demographic_parity_ratio([1, 0, 1, 0], groups)


def test_eod_group_without_positives_is_nan_with_warning():
    groups = ["A"] * 40 + ["B"] * 40
    y_true = [1] * 20 + [0] * 20 + [0] * 40
    y_pred = [1] * 40 + [0] * 40
    with pytest.warns(EguardWarning, match="group B has no actual positives"):
        result = equalized_odds_difference(y_true, y_pred, groups)
    assert np.isnan(result)
