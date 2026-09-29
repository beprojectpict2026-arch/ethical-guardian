"""Tests for eguard.metrics.labour, following docs/metric_definitions.md section 4."""

import pytest

from eguard.metrics import job_displacement_score


@pytest.mark.parametrize(
    ("automation", "reskilling", "weights", "expected"),
    [
        ([1, 1, 0, 0], 0.5, None, 25.0),
        ([1, 0], 0.0, [3, 1], 75.0),
        ([0.5, 0.5], 1.0, None, 0.0),
    ],
)
def test_jds_hand_computed(automation, reskilling, weights, expected):
    result = job_displacement_score(automation, reskilling, weights=weights)
    assert result == pytest.approx(expected)


def test_jds_partial_automation():
    assert job_displacement_score([0.5] * 4, 0.0) == pytest.approx(50.0)


def test_jds_full_automation_no_reskilling_is_maximum():
    assert job_displacement_score([1, 1, 1], 0.0) == pytest.approx(100.0)


def test_jds_synopsis_formula_is_special_case():
    # Synopsis: (tasks automated / total tasks) * (1 - R) * 100, with 3 of 5 tasks automated.
    assert job_displacement_score([1, 1, 1, 0, 0], 0.2) == pytest.approx(3 / 5 * 0.8 * 100)


def test_jds_weights_must_be_keyword_argument():
    with pytest.raises(TypeError):
        job_displacement_score([1, 0], 0.5, [1, 1])


@pytest.mark.parametrize(
    ("automation", "reskilling", "weights", "message"),
    [
        ([], 0.5, None, "empty"),
        ([1.2], 0.5, None, r"in \[0, 1\]"),
        ([-0.1], 0.5, None, r"in \[0, 1\]"),
        ([1, 0], 0.5, [1, 0], "strictly positive"),
        ([1, 0], 0.5, [1], "same length"),
        ([1], -0.1, None, "reskilling_feasibility"),
        ([1], 1.1, None, "reskilling_feasibility"),
        ([1], float("nan"), None, "reskilling_feasibility"),
    ],
)
def test_jds_rejects_invalid_input(automation, reskilling, weights, message):
    with pytest.raises(ValueError, match=message):
        job_displacement_score(automation, reskilling, weights=weights)
