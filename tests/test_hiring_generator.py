"""Tests for the synthetic hiring candidate generator."""

import pandas as pd
import pytest
from pydantic import ValidationError

from eguard.metrics import demographic_parity_ratio
from eguard.scenarios.hiring import (
    FEATURE_COLUMNS,
    HIDDEN_COLUMNS,
    LABEL_COLUMN,
    PROTECTED_COLUMNS,
    CandidatePoolConfig,
    applicant_view,
    generate_candidates,
    historical_records,
)

N = 20_000


@pytest.fixture(scope="module")
def fair_pool():
    return generate_candidates(N, seed=1)


@pytest.fixture(scope="module")
def biased_pool():
    return generate_candidates(N, seed=1, bias_strength=1.0)


# ------------------------------------------------------------------------ structure


def test_columns(fair_pool):
    expected = ["candidate_id", *FEATURE_COLUMNS, *PROTECTED_COLUMNS, *HIDDEN_COLUMNS]
    assert list(fair_pool.columns) == [*expected, LABEL_COLUMN]
    assert len(fair_pool) == N


def test_value_ranges(fair_pool):
    for col in ("skill_technical", "skill_communication", "skill_domain"):
        assert fair_pool[col].between(0, 100).all()
    assert fair_pool["experience_years"].between(0, 30).all()
    assert set(fair_pool["gender"]) == {"female", "male"}
    assert set(fair_pool["region"]) == {"urban", "rural"}
    assert set(fair_pool["qualified"]) == {0, 1}
    assert set(fair_pool["hired"]) == {0, 1}


def test_candidate_ids_are_unique(fair_pool):
    assert fair_pool["candidate_id"].is_unique


def test_provenance_is_recorded(biased_pool):
    info = biased_pool.attrs["generator"]
    assert info["seed"] == 1
    assert info["bias_strength"] == 1.0
    assert info["config"]["hire_rate"] == 0.2


# --------------------------------------------------------------------- reproducibility


def test_same_seed_gives_identical_pool():
    pd.testing.assert_frame_equal(
        generate_candidates(500, seed=7), generate_candidates(500, seed=7)
    )


def test_different_seeds_differ():
    a = generate_candidates(500, seed=7)
    b = generate_candidates(500, seed=8)
    assert not a["skill_technical"].equals(b["skill_technical"])


def test_bias_changes_only_the_hired_label(fair_pool, biased_pool):
    unchanged = [c for c in fair_pool.columns if c != LABEL_COLUMN]
    pd.testing.assert_frame_equal(fair_pool[unchanged], biased_pool[unchanged])
    assert not fair_pool[LABEL_COLUMN].equals(biased_pool[LABEL_COLUMN])


def test_seed_is_required():
    with pytest.raises(TypeError):
        generate_candidates(100)


# ------------------------------------------------------------------------ population


def test_group_shares(fair_pool):
    assert (fair_pool["gender"] == "female").mean() == pytest.approx(0.4, abs=0.02)
    assert (fair_pool["region"] == "rural").mean() == pytest.approx(0.4, abs=0.02)


def test_exact_qualified_share_and_hire_rate(fair_pool):
    assert fair_pool["qualified"].sum() == round(0.3 * N)
    assert fair_pool["hired"].sum() == round(0.2 * N)


def test_qualification_is_independent_of_protected_attributes(fair_pool):
    for attr in PROTECTED_COLUMNS:
        means = fair_pool.groupby(attr)["qualification_score"].mean()
        assert abs(means.iloc[0] - means.iloc[1]) < 0.05
        assert demographic_parity_ratio(fair_pool["qualified"], fair_pool[attr]) > 0.9


def test_postcode_is_a_proxy_for_region(fair_pool):
    rural_code = fair_pool["postcode"].str.startswith("4125")
    agreement = (rural_code == (fair_pool["region"] == "rural")).mean()
    assert agreement == pytest.approx(0.9, abs=0.02)


def test_no_proxy_when_proxy_strength_is_half():
    df = generate_candidates(N, seed=1, config=CandidatePoolConfig(proxy_strength=0.5))
    rural_code = df["postcode"].str.startswith("4125")
    agreement = (rural_code == (df["region"] == "rural")).mean()
    assert agreement == pytest.approx(0.5, abs=0.02)


# ----------------------------------------------------------------------------- bias


def test_fair_history_has_near_parity(fair_pool):
    for attr in PROTECTED_COLUMNS:
        assert demographic_parity_ratio(fair_pool["hired"], fair_pool[attr]) > 0.9


def test_biased_history_fails_four_fifths_rule(biased_pool):
    for attr in PROTECTED_COLUMNS:
        assert demographic_parity_ratio(biased_pool["hired"], biased_pool[attr]) < 0.8


def test_parity_falls_as_bias_increases():
    config = CandidatePoolConfig(bias_targets={"gender": "female"})
    ratios = [
        demographic_parity_ratio(df["hired"], df["gender"])
        for df in (
            generate_candidates(N, seed=3, bias_strength=b, config=config)
            for b in (0.0, 0.5, 1.0, 1.5)
        )
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[0] > ratios[-1] + 0.3


def test_untargeted_attribute_is_unaffected():
    config = CandidatePoolConfig(bias_targets={"gender": "female"})
    df = generate_candidates(N, seed=1, bias_strength=1.0, config=config)
    assert demographic_parity_ratio(df["hired"], df["region"]) > 0.9


# ------------------------------------------------------------------------------ views


def test_historical_records_hide_ground_truth(fair_pool):
    records = historical_records(fair_pool)
    assert not set(HIDDEN_COLUMNS) & set(records.columns)
    assert LABEL_COLUMN in records.columns


def test_applicant_view_hides_ground_truth_and_outcome(fair_pool):
    view = applicant_view(fair_pool)
    assert not {*HIDDEN_COLUMNS, LABEL_COLUMN} & set(view.columns)
    assert set(PROTECTED_COLUMNS) <= set(view.columns)


# ------------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": 5}, "at least 10"),
        ({"bias_strength": -0.1}, r"\[0, 3\]"),
        ({"bias_strength": 3.5}, r"\[0, 3\]"),
    ],
)
def test_invalid_arguments_rejected(kwargs, message):
    args = {"n": 100, "seed": 0, **kwargs}
    with pytest.raises(ValueError, match=message):
        generate_candidates(**args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"education_probabilities": (0.5, 0.5, 0.5)},
        {"education_probabilities": (1.2, -0.1, -0.1)},
        {"bias_targets": {"gender": "other"}},
        {"bias_targets": {"age": "old"}},
        {"proxy_strength": 0.4},
        {"hire_rate": 1.0},
        {"unknown_setting": 1},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValidationError):
        CandidatePoolConfig(**kwargs)


def test_tiny_share_still_selects_one_candidate():
    df = generate_candidates(10, seed=0, config=CandidatePoolConfig(hire_rate=0.01))
    assert df["hired"].sum() == 1


def test_standardise_handles_constant_values():
    import numpy as np

    from eguard.scenarios.hiring.generator import _standardise

    result = _standardise(np.array([5.0, 5.0, 5.0]))
    assert result.tolist() == [0.0, 0.0, 0.0]
