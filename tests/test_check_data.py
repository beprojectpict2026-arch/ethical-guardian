"""Tests for the data-phase check (eguard.checks.data)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eguard
from eguard import EguardWarning, Phase, Status
from eguard.checks import check, check_data
from eguard.manifest import Dimension
from eguard.results import RiskTier
from eguard.scenarios.hiring import CandidatePoolConfig, generate_candidates, historical_records

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "manifests"
N = 20_000


@pytest.fixture(scope="module")
def profile():
    return eguard.AgentProfile.from_yaml(EXAMPLES / "hiring_agent.yaml")


@pytest.fixture(scope="module")
def fair_report(profile):
    data = historical_records(generate_candidates(N, seed=42))
    return check(profile, "data", data=data, target="hired")


@pytest.fixture(scope="module")
def biased_report(profile):
    data = historical_records(generate_candidates(N, seed=42, bias_strength=1.0))
    return check(profile, "data", data=data, target="hired")


def result_of(report, check_id):
    return next(r for r in report.results if r.check_id == check_id)


# --------------------------------------------------------------------------- fair data


def test_report_structure(fair_report):
    assert fair_report.phase is Phase.DATA
    assert len(fair_report.results) == 6
    assert {r.dimension for r in fair_report.results} == {Dimension.EQUITY}


@pytest.mark.parametrize("attribute", ["gender", "region"])
def test_fair_data_passes_label_parity_and_representation(fair_report, attribute):
    assert result_of(fair_report, f"equity.label_parity_{attribute}").status is Status.PASS
    assert result_of(fair_report, f"equity.representation_{attribute}").status is Status.PASS


def test_postcode_is_flagged_as_region_proxy(fair_report):
    result = result_of(fair_report, "equity.proxy_region")
    assert result.status is Status.WARN
    assert result.evidence["strongest_proxy"] == "postcode"
    assert result.value == pytest.approx(0.75, abs=0.03)


def test_no_strong_proxy_for_gender(fair_report):
    assert result_of(fair_report, "equity.proxy_gender").status is Status.PASS


def test_fair_data_passes_with_low_risk(fair_report):
    assert fair_report.passed
    assert fair_report.risk_tier is RiskTier.LOW


def test_only_equity_is_measured(fair_report):
    assert fair_report.unevaluated_dimensions == [
        Dimension.LABOUR,
        Dimension.SUSTAINABILITY,
        Dimension.TRANSPARENCY,
    ]


def test_generator_provenance_is_recorded(fair_report):
    assert fair_report.metadata.notes["data_generator"] == {"seed": 42, "bias_strength": 0.0}
    assert fair_report.metadata.notes["target"] == "hired"


def test_identifier_columns_are_not_proxy_candidates(fair_report):
    assert "candidate_id" not in fair_report.metadata.notes["proxy_columns"]
    assert "postcode" in fair_report.metadata.notes["proxy_columns"]


# ------------------------------------------------------------------------- biased data


@pytest.mark.parametrize("attribute", ["gender", "region"])
def test_biased_data_fails_label_parity(biased_report, attribute):
    result = result_of(biased_report, f"equity.label_parity_{attribute}")
    assert result.status is Status.FAIL
    assert result.remediation
    assert "hired" in result.message


def test_biased_data_is_high_risk_and_blocks(biased_report):
    assert not biased_report.passed
    assert biased_report.composite_risk > 0.5
    with pytest.raises(eguard.EvaluationFailed):
        biased_report.raise_for_status()


def test_proxy_risk_grows_with_label_disparity(fair_report, biased_report):
    fair_risk = result_of(fair_report, "equity.proxy_region").risk
    biased_risk = result_of(biased_report, "equity.proxy_region").risk
    assert biased_risk > 5 * fair_risk


# ------------------------------------------------------------------------------ options


def test_excluding_postcode_removes_the_proxy(profile):
    data = historical_records(generate_candidates(5_000, seed=1))
    report = check(profile, "data", data=data, target="hired", exclude=["postcode"])
    assert result_of(report, "equity.proxy_region").status is Status.PASS


def test_weak_proxy_population_passes(profile):
    pool = generate_candidates(5_000, seed=1, config=CandidatePoolConfig(proxy_strength=0.5))
    report = check(profile, "data", data=historical_records(pool), target="hired")
    assert result_of(report, "equity.proxy_region").status is Status.PASS


def test_text_identifier_column_is_ignored(profile):
    data = historical_records(generate_candidates(1_000, seed=1))
    data["application_ref"] = [f"APP-{i:05d}" for i in range(len(data))]
    report = check(profile, "data", data=data, target="hired")
    assert "application_ref" not in report.metadata.notes["proxy_columns"]


def test_attribute_names_are_slugged_into_check_ids():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "Age Group": rng.choice(["young", "old"], 200),
            "score": rng.normal(size=200),
            "hired": rng.integers(0, 2, 200),
        }
    )
    profile = eguard.AgentProfile.from_dict(
        {
            "name": "Screener",
            "description": "Shortlists applicants.",
            "agent_type": "hiring",
            "decisions": ["shortlist"],
            "affected_groups": ["applicants"],
            "protected_attributes": ["Age Group"],
        }
    )
    report = check(profile, "data", data=data, target="hired")
    ids = {r.check_id for r in report.results}
    assert "equity.label_parity_age_group" in ids


def test_no_proxy_columns_gives_not_applicable():
    data = pd.DataFrame({"gender": ["f", "m"] * 50, "hired": [1, 0, 0, 1] * 25})
    profile = eguard.AgentProfile.from_dict(
        {
            "name": "Screener",
            "description": "Shortlists applicants.",
            "agent_type": "hiring",
            "decisions": ["shortlist"],
            "affected_groups": ["applicants"],
            "protected_attributes": ["gender"],
        }
    )
    report = check(profile, "data", data=data, target="hired")
    assert result_of(report, "equity.proxy_gender").status is Status.NOT_APPLICABLE


def test_no_positive_labels_gives_undefined_parity(profile):
    data = historical_records(generate_candidates(2_000, seed=1))
    data["hired"] = 0
    with pytest.warns(EguardWarning, match="no group has any selections"):
        report = check(profile, "data", data=data, target="hired")
    parity = result_of(report, "equity.label_parity_gender")
    assert parity.status is Status.UNDEFINED
    assert "undefined" in parity.message
    proxy = result_of(report, "equity.proxy_region")
    assert proxy.risk == pytest.approx(proxy.value)
    assert not report.passed


# ---------------------------------------------------------------------------- validation


@pytest.fixture(scope="module")
def small_data():
    return historical_records(generate_candidates(1000, seed=0))


@pytest.mark.parametrize(
    ("mutate", "kwargs", "message"),
    [
        (lambda d: d, {"target": "outcome"}, "target column"),
        (lambda d: d, {"target": "gender"}, "protected attribute"),
        (lambda d: d.drop(columns=["region"]), {}, "region"),
        (lambda d: d, {"exclude": ["zipcode"]}, "exclude"),
        (lambda d: d.assign(hired=2), {}, "0 and 1"),
    ],
)
def test_invalid_input_rejected(profile, small_data, mutate, kwargs, message):
    args = {"target": "hired", **kwargs}
    with pytest.raises(ValueError, match=message):
        check_data(profile, mutate(small_data.copy()), **args)


def test_check_requires_data_and_target(profile, small_data):
    with pytest.raises(ValueError, match="requires"):
        check(profile, "data", data=small_data)


def test_unimplemented_phase(profile):
    with pytest.raises(NotImplementedError, match="planning"):
        check(profile, "planning")


def test_unknown_phase(profile):
    with pytest.raises(ValueError):
        check(profile, "brainstorming")


def test_check_and_check_data_agree(profile, small_data):
    via_check = check(profile, "data", data=small_data, target="hired")
    direct = check_data(profile, small_data, "hired")
    assert [r.status for r in via_check.results] == [r.status for r in direct.results]
