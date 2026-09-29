"""Tests for the testing-phase evaluation (eguard.evaluate)."""

from pathlib import Path

import pandas as pd
import pytest
from examples.agents.ml_screener import MLScreener

import eguard
from eguard import EguardWarning, Phase, Status
from eguard.checks.testing import _flip, _parity_note
from eguard.results import RiskTier
from eguard.scenarios.hiring import SkillBasedScreener, generate_candidates, historical_records

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "manifests"
FAST = {"seeds": (1000, 1001), "n_applicants": 1000}


@pytest.fixture(scope="module")
def profile():
    return eguard.AgentProfile.from_yaml(EXAMPLES / "hiring_agent.yaml")


@pytest.fixture(scope="module")
def history():
    return historical_records(generate_candidates(5_000, seed=1, bias_strength=1.0))


def run(agent, profile, **kwargs):
    return eguard.evaluate(agent, profile, **{**FAST, **kwargs})


def result_of(report, check_id):
    return next(r for r in report.results if r.check_id == check_id)


def urban_only(applicants):
    return (applicants["region"] == "urban") & (applicants["skill_technical"] >= 65)


# ------------------------------------------------------------------------------ outcomes


def test_reference_agent_passes(profile):
    report = run(SkillBasedScreener(), profile)
    assert report.phase is Phase.TESTING
    assert len(report.results) == 6
    assert report.passed
    assert report.risk_tier is RiskTier.LOW
    assert report.metadata.seeds == [1000, 1001]
    assert report.metadata.notes["agent"] == "reference-skill-based"
    assert report.metadata.notes["precision"]["agent"] > 0.9


def test_direct_discrimination_is_caught(profile):
    report = run(urban_only, profile)
    parity = result_of(report, "equity.demographic_parity_region")
    assert parity.status is Status.FAIL
    assert "uses it directly" in parity.message
    counterfactual = result_of(report, "equity.counterfactual_region")
    assert counterfactual.status is Status.FAIL
    assert "uses region directly" in counterfactual.message
    assert result_of(report, "equity.demographic_parity_gender").status is Status.PASS


def test_proxy_discrimination_is_caught_and_explained(profile, history):
    report = run(MLScreener().fit(history), profile)
    parity = result_of(report, "equity.demographic_parity_region")
    assert parity.status is Status.FAIL
    assert "proxy" in parity.message
    assert result_of(report, "equity.equalized_odds_region").status is Status.FAIL
    assert result_of(report, "equity.counterfactual_region").status is Status.PASS
    assert result_of(report, "equity.demographic_parity_gender").status is Status.PASS
    assert not report.passed


def test_aware_model_uses_attributes_directly(profile, history):
    report = run(MLScreener(use_protected=True).fit(history), profile)
    for attribute in ("gender", "region"):
        assert result_of(report, f"equity.counterfactual_{attribute}").status is Status.FAIL


def test_reference_and_per_seed_values_are_reported(profile):
    parity = result_of(run(urban_only, profile), "equity.demographic_parity_region")
    assert parity.evidence["reference"] > 0.85
    assert len(parity.evidence["per_seed"]) == 2
    assert "pooled over 2 seeds, 2,000 applicants" in parity.message


def test_function_agents_are_accepted(profile):
    report = run(lambda a: a["skill_technical"] >= 70, profile)
    assert report.metadata.notes["agent"] == "<lambda>"


def test_results_are_reproducible(profile):
    first = run(urban_only, profile)
    second = run(urban_only, profile)
    assert [r.value for r in first.results] == [r.value for r in second.results]


def test_agent_shortlisting_everyone(profile):
    report = run(lambda a: [True] * len(a), profile)
    assert result_of(report, "equity.demographic_parity_gender").value == pytest.approx(1.0)


def test_agent_shortlisting_nobody_is_undefined(profile):
    with pytest.warns(EguardWarning, match="no group has any selections"):
        report = run(lambda a: [False] * len(a), profile)
    parity = result_of(report, "equity.demographic_parity_gender")
    assert parity.status is Status.UNDEFINED
    assert "proxy" not in parity.message
    assert not report.passed


# --------------------------------------------------------------------------------- scale


def test_development_phase_is_quick_by_default(profile):
    report = eguard.evaluate(SkillBasedScreener(), profile, "development")
    assert report.metadata.seeds == [1000]
    assert report.metadata.notes["n_applicants"] == 500
    assert "pooled over 1 seed, 500 applicants" in report.results[0].message


# ---------------------------------------------------------------------- parity explanation


def test_note_blames_the_pool_when_reference_is_similar():
    note = _parity_note("gender", 0.85, 0.88, uses_directly=False)
    assert "reflects the applicant pool" in note


def test_note_blames_direct_use():
    note = _parity_note("region", 0.4, 0.95, uses_directly=True)
    assert "uses it directly" in note


def test_note_blames_a_proxy():
    note = _parity_note("region", 0.4, 0.95, uses_directly=False)
    assert "proxy" in note


def test_no_note_when_parity_is_undefined():
    assert _parity_note("region", float("nan"), 0.95, uses_directly=False) == ""


# ---------------------------------------------------------------------------- validation


def test_static_phase_is_rejected(profile):
    with pytest.raises(ValueError, match="check()"):
        eguard.evaluate(SkillBasedScreener(), profile, "data")


def test_empty_seeds_rejected(profile):
    with pytest.raises(ValueError, match="seeds"):
        eguard.evaluate(SkillBasedScreener(), profile, seeds=())


def test_unsupported_agent_type():
    pricing = eguard.AgentProfile.from_yaml(EXAMPLES / "pricing_agent.yaml")
    with pytest.raises(NotImplementedError, match="pricing"):
        eguard.evaluate(SkillBasedScreener(), pricing)


def test_unknown_scenario(profile):
    with pytest.raises(NotImplementedError, match="market"):
        eguard.evaluate(SkillBasedScreener(), profile, scenario="market")


def test_protected_attribute_not_in_scenario():
    profile = eguard.AgentProfile.from_dict(
        {
            "name": "Screener",
            "description": "Shortlists applicants.",
            "agent_type": "hiring",
            "decisions": ["shortlist"],
            "affected_groups": ["applicants"],
            "protected_attributes": ["gender", "language"],
        }
    )
    with pytest.raises(ValueError, match="hiring scenario"):
        eguard.evaluate(SkillBasedScreener(), profile)


# ------------------------------------------------------------------------------- helpers


def test_flip_swaps_two_groups():
    assert _flip(pd.Series(["a", "b", "a"])).tolist() == ["b", "a", "b"]


def test_flip_cycles_three_groups():
    assert _flip(pd.Series(["a", "b", "c"])).tolist() == ["b", "c", "a"]
