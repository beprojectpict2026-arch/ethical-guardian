"""Tests for the planning-phase check (eguard.checks.planning)."""

from pathlib import Path

import pytest
import yaml

import eguard
from eguard import Phase, Status
from eguard.checks import check, check_planning
from eguard.results import RiskTier

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "manifests"


def load(name, **changes):
    data = yaml.safe_load((EXAMPLES / f"{name}.yaml").read_text(encoding="utf-8"))
    data.update(changes)
    return eguard.AgentProfile.from_dict(data)


def result_of(report, check_id):
    return next(r for r in report.results if r.check_id == check_id)


def task(occupation="Customer Service Representatives", automation=1.0, weight=1.0, r=None):
    data = {
        "description": "A task",
        "occupation": occupation,
        "onet_code": "43-4051.00" if occupation else None,
        "automation_fraction": automation,
        "weight": weight,
    }
    if r is not None:
        data["reskilling_feasibility"] = r
    return data


# ------------------------------------------------------------------------- examples


def test_hiring_agent_planning_report():
    report = check(load("hiring_agent"), "planning")
    assert report.phase is Phase.PLANNING
    jds = result_of(report, "labour.job_displacement")
    assert jds.value == pytest.approx(34.0)
    assert jds.status is Status.WARN
    assert jds.risk == pytest.approx(0.34)
    assert "Human Resources Specialists" in jds.message
    docs = result_of(report, "transparency.documentation_completeness")
    assert docs.value == pytest.approx(1.0)
    assert docs.status is Status.PASS
    assert report.composite_risk == pytest.approx(0.34)
    assert report.risk_tier is RiskTier.MODERATE
    assert report.passed


def test_customer_support_agent_planning_report():
    report = check(load("customer_support_agent"), "planning")
    jds = result_of(report, "labour.job_displacement")
    assert jds.value == pytest.approx(6.1 / 9 * 60)
    assert jds.status is Status.WARN


def test_pricing_agent_has_no_labour_check():
    report = check(load("pricing_agent"), "planning")
    assert "labour.job_displacement" not in {r.check_id for r in report.results}
    docs = result_of(report, "transparency.documentation_completeness")
    assert docs.value == pytest.approx(0.8)
    assert docs.status is Status.WARN
    assert "model size" in docs.message
    assert report.composite_risk is None


# ----------------------------------------------------------------------- job displacement


def test_missing_reskilling_gives_upper_bound():
    report = check(load("customer_support_agent", automated_tasks=[task()]), "planning")
    jds = result_of(report, "labour.job_displacement")
    assert jds.value == pytest.approx(100.0)
    assert jds.status is Status.FAIL
    assert "upper bound" in jds.message
    occupation = jds.evidence["occupations"]["Customer Service Representatives"]
    assert occupation["reskilling_assumed"] is True
    assert not report.passed


def test_most_affected_occupation_is_reported():
    tasks = [task("Occupation A", r=0.8), task("Occupation B", r=0.2)]
    jds = result_of(
        check(load("customer_support_agent", automated_tasks=tasks), "planning"),
        "labour.job_displacement",
    )
    assert jds.value == pytest.approx(80.0)
    assert "Occupation B" in jds.message
    assert set(jds.evidence["occupations"]) == {"Occupation A", "Occupation B"}


def test_conflicting_reskilling_values_use_the_lowest():
    tasks = [task(r=0.6), task(r=0.2)]
    jds = result_of(
        check(load("customer_support_agent", automated_tasks=tasks), "planning"),
        "labour.job_displacement",
    )
    assert jds.value == pytest.approx(80.0)


def test_tasks_without_occupation_are_grouped():
    report = check(load("customer_support_agent", automated_tasks=[task(None, r=0.5)]), "planning")
    jds = result_of(report, "labour.job_displacement")
    assert "unspecified occupation" in jds.evidence["occupations"]
    docs = result_of(report, "transparency.documentation_completeness")
    assert "occupation for every automated task" in docs.message
    assert "O*NET code" in docs.message


def test_no_automated_tasks_means_no_displacement():
    jds = result_of(
        check(load("hiring_agent", automated_tasks=[]), "planning"), "labour.job_displacement"
    )
    assert jds.value == 0.0
    assert jds.status is Status.PASS


def test_labour_check_skipped_when_not_applicable():
    report = check(load("hiring_agent", dimensions=["equity", "transparency"]), "planning")
    assert "labour.job_displacement" not in {r.check_id for r in report.results}


# ----------------------------------------------------------------------------- notes


@pytest.mark.parametrize(
    ("name", "phrase"),
    [
        ("hiring_agent", "Annex III"),
        ("customer_support_agent", "interacting with an AI system"),
        ("pricing_agent", "competition"),
    ],
)
def test_regulatory_notes(name, phrase):
    notes = check(load(name), "planning").metadata.notes
    assert phrase in notes["regulatory_note"]
    assert "not legal advice" in notes["disclaimer"]


def test_decisions_per_day_recorded():
    assert check(load("hiring_agent"), "planning").metadata.notes["decisions_per_day"] == 2000


def test_no_deployment_means_no_decisions_note():
    notes = check(load("pricing_agent", deployment=None), "planning").metadata.notes
    assert "decisions_per_day" not in notes


# ------------------------------------------------------------------------- dispatcher


def test_check_and_check_planning_agree():
    profile = load("hiring_agent")
    assert [r.status for r in check(profile, "planning").results] == [
        r.status for r in check_planning(profile).results
    ]


def test_planning_rejects_data_arguments():
    with pytest.raises(ValueError, match="only the manifest"):
        check(load("hiring_agent"), "planning", target="hired")


def test_evaluation_phases_point_to_evaluate():
    with pytest.raises(ValueError, match="evaluate()"):
        check(load("hiring_agent"), "testing")
