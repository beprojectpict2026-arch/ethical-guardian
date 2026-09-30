"""Tests for the deployment gate (eguard.checks.deployment)."""

from pathlib import Path

import pytest

import eguard
from eguard import GuardConfig, Phase, Status
from eguard.checks import check, check_deployment
from eguard.manifest import Dimension
from eguard.scenarios.hiring import SkillBasedScreener

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "manifests"
GOOD_PROMPT = "Score applicants on skills and experience. Explain each decision."
FAST = {"seeds": (1000,), "n_applicants": 500}


@pytest.fixture(scope="module")
def profile():
    return eguard.AgentProfile.from_yaml(EXAMPLES / "hiring_agent.yaml")


@pytest.fixture(scope="module")
def planning(profile):
    return check(profile, "planning")


@pytest.fixture(scope="module")
def design(profile):
    return check(profile, "design", prompt=GOOD_PROMPT)


@pytest.fixture(scope="module")
def testing(profile):
    return eguard.evaluate(SkillBasedScreener(), profile, **FAST)


def urban_only(applicants):
    return (applicants["region"] == "urban") & (applicants["skill_technical"] >= 65)


@pytest.fixture(scope="module")
def failing_testing(profile):
    return eguard.evaluate(urban_only, profile, **FAST)


def result_of(report, check_id):
    return next(r for r in report.results if r.check_id == check_id)


def gate(profile, reports, **kwargs):
    return check(profile, "deployment", reports=reports, **kwargs)


# -------------------------------------------------------------------------- happy path


def test_complete_passing_evidence_releases(profile, planning, design, testing):
    report = gate(profile, [planning, design, testing])
    assert report.phase is Phase.DEPLOYMENT
    assert report.passed
    assert result_of(report, "transparency.gate_evidence").status is Status.PASS
    assert result_of(report, "transparency.gate_identity").status is Status.PASS
    assert len(report.metadata.notes["sources"]) == 3


def test_results_are_carried_forward_with_their_source(profile, planning, design, testing):
    report = gate(profile, [planning, design, testing])
    carried = result_of(report, "labour.job_displacement")
    assert carried.phase is Phase.DEPLOYMENT
    assert carried.message.startswith("[planning]")
    assert carried.evidence["source_phase"] == "planning"
    assert carried.value == pytest.approx(34.0)


def test_composite_covers_the_whole_lifecycle(profile, planning, design, testing):
    report = gate(profile, [planning, design, testing])
    risks = report.dimension_risks
    assert risks[Dimension.LABOUR] == pytest.approx(0.34)
    assert risks[Dimension.EQUITY] is not None
    assert risks[Dimension.TRANSPARENCY] is not None


def test_saved_reports_can_be_given_as_paths(profile, planning, design, testing, tmp_path):
    paths = []
    for report in (planning, design, testing):
        path = tmp_path / f"{report.phase.value}.json"
        report.to_json(path)
        paths.append(path)
    assert gate(profile, paths).passed


# ---------------------------------------------------------------------------- blocking


def test_failing_evidence_blocks(profile, planning, design, failing_testing):
    report = gate(profile, [planning, design, failing_testing])
    assert not report.passed
    with pytest.raises(eguard.EvaluationFailed, match="demographic_parity_region"):
        report.raise_for_status()


def test_missing_phase_blocks(profile, planning, testing):
    evidence = result_of(gate(profile, [planning, testing]), "transparency.gate_evidence")
    assert evidence.status is Status.FAIL
    assert evidence.value == pytest.approx(2 / 3)
    assert "design" in evidence.message


def test_reports_for_another_agent_block(profile, planning, design, testing):
    pricing = eguard.AgentProfile.from_yaml(EXAMPLES / "pricing_agent.yaml")
    other = check(pricing, "planning")
    identity = result_of(gate(profile, [other, design, testing]), "transparency.gate_identity")
    assert identity.status is Status.FAIL
    assert "Dynamic Pricing Engine" in identity.message


# ------------------------------------------------------------------------------ waivers


def test_waived_failures_warn_and_keep_their_risk(profile, planning, design, failing_testing):
    failing = [r.check_id for r in failing_testing.results if r.status is Status.FAIL]
    waivers = {check_id: "Accepted for a supervised pilot." for check_id in failing}
    blocked = gate(profile, [planning, design, failing_testing])
    released = gate(profile, [planning, design, failing_testing], waivers=waivers)
    assert released.passed
    assert released.composite_risk == pytest.approx(blocked.composite_risk)
    waived = result_of(released, failing[0])
    assert waived.status is Status.WARN
    assert "Waived: Accepted for a supervised pilot." in waived.message
    assert waived.evidence["waiver"]["original_status"] == "fail"
    assert released.metadata.notes["waivers"] == waivers


def test_waiver_must_name_a_failing_check(profile, planning, design, testing):
    with pytest.raises(ValueError, match="not failing"):
        gate(profile, [planning, design, testing], waivers={"equity.made_up": "Because."})


def test_waiver_needs_a_justification(profile, planning, design, failing_testing):
    with pytest.raises(ValueError, match="justification"):
        gate(
            profile,
            [planning, design, failing_testing],
            waivers={"equity.demographic_parity_region": "  "},
        )


# ---------------------------------------------------------------------------- selection


def test_latest_report_per_phase_is_used(profile, design, testing):
    old = check(profile, "planning")
    strict = GuardConfig(thresholds={"job_displacement": {"warn_at": 10, "fail_at": 30}})
    new = check(profile, "planning", config=strict)
    report = gate(profile, [new, old, design, testing])
    assert result_of(report, "labour.job_displacement").status is Status.FAIL
    assert len(report.metadata.notes["superseded"]) == 1


def test_most_advanced_evaluation_phase_is_used(profile, planning, design, testing):
    development = eguard.evaluate(urban_only, profile, "development")
    report = gate(profile, [planning, design, development, testing])
    assert report.passed
    superseded = [s["phase"] for s in report.metadata.notes["superseded"]]
    assert superseded == ["development"]


def test_development_alone_does_not_satisfy_testing(profile, planning, design):
    development = eguard.evaluate(SkillBasedScreener(), profile, "development")
    evidence = result_of(
        gate(profile, [planning, design, development]), "transparency.gate_evidence"
    )
    assert evidence.status is Status.FAIL
    assert "testing" in evidence.message


def test_custom_required_phases(profile, planning):
    report = gate(profile, [planning], require=["planning"])
    assert result_of(report, "transparency.gate_evidence").status is Status.PASS


def test_no_required_phases(profile, planning):
    evidence = result_of(gate(profile, [planning], require=[]), "transparency.gate_evidence")
    assert evidence.status is Status.PASS
    assert "No phases are required" in evidence.message


# --------------------------------------------------------------------------- validation


def test_needs_at_least_one_report(profile):
    with pytest.raises(ValueError, match="at least one"):
        check_deployment(profile, [])


def test_gate_report_cannot_be_an_input(profile, planning, design, testing):
    earlier_gate = gate(profile, [planning, design, testing])
    with pytest.raises(ValueError, match="cannot be inputs"):
        gate(profile, [earlier_gate])


def test_cannot_require_later_phases(profile, planning):
    with pytest.raises(ValueError, match="before deployment"):
        gate(profile, [planning], require=["monitoring"])


def test_deployment_requires_reports(profile):
    with pytest.raises(ValueError, match="requires `reports`"):
        check(profile, "deployment")


def test_deployment_rejects_other_evidence(profile, planning):
    with pytest.raises(ValueError, match="earlier reports only"):
        check(profile, "deployment", reports=[planning], prompt="Explain decisions.")


def test_gate_arguments_only_for_deployment(profile, planning):
    with pytest.raises(ValueError, match="only used in phase 'deployment'"):
        check(profile, "planning", reports=[planning])
