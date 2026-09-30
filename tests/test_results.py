"""Tests for eguard.results and composite scoring (docs/metric_definitions.md section 7)."""

import math
from datetime import UTC

import pytest
from pydantic import ValidationError

import eguard
from eguard import EvaluationFailed
from eguard.config import GuardConfig
from eguard.manifest import AgentProfile, Dimension
from eguard.results import CheckResult, Phase, Report, RiskTier, Status, Threshold

DPR_THRESHOLD = Threshold(warn_at=0.9, fail_at=0.8, direction="lower_is_worse")
EOD_THRESHOLD = Threshold(warn_at=0.05, fail_at=0.1, direction="higher_is_worse")
LENIENT = Threshold(warn_at=1000, fail_at=1000, direction="higher_is_worse")


def hiring_profile():
    return AgentProfile.from_dict(
        {
            "name": "Test screener",
            "description": "Shortlists applicants.",
            "agent_type": "hiring",
            "decisions": ["shortlist candidates"],
            "affected_groups": ["job applicants"],
            "protected_attributes": ["gender"],
        }
    )


def result(check_id, value, threshold, risk, phase=Phase.DATA):
    dimension = Dimension(check_id.split(".")[0])
    return CheckResult.from_value(
        check_id,
        dimension,
        phase,
        value,
        threshold,
        risk=risk,
        message=f"{check_id} computed.",
        remediation=["Investigate."],
    )


def worked_example_results():
    """The worked example in docs/metric_definitions.md section 7.5."""
    higher = lambda warn, fail: Threshold(warn_at=warn, fail_at=fail, direction="higher_is_worse")  # noqa: E731
    return [
        result("equity.demographic_parity_ratio", 0.6, DPR_THRESHOLD, 0.4),
        result("equity.equalized_odds_difference", 0.15, EOD_THRESHOLD, 0.15),
        result("equity.gini_delta", 0.05, higher(0.02, 0.05), 0.5),
        result("labour.job_displacement_score", 20.0, higher(20, 50), 0.2),
    ]


# -------------------------------------------------------------------------------- Threshold


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.95, Status.PASS),
        (0.9, Status.PASS),
        (0.85, Status.WARN),
        (0.8, Status.WARN),
        (0.79, Status.FAIL),
        (None, Status.UNDEFINED),
        (math.nan, Status.UNDEFINED),
    ],
)
def test_lower_is_worse_threshold(value, expected):
    assert DPR_THRESHOLD.evaluate(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.01, Status.PASS),
        (0.05, Status.PASS),
        (0.07, Status.WARN),
        (0.1, Status.WARN),
        (0.11, Status.FAIL),
        (None, Status.UNDEFINED),
    ],
)
def test_higher_is_worse_threshold(value, expected):
    assert EOD_THRESHOLD.evaluate(value) is expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warn_at": 0.2, "fail_at": 0.1, "direction": "higher_is_worse"},
        {"warn_at": 0.7, "fail_at": 0.8, "direction": "lower_is_worse"},
        {"warn_at": math.nan, "fail_at": 0.8, "direction": "lower_is_worse"},
        {"warn_at": 0.9, "fail_at": 0.8, "direction": "sideways"},
    ],
)
def test_invalid_thresholds_rejected(kwargs):
    with pytest.raises(ValidationError):
        Threshold(**kwargs)


def test_threshold_describe():
    assert DPR_THRESHOLD.describe() == "warn below 0.9, fail below 0.8"
    assert EOD_THRESHOLD.describe() == "warn above 0.05, fail above 0.1"


# ------------------------------------------------------------------------------ CheckResult


def test_from_value_failing_result_keeps_remediation():
    r = result("equity.demographic_parity_ratio", 0.6, DPR_THRESHOLD, 0.4)
    assert r.status is Status.FAIL
    assert r.value == 0.6
    assert r.risk == 0.4
    assert r.remediation == ["Investigate."]


def test_from_value_passing_result_drops_remediation():
    r = result("equity.demographic_parity_ratio", 0.95, DPR_THRESHOLD, 0.05)
    assert r.status is Status.PASS
    assert r.remediation == []


def test_from_value_nan_is_undefined_without_value_or_risk():
    r = result("equity.demographic_parity_ratio", math.nan, DPR_THRESHOLD, 0.4)
    assert r.status is Status.UNDEFINED
    assert r.value is None
    assert r.risk is None


def test_direct_nan_value_becomes_none():
    r = CheckResult(
        check_id="equity.demographic_parity_ratio",
        dimension=Dimension.EQUITY,
        phase=Phase.DATA,
        status=Status.UNDEFINED,
        value=math.nan,
        message="No selections.",
    )
    assert r.value is None


def test_not_applicable_result():
    r = CheckResult.not_applicable(
        "market_stability.coefficient_of_variation",
        Dimension.MARKET_STABILITY,
        Phase.TESTING,
        "Hiring agents do not set prices.",
    )
    assert r.status is Status.NOT_APPLICABLE
    assert r.value is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"check_id": "labour.dpr", "dimension": "equity", "status": "pass", "value": 1.0},
        {"check_id": "Equity DPR", "dimension": "equity", "status": "pass", "value": 1.0},
        {"check_id": "equity.dpr", "dimension": "equity", "status": "undefined", "value": 1.0},
        {"check_id": "equity.dpr", "dimension": "equity", "status": "pass"},
        {
            "check_id": "equity.dpr",
            "dimension": "equity",
            "status": "pass",
            "value": 1.0,
            "risk": 1.5,
        },
    ],
)
def test_inconsistent_results_rejected(kwargs):
    with pytest.raises(ValidationError):
        CheckResult(phase="data", message="x", **kwargs)


# ----------------------------------------------------------------------- Report: scoring


def test_worked_example_composite_is_0_38_moderate():
    report = Report.create(hiring_profile(), "data", worked_example_results())
    assert report.dimension_risks[Dimension.EQUITY] == pytest.approx(0.5)
    assert report.dimension_risks[Dimension.LABOUR] == pytest.approx(0.2)
    assert report.composite_risk == pytest.approx(0.38)
    assert report.risk_tier is RiskTier.MODERATE


def test_unmeasured_dimensions_are_listed_not_scored():
    report = Report.create(hiring_profile(), "data", worked_example_results())
    assert report.dimension_risks[Dimension.TRANSPARENCY] is None
    assert report.unevaluated_dimensions == [Dimension.SUSTAINABILITY, Dimension.TRANSPARENCY]


def test_empty_report_has_no_composite():
    report = Report.create(hiring_profile(), "planning", [])
    assert report.composite_risk is None
    assert report.risk_tier is None
    assert report.passed
    assert "Composite risk: not evaluated" in report.summary()


@pytest.mark.parametrize(
    ("risk", "tier"),
    [
        (0.1, RiskTier.LOW),
        (0.25, RiskTier.MODERATE),
        (0.6, RiskTier.HIGH),
        (0.75, RiskTier.CRITICAL),
    ],
)
def test_risk_tiers(risk, tier):
    results = [result("labour.job_displacement_score", 10.0, LENIENT, risk)]
    report = Report.create(hiring_profile(), "data", results)
    assert report.composite_risk == pytest.approx(risk)
    assert report.risk_tier is tier


def test_results_in_non_applicable_dimension_do_not_affect_score():
    results = [
        result("labour.job_displacement_score", 10.0, LENIENT, 0.2),
        result("market_stability.coefficient_of_variation", 50.0, LENIENT, 1.0),
    ]
    report = Report.create(hiring_profile(), "data", results)
    assert Dimension.MARKET_STABILITY not in report.dimension_risks
    assert report.composite_risk == pytest.approx(0.2)


def test_custom_weights_change_composite():
    config = GuardConfig(weights={"equity": 0.2, "labour": 0.2})
    report = Report.create(hiring_profile(), "data", worked_example_results(), config=config)
    assert report.composite_risk == pytest.approx((0.2 * 0.5 + 0.2 * 0.2) / 0.4)


# ----------------------------------------------------------------------- Report: outcomes


def test_status_counts():
    report = Report.create(hiring_profile(), "data", worked_example_results())
    counts = report.status_counts
    assert (counts[Status.PASS], counts[Status.WARN], counts[Status.FAIL]) == (1, 1, 2)


def test_failing_report_raises_evaluation_failed():
    report = Report.create(hiring_profile(), "data", worked_example_results())
    assert not report.passed
    with pytest.raises(EvaluationFailed, match="demographic_parity_ratio"):
        report.raise_for_status()


def test_evaluation_failed_is_an_assertion_error():
    assert issubclass(EvaluationFailed, AssertionError)


def test_warnings_block_only_when_requested():
    report = Report.create(
        hiring_profile(),
        "data",
        [result("equity.demographic_parity_ratio", 0.85, DPR_THRESHOLD, 0.15)],
    )
    assert report.passed
    report.raise_for_status()
    with pytest.raises(EvaluationFailed):
        report.raise_for_status(fail_on="warn")


def test_undefined_result_blocks():
    report = Report.create(
        hiring_profile(),
        "data",
        [result("equity.demographic_parity_ratio", math.nan, DPR_THRESHOLD, 0.4)],
    )
    assert not report.passed
    with pytest.raises(EvaluationFailed, match="UNDEFINED"):
        report.raise_for_status()


def test_invalid_fail_on_rejected():
    report = Report.create(hiring_profile(), "data", [])
    with pytest.raises(ValueError, match="fail_on"):
        report.raise_for_status(fail_on="maybe")


def test_results_from_another_phase_rejected():
    results = [result("labour.job_displacement_score", 10.0, LENIENT, 0.1, phase=Phase.TESTING)]
    with pytest.raises(ValidationError, match="another phase"):
        Report.create(hiring_profile(), "data", results)


def test_duplicate_check_ids_rejected():
    r = result("labour.job_displacement_score", 10.0, LENIENT, 0.1)
    with pytest.raises(ValidationError, match="duplicate"):
        Report.create(hiring_profile(), "data", [r, r])


# ------------------------------------------------------------------- Report: reproducibility


def test_metadata_records_version_seeds_and_time():
    report = Report.create(hiring_profile(), "data", [], seeds=[0, 1, 2])
    assert report.metadata.eguard_version == eguard.__version__
    assert report.metadata.seeds == [0, 1, 2]
    assert report.metadata.created_at.tzinfo is UTC


def test_report_stores_default_config():
    assert Report.create(hiring_profile(), "data", []).config == GuardConfig()


def test_json_round_trip(tmp_path):
    report = Report.create(hiring_profile(), "data", worked_example_results(), seeds=[7])
    path = tmp_path / "report.json"
    report.to_json(path)
    assert Report.from_json(path) == report


def test_summary_content():
    summary = Report.create(hiring_profile(), "data", worked_example_results()).summary()
    assert "Composite risk: 0.38 (moderate)" in summary
    assert "Results: 1 pass, 1 warn, 2 fail, 0 undefined, 0 not_applicable" in summary
    assert "[FAIL] equity.demographic_parity_ratio = 0.600" in summary
    assert "Not scored in this phase: sustainability, transparency" in summary
