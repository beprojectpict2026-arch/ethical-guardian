"""Tests for HTML report rendering (eguard.reporting)."""

import math
from pathlib import Path

import pytest

import eguard
from eguard import CheckResult, Phase, Report, Threshold
from eguard.manifest import Dimension
from eguard.reporting import render_html

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "manifests"
DPR = Threshold(warn_at=0.9, fail_at=0.8, direction="lower_is_worse")
LENIENT = Threshold(warn_at=1000, fail_at=1000, direction="higher_is_worse")


@pytest.fixture(scope="module")
def profile():
    return eguard.AgentProfile.from_yaml(EXAMPLES / "hiring_agent.yaml")


def make(check_id, value, threshold, risk, message=None, evidence=None):
    return CheckResult.from_value(
        check_id,
        Dimension(check_id.split(".")[0]),
        Phase.DATA,
        value,
        threshold,
        risk=risk,
        message=message or f"{check_id} computed.",
        evidence=evidence,
        remediation=["Investigate the cause."],
    )


@pytest.fixture
def report(profile):
    results = [
        make("equity.demographic_parity_ratio", 0.6, DPR, 0.4, evidence={"per_seed": [0.58, 0.62]}),
        make("equity.gini_delta", 0.01, LENIENT, 0.5),
        make("labour.job_displacement_score", 20.0, LENIENT, 0.2),
    ]
    return Report.create(profile, "data", results, seeds=[1, 2])


def test_contains_headline_information(report):
    html = render_html(report)
    assert html.startswith("<!doctype html>")
    assert "Resume Screening Assistant" in html
    assert "Seeds 1, 2" in html
    assert "Blocked" in html
    assert f"{report.composite_risk:.2f}" in html
    assert report.risk_tier.value in html


def test_every_check_is_listed_with_its_threshold(report):
    html = render_html(report)
    for result in report.results:
        assert result.check_id in html
    assert "warn below 0.9, fail below 0.8" in html


def test_status_badges(report):
    html = render_html(report)
    assert 'class="badge fail"' in html
    assert 'class="badge pass"' in html


def test_remediation_and_evidence_are_shown(report):
    html = render_html(report)
    assert "What to do" in html
    assert "Investigate the cause." in html
    assert "per_seed" in html


def test_unmeasured_dimensions_are_labelled(report):
    assert "not evaluated in this phase" in render_html(report)


def test_passing_empty_report(profile):
    html = render_html(Report.create(profile, "planning", []))
    assert "Passed" in html
    assert "Not evaluated" in html
    assert "No checks were run." in html


def test_undefined_values_show_as_na(profile):
    results = [make("equity.demographic_parity_ratio", math.nan, DPR, 0.4)]
    html = render_html(Report.create(profile, "data", results))
    assert "n/a" in html
    assert 'class="badge undefined"' in html


def test_not_applicable_results_render(profile):
    result = CheckResult.not_applicable(
        "market_stability.coefficient_of_variation",
        Dimension.MARKET_STABILITY,
        Phase.DATA,
        "Hiring agents do not set prices.",
    )
    html = render_html(Report.create(profile, "data", [result]))
    assert 'class="badge not_applicable"' in html
    assert "Hiring agents do not set prices." in html


def test_messages_are_escaped(profile):
    results = [
        make("equity.demographic_parity_ratio", 0.95, DPR, 0.05, message="<script>x()</script>")
    ]
    html = render_html(Report.create(profile, "data", results))
    assert "<script>x()" not in html
    assert "&lt;script&gt;" in html


def test_page_is_self_contained(report):
    html = render_html(report)
    assert "<script" not in html
    assert "<link" not in html
    assert "http" not in html


@pytest.mark.parametrize(
    ("risk", "colour"),
    [(0.1, "--low"), (0.3, "--moderate"), (0.6, "--high"), (0.9, "--critical")],
)
def test_risk_bars_use_tier_colours(profile, risk, colour):
    results = [make("labour.job_displacement_score", 10.0, LENIENT, risk)]
    html = render_html(Report.create(profile, "data", results))
    assert f"background: var({colour});" in html


def test_to_html_creates_folders_and_writes_file(report, tmp_path):
    path = tmp_path / "nested" / "report.html"
    report.to_html(path)
    assert path.read_text(encoding="utf-8") == render_html(report)


def test_reloaded_json_renders_identically(report, tmp_path):
    path = tmp_path / "report.json"
    report.to_json(path)
    assert render_html(Report.from_json(path)) == render_html(report)
