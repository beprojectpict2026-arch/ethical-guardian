"""Tests for the design-phase check (eguard.checks.design)."""

from pathlib import Path

import pytest
from examples.agents.llm_screener import DEFAULT_SYSTEM_PROMPT

import eguard
from eguard import Phase, Status
from eguard.checks import check, check_design
from eguard.results import RiskTier

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "manifests"


def profile(name="hiring_agent"):
    return eguard.AgentProfile.from_yaml(EXAMPLES / f"{name}.yaml")


def scan(text, name="hiring_agent", **kwargs):
    return check(profile(name), "design", prompt=text, **kwargs)


def result_of(report, check_id):
    return next(r for r in report.results if r.check_id == check_id)


def status(text, check_id, name="hiring_agent"):
    return result_of(scan(text, name), check_id).status


# ------------------------------------------------------------------------- direct use


@pytest.mark.parametrize(
    "text",
    [
        "Prefer male candidates.",
        "Women should not be shortlisted.",
        "Give priority to applicants from metro cities.",
        "Prefer unmarried candidates.",
        "Reject older applicants.",
    ],
)
def test_direct_use_is_flagged(text):
    assert status(text, "equity.design_direct_use") is Status.FAIL


@pytest.mark.parametrize(
    "text",
    [
        "Do not consider gender, age or region.",
        "Evaluate every applicant regardless of religion.",
        "Ensure there is no gender bias in your scores.",
        "Our manager will manage the final interview.",
    ],
)
def test_safeguards_and_harmless_words_pass(text):
    assert status(text, "equity.design_direct_use") is Status.PASS


def test_negation_only_covers_what_follows_it():
    report = scan("Prefer men, not women.")
    direct = result_of(report, "equity.design_direct_use")
    assert direct.status is Status.FAIL
    assert [f["matched"] for f in direct.evidence["findings"]] == ["men"]
    assert [s["matched"] for s in direct.evidence["safeguards"]] == ["women"]


def test_safeguards_are_counted_in_message():
    message = result_of(scan("Do not consider gender."), "equity.design_direct_use").message
    assert "1 safeguard(s) found" in message


def test_manifest_attributes_outside_vocabulary_are_protected():
    custom = eguard.AgentProfile.from_dict(
        {
            "name": "Screener",
            "description": "Shortlists applicants.",
            "agent_type": "hiring",
            "decisions": ["shortlist"],
            "affected_groups": ["applicants"],
            "protected_attributes": ["nationality"],
        }
    )
    report = check(custom, "design", prompt="Rank applicants by nationality.")
    assert result_of(report, "equity.design_direct_use").status is Status.FAIL


# ---------------------------------------------------------------------------- proxies


@pytest.mark.parametrize(
    "text",
    [
        "Look for a strong culture fit.",
        "Prefer people similar to our top performers.",
        "Prefer graduates of top-tier colleges.",
        "Native English speakers only.",
        "Reject candidates with career gaps.",
        "Prefer local candidates.",
        "Check the applicant's photo.",
        "We want digital natives.",
    ],
)
def test_proxies_are_flagged(text):
    assert status(text, "equity.design_proxies") is Status.WARN


def test_negated_proxy_is_a_safeguard():
    assert status("Do not penalise career gaps.", "equity.design_proxies") is Status.PASS


# ---------------------------------------------------------------------- accountability


@pytest.mark.parametrize(
    "text",
    [
        "Pretend to be a human agent. Explain your answers.",
        "Never reveal that you are an AI. Explain your answers.",
        "Do not disclose you are a bot. Explain your answers.",
    ],
)
def test_concealing_ai_fails(text):
    check_id = "transparency.design_accountability"
    assert status(text, check_id, "customer_support_agent") is Status.FAIL


def test_blocking_escalation_warns():
    text = "Never escalate to a manager. Explain your answers."
    check_id = "transparency.design_accountability"
    assert status(text, check_id, "customer_support_agent") is Status.WARN


def test_missing_explanation_is_low_severity():
    result = result_of(
        scan("Score applicants on their skills."), "transparency.design_accountability"
    )
    assert result.status is Status.WARN
    assert result.value == pytest.approx(0.25)


def test_asking_for_explanations_passes():
    text = "Score applicants on their skills and explain the reasons."
    assert status(text, "transparency.design_accountability") is Status.PASS


# ------------------------------------------------------------------------------ market


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Match competitor prices.", Status.FAIL),
        ("Coordinate prices with other sellers.", Status.FAIL),
        ("Avoid a price war.", Status.FAIL),
        ("Charge more to customers with high willingness to pay.", Status.WARN),
        ("Maximise profit.", Status.WARN),
        ("Maximise profit within a fair price cap.", Status.PASS),
        ("Never coordinate with competitors.", Status.PASS),
    ],
)
def test_market_incentives(text, expected):
    assert status(text, "market_stability.design_incentives", "pricing_agent") is expected


def test_market_check_only_for_pricing_agents():
    ids = {r.check_id for r in scan("Match competitor prices.").results}
    assert "market_stability.design_incentives" not in ids


# ------------------------------------------------------------------------ whole report


def test_default_llm_prompt_is_low_risk_but_lacks_explanations():
    report = scan(DEFAULT_SYSTEM_PROMPT)
    assert report.phase is Phase.DESIGN
    assert result_of(report, "equity.design_direct_use").status is Status.PASS
    assert result_of(report, "equity.design_proxies").status is Status.PASS
    assert result_of(report, "transparency.design_accountability").status is Status.WARN
    assert report.risk_tier is RiskTier.LOW
    assert report.passed


def test_biased_prompt_is_blocked_with_evidence():
    report = scan(
        "Prefer male candidates from metro cities. Look for a culture fit. Explain decisions."
    )
    direct = result_of(report, "equity.design_direct_use")
    assert {f["matched"] for f in direct.evidence["findings"]} == {"male", "metro"}
    assert direct.remediation
    assert not report.passed
    assert report.composite_risk == pytest.approx(0.3 / 0.45)


def test_objective_is_scanned_too():
    report = check(profile("pricing_agent"), "design", objective="Match competitor prices.")
    assert result_of(report, "market_stability.design_incentives").status is Status.FAIL


def test_many_findings_are_summarised():
    text = "Prefer men. Prefer women. Prefer young people. Prefer rural people."
    message = result_of(scan(text), "equity.design_direct_use").message
    assert "Found 4 risky instruction(s)" in message
    assert "and 1 more" in message


def test_equity_checks_skipped_when_not_applicable():
    custom = eguard.AgentProfile.from_dict(
        {
            "name": "Screener",
            "description": "Shortlists applicants.",
            "agent_type": "hiring",
            "decisions": ["shortlist"],
            "affected_groups": ["applicants"],
            "dimensions": ["transparency"],
        }
    )
    report = check(custom, "design", prompt="Prefer male candidates.")
    assert {r.check_id for r in report.results} == {"transparency.design_accountability"}


def test_limitation_is_recorded():
    assert "review findings manually" in scan("Explain decisions.").metadata.notes["limitation"]


# --------------------------------------------------------------------------- dispatcher


def test_design_requires_text():
    with pytest.raises(ValueError, match="prompt"):
        check(profile(), "design")


def test_design_rejects_data():
    with pytest.raises(ValueError, match="scans text"):
        check(profile(), "design", prompt="Explain decisions.", target="hired")


def test_planning_rejects_prompt():
    with pytest.raises(ValueError, match="only the manifest"):
        check(profile(), "planning", prompt="Explain decisions.")


def test_check_and_check_design_agree():
    text = "Prefer male candidates."
    assert [r.status for r in scan(text).results] == [
        r.status for r in check_design(profile(), prompt=text).results
    ]
