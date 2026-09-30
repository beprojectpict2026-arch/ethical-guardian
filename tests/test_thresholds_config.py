"""Tests for configurable thresholds (GuardConfig) and how checks apply them."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import eguard
from eguard import ConfigError, GuardConfig, Status
from eguard.checks.thresholds import get_threshold
from eguard.config import DEFAULT_THRESHOLDS
from eguard.results import Threshold as ReexportedThreshold
from eguard.scenarios.hiring import SkillBasedScreener, generate_candidates, historical_records
from eguard.scoring import Threshold

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = ROOT / "examples" / "guard_config.yaml"


@pytest.fixture(scope="module")
def hiring():
    return eguard.AgentProfile.from_yaml(ROOT / "examples" / "manifests" / "hiring_agent.yaml")


def jds_status(profile, config=None):
    report = eguard.check(profile, "planning", config=config)
    return next(r for r in report.results if r.check_id == "labour.job_displacement").status


# ------------------------------------------------------------------------ configuration


def test_threshold_is_reexported_from_results():
    assert ReexportedThreshold is Threshold


def test_defaults_used_without_overrides():
    config = GuardConfig()
    for name, default in DEFAULT_THRESHOLDS.items():
        assert config.threshold(name) == default


def test_partial_override_keeps_other_fields():
    config = GuardConfig(thresholds={"job_displacement": {"fail_at": 40}})
    assert config.threshold("job_displacement") == Threshold(
        warn_at=20, fail_at=40, direction="higher_is_worse"
    )
    assert config.threshold("design") == DEFAULT_THRESHOLDS["design"]


def test_threshold_objects_are_accepted():
    override = Threshold(warn_at=0.95, fail_at=0.9, direction="lower_is_worse")
    config = GuardConfig(thresholds={"demographic_parity": override})
    assert config.threshold("demographic_parity") == override


@pytest.mark.parametrize(
    ("thresholds", "message"),
    [
        ({"parity": {"fail_at": 0.7}}, "unknown threshold"),
        ({"demographic_parity": {"direction": "higher_is_worse"}}, "direction cannot change"),
        ({"job_displacement": {"warn_at": 60}}, "must not exceed"),
        ({"job_displacement": {"warn": 10}}, "warn"),
        ({"job_displacement": 5}, "must be a mapping"),
    ],
)
def test_invalid_overrides_rejected(thresholds, message):
    with pytest.raises(ValidationError, match=message):
        GuardConfig(thresholds=thresholds)


def test_unknown_threshold_lookup():
    with pytest.raises(KeyError, match="unknown threshold"):
        GuardConfig().threshold("parity")


# ------------------------------------------------------------------------------- YAML


def test_example_config_loads():
    config = GuardConfig.from_yaml(EXAMPLE_CONFIG)
    assert config.threshold("job_displacement").fail_at == 30
    assert config.weights[eguard.manifest.Dimension.EQUITY] == 0.35


def test_empty_file_gives_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert GuardConfig.from_yaml(path) == GuardConfig()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("thresholds: [unclosed\n", "invalid YAML"),
        ("- a list\n", "must be a YAML mapping"),
        ("thresholds:\n  parity: {fail_at: 0.7}\n", "unknown threshold"),
    ],
)
def test_bad_files_raise_config_error(tmp_path, content, message):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        GuardConfig.from_yaml(path)


def test_config_error_names_the_file(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text("weights: {equity: -1}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="policy.yaml"):
        GuardConfig.from_yaml(path)


def test_yaml_round_trip(tmp_path):
    original = GuardConfig.from_yaml(EXAMPLE_CONFIG)
    path = tmp_path / "saved.yaml"
    original.to_yaml(path)
    assert GuardConfig.from_yaml(path) == original


# ------------------------------------------------------------------ checks honour config


def test_planning_check_uses_override(hiring):
    assert jds_status(hiring) is Status.WARN
    strict = GuardConfig(thresholds={"job_displacement": {"warn_at": 10, "fail_at": 30}})
    assert jds_status(hiring, strict) is Status.FAIL
    lenient = GuardConfig(thresholds={"job_displacement": {"warn_at": 40, "fail_at": 60}})
    assert jds_status(hiring, lenient) is Status.PASS


def test_override_does_not_leak_into_later_calls(hiring):
    strict = GuardConfig(thresholds={"job_displacement": {"warn_at": 10, "fail_at": 30}})
    jds_status(hiring, strict)
    assert jds_status(hiring) is Status.WARN
    assert get_threshold("job_displacement") == DEFAULT_THRESHOLDS["job_displacement"]


def test_data_check_uses_override(hiring):
    data = historical_records(generate_candidates(2_000, seed=1, bias_strength=1.0))
    lenient = GuardConfig(thresholds={"label_parity": {"warn_at": 0.1, "fail_at": 0.05}})
    report = eguard.check(hiring, "data", data=data, target="hired", config=lenient)
    parity = next(r for r in report.results if r.check_id == "equity.label_parity_gender")
    assert parity.status is Status.PASS
    assert parity.threshold == lenient.threshold("label_parity")


def test_design_check_uses_override(hiring):
    text = "Look for a strong culture fit. Explain each decision."
    lenient = GuardConfig(thresholds={"design": {"warn_at": 0.5, "fail_at": 1.0}})
    report = eguard.check(hiring, "design", prompt=text, config=lenient)
    proxies = next(r for r in report.results if r.check_id == "equity.design_proxies")
    assert proxies.status is Status.PASS


def test_evaluation_uses_override(hiring):
    strict = GuardConfig(thresholds={"demographic_parity": {"warn_at": 0.99, "fail_at": 0.98}})
    report = eguard.evaluate(
        SkillBasedScreener(), hiring, seeds=(1000,), n_applicants=500, config=strict
    )
    parity = next(r for r in report.results if r.check_id == "equity.demographic_parity_gender")
    assert parity.threshold == strict.threshold("demographic_parity")


def test_report_stores_config_overrides(hiring, tmp_path):
    strict = GuardConfig.from_yaml(EXAMPLE_CONFIG)
    report = eguard.check(hiring, "planning", config=strict)
    path = tmp_path / "report.json"
    report.to_json(path)
    assert eguard.Report.from_json(path).config == strict


def test_thresholds_must_be_a_mapping():
    with pytest.raises(ValidationError):
        GuardConfig(thresholds=["job_displacement"])
