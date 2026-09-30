"""Tests for the agent manifest schema (eguard.manifest)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from eguard import ManifestError
from eguard.manifest import AgentProfile, Dimension

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "manifests"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.yaml"))

ALL_BUT_MARKET = frozenset(
    {Dimension.EQUITY, Dimension.LABOUR, Dimension.TRANSPARENCY, Dimension.SUSTAINABILITY}
)
PRICING_DEFAULTS = frozenset(
    {
        Dimension.EQUITY,
        Dimension.MARKET_STABILITY,
        Dimension.TRANSPARENCY,
        Dimension.SUSTAINABILITY,
    }
)


def minimal(**overrides):
    """Smallest valid hiring manifest, with optional field overrides."""
    data = {
        "name": "Test screener",
        "description": "Shortlists applicants.",
        "agent_type": "hiring",
        "decisions": ["shortlist candidates"],
        "affected_groups": ["job applicants"],
        "protected_attributes": ["gender"],
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------- examples


def test_three_example_manifests_exist():
    assert len(EXAMPLE_FILES) == 3


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_manifests_are_valid(path):
    assert AgentProfile.from_yaml(path).name


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_yaml_round_trip_preserves_profile(path, tmp_path):
    original = AgentProfile.from_yaml(path)
    copy_path = tmp_path / "copy.yaml"
    original.to_yaml(copy_path)
    assert AgentProfile.from_yaml(copy_path) == original


# ---------------------------------------------------------------------------- defaults


def test_minimal_manifest_uses_defaults():
    profile = AgentProfile.from_dict(minimal())
    assert profile.version == "0.1.0"
    assert profile.automated_tasks == []
    assert profile.ai_model is None
    assert profile.deployment is None


@pytest.mark.parametrize(
    ("agent_type", "expected"),
    [
        ("hiring", ALL_BUT_MARKET),
        ("pricing", PRICING_DEFAULTS),
        ("customer_support", ALL_BUT_MARKET),
    ],
)
def test_default_dimensions_per_agent_type(agent_type, expected):
    tasks = [{"description": "Answer queries"}] if agent_type == "customer_support" else []
    profile = AgentProfile.from_dict(minimal(agent_type=agent_type, automated_tasks=tasks))
    assert profile.applicable_dimensions == expected


def test_explicit_dimensions_override_defaults():
    profile = AgentProfile.from_dict(minimal(dimensions=["equity"]))
    assert profile.applicable_dimensions == frozenset({Dimension.EQUITY})


def test_whitespace_is_stripped():
    assert AgentProfile.from_dict(minimal(name="  Screener  ")).name == "Screener"


def test_profile_is_immutable():
    profile = AgentProfile.from_dict(minimal())
    with pytest.raises(ValidationError):
        profile.name = "changed"


# ---------------------------------------------------------------------- consistency rules


def test_equity_requires_protected_attributes():
    with pytest.raises(ManifestError, match="protected_attributes"):
        AgentProfile.from_dict(minimal(protected_attributes=[]))


def test_protected_attributes_optional_when_equity_not_applicable():
    profile = AgentProfile.from_dict(minimal(protected_attributes=[], dimensions=["labour"]))
    assert Dimension.EQUITY not in profile.applicable_dimensions


def test_customer_support_requires_automated_tasks():
    with pytest.raises(ManifestError, match="automated_tasks"):
        AgentProfile.from_dict(minimal(agent_type="customer_support"))


# ------------------------------------------------------------------------ invalid input


def test_missing_required_field_is_reported():
    data = minimal()
    del data["decisions"]
    with pytest.raises(ManifestError, match="decisions: Field required"):
        AgentProfile.from_dict(data)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"agent_type": "trading"}, "agent_type"),
        ({"decisions": []}, "decisions"),
        ({"name": "   "}, "name"),
        ({"protected_atributes": ["gender"]}, "protected_atributes"),
        ({"protected_attributes": ["gender", "gender"]}, "protected_attributes"),
        ({"dimensions": ["equity", "happiness"]}, "dimensions"),
        ({"dimensions": []}, "dimensions"),
        ({"schema_version": 2}, "schema_version"),
        ({"version": 1.0}, "version"),
        (
            {"automated_tasks": [{"description": "x", "automation_fraction": 1.5}]},
            "automation_fraction",
        ),
        ({"automated_tasks": [{"description": "x", "weight": 0}]}, "weight"),
        ({"automated_tasks": [{"description": "x", "onet_code": "43-4051"}]}, "onet_code"),
        ({"ai_model": {"provider": "x", "name": "y", "hosting": "cloud"}}, "hosting"),
        ({"deployment": {"decisions_per_day": -1}}, "decisions_per_day"),
        (
            {"automated_tasks": [{"description": "x", "reskilling_feasibility": 1.5}]},
            "reskilling_feasibility",
        ),
    ],
)
def test_invalid_fields_are_rejected_with_field_name(overrides, field):
    with pytest.raises(ManifestError, match=field):
        AgentProfile.from_dict(minimal(**overrides))


# ------------------------------------------------------------------------- YAML loading


def test_from_yaml_reports_syntax_errors(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="invalid YAML"):
        AgentProfile.from_yaml(path)


@pytest.mark.parametrize("content", ["- just\n- a list\n", ""])
def test_from_yaml_requires_a_mapping(tmp_path, content):
    path = tmp_path / "not_mapping.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestError, match="mapping"):
        AgentProfile.from_yaml(path)


def test_from_yaml_error_includes_file_name(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("name: Only a name\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="agent.yaml"):
        AgentProfile.from_yaml(path)


def test_from_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AgentProfile.from_yaml(tmp_path / "missing.yaml")
