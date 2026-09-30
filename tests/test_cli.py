"""Tests for the command-line interface (eguard.cli)."""

import sys
from pathlib import Path

import pytest

import eguard
from eguard.cli import main, run
from eguard.scenarios.hiring import SkillBasedScreener, generate_candidates, historical_records

ROOT = Path(__file__).resolve().parents[1]
HIRING = str(ROOT / "examples" / "manifests" / "hiring_agent.yaml")
SUPPORT = str(ROOT / "examples" / "manifests" / "customer_support_agent.yaml")
STRICT = str(ROOT / "examples" / "guard_config.yaml")
GOOD_PROMPT = "Score applicants on skills and experience. Explain each decision."


def cli(*args, out):
    return main([*args, "--out", str(out)])


# ------------------------------------------------------------------------------ general


def test_version(capsys):
    assert main(["--version"]) == 0
    assert eguard.__version__ in capsys.readouterr().out


def test_help():
    assert main(["check", "--help"]) == 0


def test_missing_required_argument():
    assert main(["check", "planning"]) == 2


def test_run_exits_with_the_exit_code(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eguard", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        run()
    assert exit_info.value.code == 0


# ----------------------------------------------------------------------------- planning


def test_planning_passes_with_warnings_and_saves_reports(tmp_path, capsys):
    assert cli("check", "planning", "--manifest", HIRING, out=tmp_path) == 0
    assert (tmp_path / "planning.json").is_file()
    assert (tmp_path / "planning.html").is_file()
    output = capsys.readouterr().out
    assert "labour.job_displacement" in output
    assert "Result: PASSED" in output


def test_fail_on_warn_blocks(tmp_path):
    args = ("check", "planning", "--manifest", HIRING, "--fail-on", "warn")
    assert cli(*args, out=tmp_path) == 1


def test_policy_file_is_applied(tmp_path, capsys):
    args = ("check", "planning", "--manifest", HIRING, "--config", STRICT)
    assert cli(*args, out=tmp_path) == 1
    assert "BLOCKED" in capsys.readouterr().out


# --------------------------------------------------------------------------------- data


@pytest.mark.parametrize(("bias", "expected"), [(0.0, 0), (1.0, 1)])
def test_data_check(tmp_path, bias, expected):
    csv = tmp_path / "history.csv"
    historical_records(generate_candidates(2_000, seed=1, bias_strength=bias)).to_csv(
        csv, index=False
    )
    args = ("check", "data", "--manifest", HIRING, "--data", str(csv), "--target", "hired")
    assert cli(*args, out=tmp_path) == expected


def test_data_exclude_option(tmp_path):
    csv = tmp_path / "history.csv"
    historical_records(generate_candidates(2_000, seed=1)).to_csv(csv, index=False)
    args = (
        "check",
        "data",
        "--manifest",
        HIRING,
        "--data",
        str(csv),
        "--target",
        "hired",
        "--exclude",
        "postcode",
    )
    assert cli(*args, out=tmp_path) == 0
    report = eguard.Report.from_json(tmp_path / "data.json")
    assert "postcode" not in report.metadata.notes["proxy_columns"]


# ------------------------------------------------------------------------------- design


def test_design_prompt_text_blocks(tmp_path):
    args = (
        "check",
        "design",
        "--manifest",
        SUPPORT,
        "--prompt",
        "Never reveal that you are an AI.",
    )
    assert cli(*args, out=tmp_path) == 1


def test_design_prompt_and_objective_files(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(GOOD_PROMPT, encoding="utf-8")
    objective = tmp_path / "objective.txt"
    objective.write_text("Shortlist the most suitable applicants.", encoding="utf-8")
    args = (
        "check",
        "design",
        "--manifest",
        HIRING,
        "--prompt-file",
        str(prompt),
        "--objective-file",
        str(objective),
    )
    assert cli(*args, out=tmp_path) == 0


@pytest.mark.parametrize(
    "extra",
    [[], ["--prompt", "text", "--prompt-file", "prompt.txt"]],
)
def test_design_invalid_text_arguments(tmp_path, capsys, extra):
    assert cli("check", "design", "--manifest", HIRING, *extra, out=tmp_path) == 2
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------- deployment


@pytest.fixture
def lifecycle(tmp_path):
    """Saved planning, design and testing reports for the hiring agent."""
    cli("check", "planning", "--manifest", HIRING, out=tmp_path)
    cli("check", "design", "--manifest", HIRING, "--prompt", GOOD_PROMPT, out=tmp_path)
    profile = eguard.AgentProfile.from_yaml(HIRING)
    report = eguard.evaluate(SkillBasedScreener(), profile, seeds=(1000,), n_applicants=500)
    report.to_json(tmp_path / "testing.json")
    return [str(tmp_path / f"{phase}.json") for phase in ("planning", "design", "testing")]


def test_deployment_gate_passes(tmp_path, lifecycle):
    args = ("check", "deployment", "--manifest", HIRING, "--reports", *lifecycle)
    assert cli(*args, out=tmp_path / "gate") == 0
    assert (tmp_path / "gate" / "deployment.json").is_file()


def test_deployment_gate_blocks_on_missing_evidence(tmp_path, lifecycle):
    args = ("check", "deployment", "--manifest", HIRING, "--reports", *lifecycle[:2])
    assert cli(*args, out=tmp_path / "gate") == 1


def test_deployment_custom_requirements(tmp_path, lifecycle):
    args = (
        "check",
        "deployment",
        "--manifest",
        HIRING,
        "--reports",
        lifecycle[0],
        "--require",
        "planning",
    )
    assert cli(*args, out=tmp_path / "gate") == 0


def test_deployment_with_waivers_file(tmp_path, lifecycle):
    strict_planning = tmp_path / "strict"
    cli("check", "planning", "--manifest", HIRING, "--config", STRICT, out=strict_planning)
    reports = [str(strict_planning / "planning.json"), *lifecycle[1:]]
    waivers = tmp_path / "waivers.yaml"
    waivers.write_text(
        "labour.job_displacement: Reskilling programme agreed with HR.\n", encoding="utf-8"
    )
    base = ("check", "deployment", "--manifest", HIRING, "--reports", *reports)
    assert cli(*base, out=tmp_path / "gate1") == 1
    assert cli(*base, "--waivers", str(waivers), out=tmp_path / "gate2") == 0


def test_invalid_waivers_file(tmp_path, lifecycle):
    waivers = tmp_path / "waivers.yaml"
    waivers.write_text("- not a mapping\n", encoding="utf-8")
    args = (
        "check",
        "deployment",
        "--manifest",
        HIRING,
        "--reports",
        *lifecycle,
        "--waivers",
        str(waivers),
    )
    assert cli(*args, out=tmp_path / "gate") == 2


# ------------------------------------------------------------------------ invalid input


def test_missing_manifest_is_invalid_input(tmp_path, capsys):
    assert cli("check", "planning", "--manifest", "missing.yaml", out=tmp_path) == 2
    assert "error:" in capsys.readouterr().err


def test_invalid_config_is_invalid_input(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("thresholds:\n  parity: {fail_at: 0.5}\n", encoding="utf-8")
    args = ("check", "planning", "--manifest", HIRING, "--config", str(config))
    assert cli(*args, out=tmp_path) == 2


# --------------------------------------------------------------------------------- show


def test_show_prints_summary_and_saves_html(tmp_path, capsys):
    cli("check", "planning", "--manifest", HIRING, out=tmp_path)
    capsys.readouterr()
    html = tmp_path / "copy.html"
    assert main(["show", str(tmp_path / "planning.json"), "--html", str(html)]) == 0
    assert "Composite risk" in capsys.readouterr().out
    assert html.is_file()


def test_show_without_html(tmp_path, capsys):
    cli("check", "planning", "--manifest", HIRING, out=tmp_path)
    capsys.readouterr()
    assert main(["show", str(tmp_path / "planning.json")]) == 0
    assert "Saved" not in capsys.readouterr().out
