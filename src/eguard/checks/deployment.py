"""Deployment gate: combine earlier lifecycle reports into one release decision.

The gate carries forward every result from the latest report of each earlier phase (for the
evaluation phases, only the most advanced one: testing over training over development),
re-labelled as deployment-phase results, and adds two checks of its own:
    evidence   is there a report for every required phase?
    identity   were all reports produced for this agent's name and version?

A failing result can be waived with a written justification. It then counts as a warning,
keeps its risk in the composite score, and the justification is recorded in the report.
Carried results keep the status each source report assigned under its own configuration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from eguard.checks.thresholds import get_threshold, with_config
from eguard.config import GuardConfig
from eguard.manifest import AgentProfile, Dimension
from eguard.results import CheckResult, Phase, Report, Status

PHASE = Phase.DEPLOYMENT
DEFAULT_REQUIRED = (Phase.PLANNING, Phase.DESIGN, Phase.TESTING)
EVALUATION_ORDER = (Phase.DEVELOPMENT, Phase.TRAINING, Phase.TESTING)
NOT_INPUTS = (Phase.DEPLOYMENT, Phase.MONITORING)

ReportSource = Report | str | Path


@with_config
def check_deployment(
    profile: AgentProfile,
    reports: Iterable[ReportSource],
    *,
    require: Sequence[Phase | str] | None = None,
    waivers: Mapping[str, str] | None = None,
    config: GuardConfig | None = None,
) -> Report:
    """Decide whether an agent may be released, from its earlier lifecycle reports.

    Args:
        profile: the agent's manifest.
        reports: earlier reports, as Report objects or paths to saved JSON reports.
        require: phases that must have a report (default: planning, design, testing).
        waivers: failing check IDs mapped to a written justification for accepting them.
        config: scoring configuration for the gate report; defaults to GuardConfig().
    """
    loaded = [_load(source) for source in reports]
    if not loaded:
        raise ValueError("the deployment gate needs at least one earlier report.")
    if any(report.phase in NOT_INPUTS for report in loaded):
        raise ValueError("deployment and monitoring reports cannot be inputs to the gate.")

    required = tuple(Phase(p) for p in (DEFAULT_REQUIRED if require is None else require))
    if any(phase in NOT_INPUTS for phase in required):
        raise ValueError("only phases before deployment can be required by the gate.")

    waivers = dict(waivers or {})
    empty = [check_id for check_id, reason in waivers.items() if not reason.strip()]
    if empty:
        raise ValueError(f"waivers need a written justification: {', '.join(empty)}")

    latest, older = _latest_per_phase(loaded)
    selected, superseded = _select_evaluation(latest)
    superseded = older + superseded

    carried, used = _carry_forward(selected, waivers)
    unused = sorted(set(waivers) - used)
    if unused:
        raise ValueError(
            "waivers can only accept failing checks; not failing in the given reports: "
            + ", ".join(unused)
        )

    results = [_evidence(required, latest), _identity(profile, loaded), *carried]
    notes: dict[str, Any] = {
        "required_phases": [phase.value for phase in required],
        "sources": [_describe(report) for report in selected],
        "superseded": [_describe(report) for report in superseded],
        "waivers": waivers,
    }
    return Report.create(profile, PHASE, results, config=config, notes=notes)


# --------------------------------------------------------------------------- selection


def _load(source: ReportSource) -> Report:
    return source if isinstance(source, Report) else Report.from_json(source)


def _latest_per_phase(reports: list[Report]) -> tuple[dict[Phase, Report], list[Report]]:
    """The most recent report of each phase, plus the older ones it supersedes."""
    latest: dict[Phase, Report] = {}
    older: list[Report] = []
    for report in sorted(reports, key=lambda r: r.metadata.created_at):
        if report.phase in latest:
            older.append(latest[report.phase])
        latest[report.phase] = report
    return latest, older


def _select_evaluation(latest: dict[Phase, Report]) -> tuple[list[Report], list[Report]]:
    """Keep only the most advanced evaluation phase; return reports in lifecycle order."""
    evaluation = [phase for phase in EVALUATION_ORDER if phase in latest]
    dropped = set(evaluation[:-1])
    selected = [latest[p] for p in Phase if p in latest and p not in dropped]
    superseded = [latest[p] for p in EVALUATION_ORDER if p in dropped]
    return selected, superseded


def _describe(report: Report) -> dict[str, Any]:
    return {
        "phase": report.phase.value,
        "created_at": report.metadata.created_at.isoformat(),
        "passed": report.passed,
        "composite_risk": report.composite_risk,
        "risk_tier": report.risk_tier.value if report.risk_tier else None,
    }


# ------------------------------------------------------------------------------ checks


def _evidence(required: tuple[Phase, ...], latest: dict[Phase, Report]) -> CheckResult:
    missing = [phase.value for phase in required if phase not in latest]
    share = 1.0 if not required else 1 - len(missing) / len(required)
    if missing:
        message = f"No report for required phase(s): {', '.join(missing)}."
    elif required:
        names = ", ".join(phase.value for phase in required)
        message = f"Reports are present for every required phase ({names})."
    else:
        message = "No phases are required."
    return CheckResult.from_value(
        "transparency.gate_evidence",
        Dimension.TRANSPARENCY,
        PHASE,
        share,
        get_threshold("gate_evidence"),
        risk=None,
        message=message,
        evidence={"required": [p.value for p in required], "missing": missing},
        remediation=[f"Run the {phase} checks and include the report." for phase in missing],
    )


def _identity(profile: AgentProfile, reports: list[Report]) -> CheckResult:
    mismatched = [
        f"{r.phase.value}: {r.agent_name} v{r.agent_version}"
        for r in reports
        if (r.agent_name, r.agent_version) != (profile.name, profile.version)
    ]
    share = 1 - len(mismatched) / len(reports)
    if mismatched:
        message = (
            f"Report(s) produced for a different agent or version than "
            f"{profile.name} v{profile.version}: {'; '.join(mismatched)}."
        )
    else:
        message = f"All reports are for {profile.name} v{profile.version}."
    return CheckResult.from_value(
        "transparency.gate_identity",
        Dimension.TRANSPARENCY,
        PHASE,
        share,
        get_threshold("gate_evidence"),
        risk=None,
        message=message,
        evidence={"mismatched": mismatched},
        remediation=["Re-run the checks for the exact agent version being released."],
    )


def _carry_forward(
    reports: list[Report], waivers: dict[str, str]
) -> tuple[list[CheckResult], set[str]]:
    carried: list[CheckResult] = []
    used: set[str] = set()
    for report in reports:
        source = report.phase.value
        for result in report.results:
            status = result.status
            message = f"[{source}] {result.message}"
            evidence = {**result.evidence, "source_phase": source}
            if result.check_id in waivers and result.status is Status.FAIL:
                justification = waivers[result.check_id]
                status = Status.WARN
                message += f" Waived: {justification}"
                evidence["waiver"] = {"justification": justification, "original_status": "fail"}
                used.add(result.check_id)
            carried.append(
                CheckResult(
                    check_id=result.check_id,
                    dimension=result.dimension,
                    phase=PHASE,
                    status=status,
                    value=result.value,
                    threshold=result.threshold,
                    risk=result.risk,
                    message=message,
                    evidence=evidence,
                    remediation=result.remediation,
                )
            )
    return carried, used
