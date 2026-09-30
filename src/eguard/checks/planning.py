"""Planning-phase checks: evaluate an agent from its manifest, before code or data exist.

    labour displacement          job displacement score for each affected occupation
    documentation completeness   does the manifest hold what later checks and audits need?

The report notes also record an indicative regulatory classification.
Definitions: docs/metric_definitions.md sections 4 and 6a.
"""

from __future__ import annotations

from collections import defaultdict

from eguard.checks.thresholds import get_threshold, with_config
from eguard.config import GuardConfig
from eguard.manifest import AgentProfile, AgentType, AutomatedTask, Dimension
from eguard.metrics import job_displacement_score
from eguard.results import CheckResult, Phase, Report

PHASE = Phase.PLANNING
UNSPECIFIED = "unspecified occupation"

REGULATORY_NOTES = {
    AgentType.HIRING: (
        "Likely high-risk under the EU AI Act (Annex III: employment, including recruitment "
        "and selection). Some jurisdictions require bias audits of automated hiring tools, "
        "e.g. New York City Local Law 144."
    ),
    AgentType.CUSTOMER_SUPPORT: (
        "Likely subject to EU AI Act transparency obligations: people must be told they are "
        "interacting with an AI system."
    ),
    AgentType.PRICING: (
        "Not high-risk by default under the EU AI Act; competition and consumer-protection "
        "law apply to algorithmic pricing."
    ),
}
DISCLAIMER = "Indicative classification only; not legal advice."


@with_config
def check_planning(profile: AgentProfile, *, config: GuardConfig | None = None) -> Report:
    """Run planning-phase checks using only the agent manifest."""
    results: list[CheckResult] = []
    if Dimension.LABOUR in profile.applicable_dimensions:
        results.append(_job_displacement(profile.automated_tasks))
    results.append(_documentation(profile))

    notes: dict[str, object] = {
        "regulatory_note": REGULATORY_NOTES[profile.agent_type],
        "disclaimer": DISCLAIMER,
    }
    if profile.deployment is not None:
        notes["decisions_per_day"] = profile.deployment.decisions_per_day
    return Report.create(profile, PHASE, results, config=config, notes=notes)


def _job_displacement(tasks: list[AutomatedTask]) -> CheckResult:
    threshold = get_threshold("job_displacement")
    check_id = "labour.job_displacement"
    if not tasks:
        return CheckResult.from_value(
            check_id,
            Dimension.LABOUR,
            PHASE,
            0.0,
            threshold,
            risk=0.0,
            message="The manifest declares no automated tasks, so no work is displaced.",
        )

    groups: dict[str, list[AutomatedTask]] = defaultdict(list)
    for task in tasks:
        groups[task.occupation or UNSPECIFIED].append(task)

    scores: dict[str, float] = {}
    evidence: dict[str, dict[str, object]] = {}
    assumed: list[str] = []
    for occupation, group in groups.items():
        declared = [t.reskilling_feasibility for t in group if t.reskilling_feasibility is not None]
        reskilling = min(declared) if declared else 0.0
        if not declared:
            assumed.append(occupation)
        automation = [t.automation_fraction for t in group]
        weights = [t.weight for t in group]
        score = job_displacement_score(automation, reskilling, weights=weights)
        share = sum(a * w for a, w in zip(automation, weights, strict=True)) / sum(weights)
        scores[occupation] = score
        evidence[occupation] = {
            "job_displacement_score": round(score, 2),
            "automated_share": round(share, 4),
            "reskilling_feasibility": reskilling,
            "reskilling_assumed": not declared,
            "tasks": len(group),
        }

    worst = max(scores, key=lambda occupation: scores[occupation])
    value = scores[worst]
    message = f"Most affected occupation: {worst}, job displacement score {value:.1f} of 100."
    remediation = [
        f"Plan reskilling or redeployment for {worst} before deployment.",
        "Consider assisting rather than replacing workers on the highest-weight tasks.",
    ]
    if assumed:
        message += (
            f" Reskilling feasibility was not declared for {', '.join(assumed)}, so 0 was "
            "assumed and the score is an upper bound."
        )
        remediation.append("Declare reskilling feasibility for each automated task.")

    return CheckResult.from_value(
        check_id,
        Dimension.LABOUR,
        PHASE,
        value,
        threshold,
        risk=value / 100,
        message=message,
        evidence={"occupations": evidence},
        remediation=remediation,
    )


def _documentation(profile: AgentProfile) -> CheckResult:
    dimensions = profile.applicable_dimensions
    tasks = profile.automated_tasks
    items = {
        "owner": profile.owner is not None,
        "AI model": profile.ai_model is not None,
        "deployment scale": profile.deployment is not None,
    }
    if Dimension.EQUITY in dimensions:
        items["protected attributes"] = bool(profile.protected_attributes)
    if Dimension.LABOUR in dimensions and tasks:
        items["occupation for every automated task"] = all(t.occupation for t in tasks)
        items["O*NET code for every automated task"] = all(t.onet_code for t in tasks)
        items["reskilling feasibility for every automated task"] = all(
            t.reskilling_feasibility is not None for t in tasks
        )
    if Dimension.SUSTAINABILITY in dimensions:
        items["model size (parameters)"] = (
            profile.ai_model is not None and profile.ai_model.parameters_billion is not None
        )

    missing = [name for name, present in items.items() if not present]
    completeness = 1 - len(missing) / len(items)
    if missing:
        message = f"The manifest is missing: {', '.join(missing)}."
    else:
        message = "The manifest documents everything later checks need."
    return CheckResult.from_value(
        "transparency.documentation_completeness",
        Dimension.TRANSPARENCY,
        PHASE,
        completeness,
        get_threshold("documentation"),
        risk=None,
        message=message,
        evidence={"items": items},
        remediation=[f"Add the {name} to the manifest." for name in missing],
    )
