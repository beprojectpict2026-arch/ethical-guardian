"""Checks for each lifecycle phase.

Entry points: ``check(profile, phase, ...)`` for static evidence (manifest, data, prompt,
earlier reports), and ``evaluate(agent, profile, phase)`` for phases where the agent is run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

from eguard.checks.data import check_data
from eguard.checks.deployment import ReportSource, check_deployment
from eguard.checks.design import check_design
from eguard.checks.planning import check_planning
from eguard.checks.testing import EVALUATION_PHASES, evaluate
from eguard.config import GuardConfig
from eguard.manifest import AgentProfile
from eguard.results import Phase, Report

__all__ = [
    "check",
    "check_data",
    "check_deployment",
    "check_design",
    "check_planning",
    "evaluate",
]


def check(
    profile: AgentProfile,
    phase: Phase | str,
    *,
    data: pd.DataFrame | None = None,
    target: str | None = None,
    exclude: Sequence[str] = (),
    prompt: str | None = None,
    objective: str | None = None,
    reports: Iterable[ReportSource] | None = None,
    require: Sequence[Phase | str] | None = None,
    waivers: Mapping[str, str] | None = None,
    config: GuardConfig | None = None,
) -> Report:
    """Run the checks for one lifecycle phase using static evidence.

    Implemented: "planning" (manifest only), "data" (historical data), "design" (prompt
    and/or objective text) and "deployment" (earlier reports).
    """
    phase = Phase(phase)
    has_data = data is not None or target is not None
    has_text = prompt is not None or objective is not None
    has_gate = reports is not None or require is not None or waivers is not None

    if has_gate and phase is not Phase.DEPLOYMENT:
        raise ValueError("`reports`, `require` and `waivers` are only used in phase 'deployment'.")
    if phase is Phase.PLANNING:
        if has_data or has_text:
            raise ValueError(
                "phase 'planning' uses only the manifest; do not pass data, target, "
                "prompt or objective."
            )
        return check_planning(profile, config=config)
    if phase is Phase.DATA:
        if data is None or target is None:
            raise ValueError("phase 'data' requires both `data` and `target`.")
        return check_data(profile, data, target, exclude=exclude, config=config)
    if phase is Phase.DESIGN:
        if has_data:
            raise ValueError("phase 'design' scans text; do not pass `data` or `target`.")
        return check_design(profile, prompt=prompt, objective=objective, config=config)
    if phase is Phase.DEPLOYMENT:
        if reports is None:
            raise ValueError("phase 'deployment' requires `reports` from earlier phases.")
        if has_data or has_text:
            raise ValueError("phase 'deployment' uses earlier reports only.")
        return check_deployment(profile, reports, require=require, waivers=waivers, config=config)
    if phase in EVALUATION_PHASES:
        raise ValueError(f"phase '{phase.value}' runs the agent; use evaluate() instead.")
    raise NotImplementedError(f"checks for phase '{phase.value}' are not implemented yet.")
