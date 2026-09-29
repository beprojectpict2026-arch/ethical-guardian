"""Checks for each lifecycle phase. Entry point: ``check(profile, phase, ...)``."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from eguard.checks.data import check_data
from eguard.config import GuardConfig
from eguard.manifest import AgentProfile
from eguard.results import Phase, Report

__all__ = ["check", "check_data"]


def check(
    profile: AgentProfile,
    phase: Phase | str,
    *,
    data: pd.DataFrame | None = None,
    target: str | None = None,
    exclude: Sequence[str] = (),
    config: GuardConfig | None = None,
) -> Report:
    """Run the checks for one lifecycle phase using static evidence.

    Currently implemented: phase "data". Other phases are added in later stages.
    """
    phase = Phase(phase)
    if phase is Phase.DATA:
        if data is None or target is None:
            raise ValueError("phase 'data' requires both `data` and `target`.")
        return check_data(profile, data, target, exclude=exclude, config=config)
    raise NotImplementedError(f"checks for phase '{phase.value}' are not implemented yet.")
