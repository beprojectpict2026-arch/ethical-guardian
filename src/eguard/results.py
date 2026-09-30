"""Result objects shared by every check: Phase, CheckResult and Report.

Status, RiskTier and Threshold live in eguard.scoring and are re-exported here.
Design: docs/api_design.md section 4. Composite scoring: docs/metric_definitions.md section 7.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eguard.config import GuardConfig
from eguard.exceptions import EvaluationFailed
from eguard.manifest import AgentProfile, Dimension
from eguard.scoring import RiskTier, Status, Threshold

__all__ = [
    "CheckResult",
    "Phase",
    "Report",
    "ReportMetadata",
    "RiskTier",
    "Status",
    "Threshold",
]


class Phase(StrEnum):
    """The eight phases of the agent development lifecycle."""

    PLANNING = "planning"
    DATA = "data"
    DESIGN = "design"
    DEVELOPMENT = "development"
    TRAINING = "training"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


_BLOCKING = frozenset({Status.FAIL, Status.UNDEFINED})


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckResult(_Frozen):
    """One metric evaluated against one threshold (docs/api_design.md section 4.3)."""

    check_id: str = Field(pattern=r"^[a-z_]+\.[a-z0-9_]+$")
    dimension: Dimension
    phase: Phase
    status: Status
    value: float | None = None
    threshold: Threshold | None = None
    risk: float | None = Field(default=None, ge=0.0, le=1.0)
    message: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    remediation: list[str] = Field(default_factory=list)

    @field_validator("value", mode="before")
    @classmethod
    def _nan_to_none(cls, value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        prefix = self.check_id.split(".", 1)[0]
        if prefix != self.dimension.value:
            raise ValueError(
                f"check_id '{self.check_id}' must start with its dimension "
                f"'{self.dimension.value}.'"
            )
        if self.status in (Status.UNDEFINED, Status.NOT_APPLICABLE):
            if self.value is not None or self.risk is not None:
                raise ValueError(f"a '{self.status.value}' result must not have a value or risk")
        elif self.value is None:
            raise ValueError(f"a '{self.status.value}' result needs a value")
        return self

    @classmethod
    def from_value(
        cls,
        check_id: str,
        dimension: Dimension,
        phase: Phase,
        value: float | None,
        threshold: Threshold,
        *,
        risk: float | None,
        message: str,
        evidence: dict[str, Any] | None = None,
        remediation: Iterable[str] = (),
    ) -> CheckResult:
        """Build a result whose status comes from the threshold. NaN becomes UNDEFINED.

        Remediation is kept only when the check does not pass.
        """
        clean = None if value is None or math.isnan(value) else float(value)
        status = threshold.evaluate(clean)
        return cls(
            check_id=check_id,
            dimension=dimension,
            phase=phase,
            status=status,
            value=clean,
            threshold=threshold,
            risk=None if clean is None else risk,
            message=message,
            evidence=evidence or {},
            remediation=[] if status is Status.PASS else list(remediation),
        )

    @classmethod
    def not_applicable(
        cls, check_id: str, dimension: Dimension, phase: Phase, reason: str
    ) -> CheckResult:
        """A check that does not apply to this agent, with the reason."""
        return cls(
            check_id=check_id,
            dimension=dimension,
            phase=phase,
            status=Status.NOT_APPLICABLE,
            message=reason,
        )


class ReportMetadata(_Frozen):
    """Everything needed to reproduce a report."""

    eguard_version: str
    created_at: datetime
    seeds: list[int] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)


class Report(_Frozen):
    """All results for one agent at one lifecycle phase (docs/api_design.md section 4.4)."""

    agent_name: str
    agent_version: str
    phase: Phase
    applicable_dimensions: list[Dimension]
    results: list[CheckResult]
    config: GuardConfig = Field(default_factory=GuardConfig)
    metadata: ReportMetadata

    @model_validator(mode="after")
    def _check_results(self) -> Self:
        wrong_phase = [r.check_id for r in self.results if r.phase is not self.phase]
        if wrong_phase:
            raise ValueError(f"results from another phase: {', '.join(wrong_phase)}")
        ids = [r.check_id for r in self.results]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate check_id: {', '.join(duplicates)}")
        return self

    @classmethod
    def create(
        cls,
        profile: AgentProfile,
        phase: Phase | str,
        results: Iterable[CheckResult],
        *,
        config: GuardConfig | None = None,
        seeds: Iterable[int] = (),
        notes: dict[str, Any] | None = None,
    ) -> Report:
        """Build a report for an agent, recording version, time, seeds and configuration."""
        import eguard  # imported here to avoid a circular import at module load

        return cls(
            agent_name=profile.name,
            agent_version=profile.version,
            phase=Phase(phase),
            applicable_dimensions=sorted(profile.applicable_dimensions),
            results=list(results),
            config=config or GuardConfig(),
            metadata=ReportMetadata(
                eguard_version=eguard.__version__,
                created_at=datetime.now(UTC),
                seeds=list(seeds),
                notes=notes or {},
            ),
        )

    # ------------------------------------------------------------------------- scoring

    @property
    def dimension_risks(self) -> dict[Dimension, float | None]:
        """Worst (maximum) risk per applicable dimension; None if nothing measured it."""
        risks: dict[Dimension, float | None] = {}
        for dimension in self.applicable_dimensions:
            values = [
                r.risk for r in self.results if r.dimension is dimension and r.risk is not None
            ]
            risks[dimension] = max(values) if values else None
        return risks

    @property
    def unevaluated_dimensions(self) -> list[Dimension]:
        """Applicable dimensions with no risk-scored result in this report."""
        return [dim for dim, risk in self.dimension_risks.items() if risk is None]

    @property
    def composite_risk(self) -> float | None:
        """Weighted mean risk, renormalised over the dimensions actually measured."""
        measured = {dim: risk for dim, risk in self.dimension_risks.items() if risk is not None}
        if not measured:
            return None
        total_weight = sum(self.config.weights[dim] for dim in measured)
        weighted = sum(self.config.weights[dim] * risk for dim, risk in measured.items())
        return weighted / total_weight

    @property
    def risk_tier(self) -> RiskTier | None:
        composite = self.composite_risk
        if composite is None:
            return None
        low, moderate, high = self.config.tier_boundaries
        if composite < low:
            return RiskTier.LOW
        if composite < moderate:
            return RiskTier.MODERATE
        if composite < high:
            return RiskTier.HIGH
        return RiskTier.CRITICAL

    # ------------------------------------------------------------------------- outcomes

    @property
    def status_counts(self) -> dict[Status, int]:
        return {status: sum(r.status is status for r in self.results) for status in Status}

    @property
    def passed(self) -> bool:
        """True only if no check failed or was undefined."""
        return not any(r.status in _BLOCKING for r in self.results)

    def raise_for_status(self, fail_on: Literal["fail", "warn"] = "fail") -> None:
        """Raise EvaluationFailed if any check is at or worse than `fail_on`."""
        if fail_on not in ("fail", "warn"):
            raise ValueError("`fail_on` must be 'fail' or 'warn'.")
        blocking = _BLOCKING | {Status.WARN} if fail_on == "warn" else _BLOCKING
        bad = [r for r in self.results if r.status in blocking]
        if bad:
            details = "\n".join(
                f"  [{r.status.value.upper()}] {r.check_id}: {r.message}" for r in bad
            )
            raise EvaluationFailed(
                f"{self.agent_name} has {len(bad)} blocking check(s) "
                f"in phase '{self.phase.value}':\n{details}"
            )

    def summary(self) -> str:
        """Short text summary for terminals and logs."""
        lines = [f"Report: {self.agent_name} v{self.agent_version} | phase: {self.phase.value}"]
        composite = self.composite_risk
        if composite is None or self.risk_tier is None:
            lines.append("Composite risk: not evaluated")
        else:
            lines.append(f"Composite risk: {composite:.2f} ({self.risk_tier.value})")
        counts = self.status_counts
        lines.append("Results: " + ", ".join(f"{counts[s]} {s.value}" for s in Status))
        for r in self.results:
            value = "n/a" if r.value is None else f"{r.value:.3f}"
            lines.append(f"  [{r.status.value.upper()}] {r.check_id} = {value}: {r.message}")
        if self.unevaluated_dimensions:
            names = ", ".join(dim.value for dim in self.unevaluated_dimensions)
            lines.append(f"Not scored in this phase: {names}")
        return "\n".join(lines)

    # ------------------------------------------------------------------------ saving

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> Report:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_html(self, path: str | Path) -> None:
        """Save the report as a self-contained HTML page, creating folders as needed."""
        from eguard.reporting import render_html  # avoids a circular import

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(self), encoding="utf-8")
