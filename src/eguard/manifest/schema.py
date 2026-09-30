"""Pydantic schema for the agent manifest (``AgentProfile``) and its YAML loader.

A company fills in the manifest at the planning phase. It describes what the agent decides,
what work it automates and who it affects, so the suite can choose which checks and metrics
apply. Task fields feed the job displacement score in docs/metric_definitions.md.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from eguard.exceptions import ManifestError

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OnetCode = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^\d{2}-\d{4}\.\d{2}$")]


class AgentType(StrEnum):
    """Agent types the suite currently supports."""

    HIRING = "hiring"
    PRICING = "pricing"
    CUSTOMER_SUPPORT = "customer_support"


class Dimension(StrEnum):
    """The five dimensions of the composite ethical-economic risk score."""

    EQUITY = "equity"
    MARKET_STABILITY = "market_stability"
    LABOUR = "labour"
    TRANSPARENCY = "transparency"
    SUSTAINABILITY = "sustainability"


DEFAULT_DIMENSIONS: dict[AgentType, frozenset[Dimension]] = {
    AgentType.HIRING: frozenset(
        {Dimension.EQUITY, Dimension.LABOUR, Dimension.TRANSPARENCY, Dimension.SUSTAINABILITY}
    ),
    AgentType.PRICING: frozenset(
        {
            Dimension.EQUITY,
            Dimension.MARKET_STABILITY,
            Dimension.TRANSPARENCY,
            Dimension.SUSTAINABILITY,
        }
    ),
    AgentType.CUSTOMER_SUPPORT: frozenset(
        {Dimension.EQUITY, Dimension.LABOUR, Dimension.TRANSPARENCY, Dimension.SUSTAINABILITY}
    ),
}


class _StrictModel(BaseModel):
    """Base model: unknown fields are errors and instances cannot be modified."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AutomatedTask(_StrictModel):
    """One task the agent takes over from people. Feeds the job displacement score."""

    description: NonEmptyStr
    occupation: NonEmptyStr | None = Field(
        default=None,
        description=(
            "Occupation that performs this task today, e.g. 'Customer Service Representatives'."
        ),
    )
    onet_code: OnetCode | None = Field(
        default=None, description="O*NET-SOC code of that occupation, e.g. '43-4051.00'."
    )
    automation_fraction: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Share of the task the agent performs (a_t)."
    )
    weight: float = Field(
        default=1.0, gt=0.0, description="Importance or time share of the task (u_t)."
    )
    reskilling_feasibility: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How easily workers in this occupation can move to other roles (R).",
    )


class ModelInfo(_StrictModel):
    """The AI model behind the agent. Used for the sustainability dimension."""

    provider: NonEmptyStr
    name: NonEmptyStr
    parameters_billion: float | None = Field(default=None, gt=0.0)
    hosting: Literal["cloud_api", "self_hosted"] | None = None


class DeploymentScale(_StrictModel):
    """Expected scale of use once deployed."""

    decisions_per_day: int = Field(ge=0)
    regions: list[NonEmptyStr] = Field(default_factory=list)


class AgentProfile(_StrictModel):
    """Agent manifest: a structured description of an agent, filled in at planning time."""

    schema_version: Literal[1] = 1
    name: NonEmptyStr
    version: NonEmptyStr = "0.1.0"
    description: NonEmptyStr
    owner: NonEmptyStr | None = None
    agent_type: AgentType
    decisions: list[NonEmptyStr] = Field(min_length=1)
    affected_groups: list[NonEmptyStr] = Field(min_length=1)
    protected_attributes: list[NonEmptyStr] = Field(default_factory=list)
    automated_tasks: list[AutomatedTask] = Field(default_factory=list)
    ai_model: ModelInfo | None = None
    deployment: DeploymentScale | None = None
    dimensions: Annotated[list[Dimension], Field(min_length=1)] | None = Field(
        default=None,
        description="Dimensions to evaluate. If omitted, defaults for the agent type are used.",
    )

    @property
    def applicable_dimensions(self) -> frozenset[Dimension]:
        """Dimensions evaluated for this agent: explicit override, else the type's defaults."""
        if self.dimensions is not None:
            return frozenset(self.dimensions)
        return DEFAULT_DIMENSIONS[self.agent_type]

    @field_validator("decisions", "affected_groups", "protected_attributes", "dimensions")
    @classmethod
    def _no_duplicates(cls, value: list[Any] | None) -> list[Any] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("must not contain duplicate entries")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if Dimension.EQUITY in self.applicable_dimensions and not self.protected_attributes:
            raise ValueError(
                "equity is an applicable dimension, so `protected_attributes` must list at "
                "least one attribute (e.g. gender, region, language)"
            )
        if self.agent_type is AgentType.CUSTOMER_SUPPORT and not self.automated_tasks:
            raise ValueError(
                "customer_support agents must list `automated_tasks`, since labour "
                "displacement is their main economic impact"
            )
        return self

    # ------------------------------------------------------------------ loading and saving

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentProfile:
        """Build a profile from a dictionary. Raises ManifestError with readable messages."""
        return cls._validate(data, source=None)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentProfile:
        """Load a profile from a YAML file. Raises ManifestError with the file path."""
        path = Path(path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ManifestError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ManifestError(
                f"{path}: manifest must be a YAML mapping of fields, got {type(data).__name__}."
            )
        return cls._validate(data, source=str(path))

    def to_yaml(self, path: str | Path) -> None:
        """Write the profile to a YAML file, omitting empty optional fields."""
        data = self.model_dump(mode="json", exclude_none=True)
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def _validate(cls, data: dict[str, Any], source: str | None) -> AgentProfile:
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            prefix = f"{source}: " if source else ""
            raise ManifestError(f"{prefix}invalid agent manifest:\n{_format_errors(exc)}") from exc


def _format_errors(exc: ValidationError) -> str:
    """Turn Pydantic errors into one readable line per problem, e.g. '- decisions: ...'."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(whole manifest)"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)
