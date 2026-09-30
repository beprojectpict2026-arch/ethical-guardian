"""Configuration: choices rather than definitions, such as thresholds, weights and risk tiers.

Defaults follow docs/metric_definitions.md. A project can override any of them in a YAML file
(see examples/guard_config.yaml); anything not overridden keeps its default. Every report
stores the configuration it was produced with, so results can always be reproduced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from eguard.exceptions import ConfigError
from eguard.manifest import Dimension
from eguard.scoring import Threshold

DEFAULT_WEIGHTS: dict[Dimension, float] = {
    Dimension.EQUITY: 0.30,
    Dimension.MARKET_STABILITY: 0.25,
    Dimension.LABOUR: 0.20,
    Dimension.TRANSPARENCY: 0.15,
    Dimension.SUSTAINABILITY: 0.10,
}

DEFAULT_THRESHOLDS: dict[str, Threshold] = {
    # data phase
    "representation": Threshold(warn_at=0.5, fail_at=0.25, direction="lower_is_worse"),
    "label_parity": Threshold(warn_at=0.9, fail_at=0.8, direction="lower_is_worse"),
    "proxy": Threshold(warn_at=0.3, fail_at=0.9, direction="higher_is_worse"),
    # testing phase
    "demographic_parity": Threshold(warn_at=0.9, fail_at=0.8, direction="lower_is_worse"),
    "equalized_odds": Threshold(warn_at=0.05, fail_at=0.1, direction="higher_is_worse"),
    "counterfactual_flip": Threshold(warn_at=0.01, fail_at=0.05, direction="higher_is_worse"),
    # planning phase
    "job_displacement": Threshold(warn_at=20, fail_at=50, direction="higher_is_worse"),
    "documentation": Threshold(warn_at=1.0, fail_at=0.5, direction="lower_is_worse"),
    # design phase
    "design": Threshold(warn_at=0.0, fail_at=0.5, direction="higher_is_worse"),
    # deployment gate
    "gate_evidence": Threshold(warn_at=1.0, fail_at=1.0, direction="lower_is_worse"),
}


class GuardConfig(BaseModel):
    """Thresholds, dimension weights, risk-tier boundaries and reference constants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: dict[Dimension, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    thresholds: dict[str, Threshold] = Field(
        default_factory=dict,
        description="Overrides of DEFAULT_THRESHOLDS by name; others keep their defaults.",
    )
    tier_boundaries: tuple[float, float, float] = (0.25, 0.50, 0.75)
    delta_gini_ref: float = Field(default=0.10, gt=0.0)
    cv_ref: float = Field(default=20.0, gt=0.0)

    # ------------------------------------------------------------------------ validation

    @field_validator("weights", mode="before")
    @classmethod
    def _merge_with_defaults(cls, value: Any) -> Any:
        """Allow partial overrides: unspecified dimensions keep their default weight."""
        if isinstance(value, dict):
            return {**DEFAULT_WEIGHTS, **value}
        return value

    @field_validator("weights")
    @classmethod
    def _positive_weights(cls, value: dict[Dimension, float]) -> dict[Dimension, float]:
        non_positive = [dim.value for dim, weight in value.items() if weight <= 0]
        if non_positive:
            raise ValueError(f"weights must be positive: {', '.join(non_positive)}")
        return value

    @field_validator("thresholds", mode="before")
    @classmethod
    def _merge_thresholds(cls, value: Any) -> Any:
        """Validate names and fill unspecified fields of each override from its default."""
        if not isinstance(value, dict):
            return value
        unknown = sorted(set(value) - set(DEFAULT_THRESHOLDS))
        if unknown:
            raise ValueError(
                f"unknown threshold(s): {', '.join(unknown)}; "
                f"known: {', '.join(sorted(DEFAULT_THRESHOLDS))}"
            )
        merged: dict[str, Any] = {}
        for name, override in value.items():
            default = DEFAULT_THRESHOLDS[name]
            if isinstance(override, Threshold):
                override = override.model_dump()
            if not isinstance(override, dict):
                raise ValueError(f"threshold '{name}' must be a mapping of warn_at and/or fail_at")
            direction = override.get("direction", default.direction)
            if direction != default.direction:
                raise ValueError(
                    f"threshold '{name}' is '{default.direction}'; its direction cannot change"
                )
            merged[name] = {**default.model_dump(), **override}
        return merged

    @field_validator("tier_boundaries")
    @classmethod
    def _increasing(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        low, moderate, high = value
        if not 0 < low < moderate < high < 1:
            raise ValueError("tier boundaries must satisfy 0 < low < moderate < high < 1")
        return value

    # --------------------------------------------------------------------------- access

    def threshold(self, name: str) -> Threshold:
        """The threshold in force for `name`: the override if given, else the default."""
        if name not in DEFAULT_THRESHOLDS:
            raise KeyError(f"unknown threshold '{name}'")
        return self.thresholds.get(name, DEFAULT_THRESHOLDS[name])

    # ---------------------------------------------------------------- loading and saving

    @classmethod
    def from_yaml(cls, path: str | Path) -> GuardConfig:
        """Load a configuration file. An empty file gives the defaults."""
        path = Path(path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: configuration must be a YAML mapping.")
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            lines = []
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"]) or "(whole file)"
                lines.append(f"  - {location}: {error['msg']}")
            raise ConfigError(f"{path}: invalid configuration:\n" + "\n".join(lines)) from exc

    def to_yaml(self, path: str | Path) -> None:
        """Save the configuration as YAML."""
        text = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        Path(path).write_text(text, encoding="utf-8")
