"""Configuration: choices rather than definitions, such as weights and risk tiers.

Defaults follow docs/metric_definitions.md section 7. Every report stores the configuration it
was produced with, so results can always be reproduced.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from eguard.manifest import Dimension

DEFAULT_WEIGHTS: dict[Dimension, float] = {
    Dimension.EQUITY: 0.30,
    Dimension.MARKET_STABILITY: 0.25,
    Dimension.LABOUR: 0.20,
    Dimension.TRANSPARENCY: 0.15,
    Dimension.SUSTAINABILITY: 0.10,
}


class GuardConfig(BaseModel):
    """Weights, risk-tier boundaries and reference constants for the composite risk score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: dict[Dimension, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    tier_boundaries: tuple[float, float, float] = (0.25, 0.50, 0.75)
    delta_gini_ref: float = Field(default=0.10, gt=0.0)
    cv_ref: float = Field(default=20.0, gt=0.0)

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

    @field_validator("tier_boundaries")
    @classmethod
    def _increasing(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        low, moderate, high = value
        if not 0 < low < moderate < high < 1:
            raise ValueError("tier boundaries must satisfy 0 < low < moderate < high < 1")
        return value
