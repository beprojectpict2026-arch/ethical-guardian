"""Scoring primitives shared by results and configuration: Status, RiskTier and Threshold."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Status(StrEnum):
    """Outcome of one check. UNDEFINED never counts as a pass."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNDEFINED = "undefined"
    NOT_APPLICABLE = "not_applicable"


class RiskTier(StrEnum):
    """Band of the composite risk score."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class Threshold(BaseModel):
    """Warning and failure boundaries for one metric.

    A value strictly past a boundary triggers it; a value exactly on a boundary does not.
    This matches the definitions: DPR below 0.8 fails, DPR of exactly 0.8 does not.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    warn_at: float = Field(allow_inf_nan=False)
    fail_at: float = Field(allow_inf_nan=False)
    direction: Literal["higher_is_worse", "lower_is_worse"]

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.direction == "higher_is_worse" and self.warn_at > self.fail_at:
            raise ValueError("for higher_is_worse, warn_at must not exceed fail_at")
        if self.direction == "lower_is_worse" and self.warn_at < self.fail_at:
            raise ValueError("for lower_is_worse, warn_at must not be below fail_at")
        return self

    def evaluate(self, value: float | None) -> Status:
        """Status for a metric value. None or NaN gives UNDEFINED."""
        if value is None or math.isnan(value):
            return Status.UNDEFINED
        if self.direction == "higher_is_worse":
            if value > self.fail_at:
                return Status.FAIL
            if value > self.warn_at:
                return Status.WARN
            return Status.PASS
        if value < self.fail_at:
            return Status.FAIL
        if value < self.warn_at:
            return Status.WARN
        return Status.PASS

    def describe(self) -> str:
        """Plain-language description, e.g. 'warn below 0.9, fail below 0.8'."""
        word = "above" if self.direction == "higher_is_worse" else "below"
        return f"warn {word} {self.warn_at:g}, fail {word} {self.fail_at:g}"
