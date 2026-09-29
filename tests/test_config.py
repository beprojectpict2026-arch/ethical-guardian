"""Tests for eguard.config."""

import pytest
from pydantic import ValidationError

from eguard.config import DEFAULT_WEIGHTS, GuardConfig
from eguard.manifest import Dimension


def test_default_weights_match_metric_definitions():
    assert GuardConfig().weights == {
        Dimension.EQUITY: 0.30,
        Dimension.MARKET_STABILITY: 0.25,
        Dimension.LABOUR: 0.20,
        Dimension.TRANSPARENCY: 0.15,
        Dimension.SUSTAINABILITY: 0.10,
    }


def test_default_weights_sum_to_one():
    assert sum(GuardConfig().weights.values()) == pytest.approx(1.0)


def test_partial_weight_override_keeps_other_defaults():
    config = GuardConfig(weights={"equity": 0.5})
    assert config.weights[Dimension.EQUITY] == 0.5
    assert config.weights[Dimension.LABOUR] == DEFAULT_WEIGHTS[Dimension.LABOUR]
    assert len(config.weights) == 5


@pytest.mark.parametrize("weights", [{"happiness": 0.1}, {"equity": 0}, {"labour": -0.2}])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValidationError):
        GuardConfig(weights=weights)


@pytest.mark.parametrize(
    "bounds",
    [(0.5, 0.25, 0.75), (0.0, 0.5, 0.75), (0.25, 0.5, 1.0), (0.25, 0.25, 0.75)],
)
def test_invalid_tier_boundaries_rejected(bounds):
    with pytest.raises(ValidationError, match="tier boundaries"):
        GuardConfig(tier_boundaries=bounds)


@pytest.mark.parametrize("field", ["delta_gini_ref", "cv_ref"])
def test_reference_constants_must_be_positive(field):
    with pytest.raises(ValidationError):
        GuardConfig(**{field: 0})


def test_weights_must_be_a_mapping():
    with pytest.raises(ValidationError):
        GuardConfig(weights=[0.3, 0.2])
