"""Tests for eguard.metrics.proxy, following docs/metric_definitions.md section 3b."""

import numpy as np
import pytest

from eguard import EguardWarning
from eguard.metrics import proxy_strength


@pytest.mark.parametrize(
    ("feature", "attribute", "expected"),
    [
        (["a", "a", "b", "b"], ["x", "x", "y", "y"], 1.0),
        (["a", "b", "a", "b"], ["x", "x", "y", "y"], 0.0),
        (["a", "a", "a", "b"], ["x", "x", "y", "y"], 0.5),
    ],
)
def test_hand_computed(feature, attribute, expected):
    with pytest.warns(EguardWarning, match="rows per level"):
        assert proxy_strength(feature, attribute) == pytest.approx(expected)


def test_independent_numeric_feature_is_near_zero():
    rng = np.random.default_rng(0)
    feature = rng.normal(size=10_000)
    attribute = rng.choice(["x", "y"], size=10_000)
    assert proxy_strength(feature, attribute) < 0.05


def test_numeric_feature_driven_by_attribute_is_strong():
    rng = np.random.default_rng(0)
    attribute = rng.choice(["x", "y"], size=10_000)
    feature = (attribute == "y") * 3.0 + rng.normal(size=10_000)
    assert proxy_strength(feature, attribute) > 0.8


def test_missing_values_form_their_own_category():
    feature = ["a", None] * 50 + ["b"] * 100
    attribute = ["x", "y"] * 50 + ["y"] * 100
    # 'a' -> always x, missing -> always y, 'b' -> always y: perfect prediction.
    assert proxy_strength(feature, attribute) == pytest.approx(1.0)


def test_result_is_python_float():
    rng = np.random.default_rng(1)
    value = proxy_strength(rng.normal(size=1000), rng.choice(["x", "y"], size=1000))
    assert isinstance(value, float)


@pytest.mark.parametrize(
    ("feature", "attribute", "kwargs", "message"),
    [
        ([], [], {}, "empty"),
        (["a", "b"], ["x", "y", "x"], {}, "length"),
        (["a", "b"], ["x", "x"], {}, "at least 2"),
        (["a", "b"], ["x", "y"], {"bins": 1}, "bins"),
    ],
)
def test_invalid_input(feature, attribute, kwargs, message):
    with pytest.raises(ValueError, match=message):
        proxy_strength(feature, attribute, **kwargs)
