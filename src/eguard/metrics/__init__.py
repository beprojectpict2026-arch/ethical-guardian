"""Ethical and economic metrics.

Submodules:
    inequality: Gini coefficient and Gini delta.
    fairness:   Demographic parity ratio and equalized odds difference.
    labour:     Job displacement score.
    market:     Coefficient of variation (market volatility).

All metric functions are pure: they take arrays and return floats, with no side effects.
Definitions follow docs/metric_definitions.md.
"""

from eguard.metrics.fairness import demographic_parity_ratio, equalized_odds_difference
from eguard.metrics.inequality import gini, gini_delta
from eguard.metrics.labour import job_displacement_score
from eguard.metrics.market import coefficient_of_variation

__all__ = [
    "coefficient_of_variation",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "gini",
    "gini_delta",
    "job_displacement_score",
]
