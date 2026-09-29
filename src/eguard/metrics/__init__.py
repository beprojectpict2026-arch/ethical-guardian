"""Ethical and economic metrics.

Submodules:
    inequality: Gini coefficient and Gini delta.
    fairness:   Demographic parity ratio and equalized odds difference.
    labour:     Job displacement score.
    market:     Coefficient of variation (market volatility).

All metric functions are pure: they take arrays or DataFrames and return floats,
with no side effects, so they are easy to test and reuse.
"""
