"""Inequality metrics: Gini coefficient and Gini delta between two distributions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eguard.metrics._validation import as_1d_float_array


def gini(values: npt.ArrayLike) -> float:
    """Gini coefficient of a non-negative distribution.

    G = (2 * sum(i * w_i)) / (n * sum(w_i)) - (n + 1) / n,
    with w sorted ascending and i = 1..n.

    Returns a value in [0, (n-1)/n]; 0 means perfect equality. Higher is worse.
    Returns 0.0 for a single value or when all values are zero.

    Raises:
        ValueError: if input is empty, non-numeric, non-finite or contains negatives.
    """
    w = as_1d_float_array(values, "values")
    if np.any(w < 0):
        raise ValueError("`values` must be non-negative.")

    n = w.size
    total = w.sum()
    if n == 1 or total == 0:
        return 0.0

    w = np.sort(w)
    ranks = np.arange(1, n + 1)
    g = 2.0 * np.sum(ranks * w) / (n * total) - (n + 1) / n
    return float(max(g, 0.0))  # guard against tiny negative floating-point error


def gini_delta(baseline: npt.ArrayLike, agent: npt.ArrayLike) -> float:
    """Change in inequality caused by the agent: G(agent) - G(baseline).

    Both distributions must come from the same population, seed and evaluation window.
    Returns a value in [-1, 1]; positive means the agent increased inequality.
    """
    return gini(agent) - gini(baseline)
