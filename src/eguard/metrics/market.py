"""Market stability metrics: coefficient of variation of a price series."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eguard.metrics._validation import as_1d_float_array


def coefficient_of_variation(prices: npt.ArrayLike) -> float:
    """Price volatility relative to the mean price, in percent.

    CV = 100 * sigma / mu, using the population standard deviation (ddof=0),
    since the series is the complete evaluation window.

    Returns a value >= 0. Higher is worse, compared against the baseline run.

    Raises:
        ValueError: if fewer than 2 prices, or any price is non-positive or non-finite.
    """
    p = as_1d_float_array(prices, "prices")
    if p.size < 2:
        raise ValueError("`prices` must contain at least 2 values.")
    if np.any(p <= 0):
        raise ValueError("`prices` must be strictly positive.")
    return float(100.0 * p.std(ddof=0) / p.mean())
