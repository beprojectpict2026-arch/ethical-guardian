"""Labour impact metrics: job displacement score."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from eguard.metrics._validation import as_1d_float_array, check_same_length


def job_displacement_score(
    automation: npt.ArrayLike,
    reskilling_feasibility: float,
    *,
    weights: npt.ArrayLike | None = None,
) -> float:
    """Share of an occupation's work automated by the agent, adjusted for reskilling.

    A = sum(u_t * a_t) / sum(u_t)
    JDS = 100 * A * (1 - R)

    Args:
        automation: fraction of each task automated, each in [0, 1].
        reskilling_feasibility: R in [0, 1]; how easily affected workers can move to other roles.
        weights: importance or time share of each task, each > 0. Defaults to equal weights.

    Returns a value in [0, 100]. Higher is worse.
    """
    a = as_1d_float_array(automation, "automation")
    if np.any((a < 0) | (a > 1)):
        raise ValueError("`automation` values must be in [0, 1].")

    if weights is None:
        u = np.ones_like(a)
    else:
        u = as_1d_float_array(weights, "weights")
        check_same_length(a, u, "automation", "weights")
        if np.any(u <= 0):
            raise ValueError("`weights` must be strictly positive.")

    r = float(reskilling_feasibility)
    if not math.isfinite(r) or not 0.0 <= r <= 1.0:
        raise ValueError("`reskilling_feasibility` must be in [0, 1].")

    automated_share = np.sum(u * a) / np.sum(u)
    return float(100.0 * automated_share * (1.0 - r))
