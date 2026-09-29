"""Group fairness metrics: demographic parity ratio and equalized odds difference."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from eguard.metrics._validation import (
    as_binary_array,
    as_group_array,
    check_same_length,
    warn_small_groups,
    warn_undefined,
)


def demographic_parity_ratio(y_pred: npt.ArrayLike, groups: npt.ArrayLike) -> float:
    """Ratio of the lowest to the highest group selection rate.

    DPR = min_g P(Y_hat = 1 | A = g) / max_g P(Y_hat = 1 | A = g)

    Returns a value in [0, 1]; 1 means identical selection rates. Lower is worse.
    Values below 0.8 fail the four-fifths rule.
    Returns nan (with an EguardWarning) if no group has any positive predictions.
    """
    pred = as_binary_array(y_pred, "y_pred")
    grp, labels = as_group_array(groups, expected_length=pred.size)
    warn_small_groups(grp, labels)

    rates = np.array([pred[grp == label].mean() for label in labels])
    highest = rates.max()
    if highest == 0:
        warn_undefined("Demographic parity ratio is undefined: no group has any selections.")
        return float("nan")
    return float(rates.min() / highest)


def equalized_odds_difference(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, groups: npt.ArrayLike
) -> float:
    """Largest gap in true positive rate or false positive rate between groups.

    EOD = max(max_g TPR_g - min_g TPR_g, max_g FPR_g - min_g FPR_g)

    Returns a value in [0, 1]; 0 means equal error rates. Higher is worse.
    Returns nan (with an EguardWarning) if any group lacks actual positives or negatives.
    """
    true = as_binary_array(y_true, "y_true")
    pred = as_binary_array(y_pred, "y_pred")
    check_same_length(true, pred, "y_true", "y_pred")
    grp, labels = as_group_array(groups, expected_length=true.size)
    warn_small_groups(grp, labels)

    tprs: list[float] = []
    fprs: list[float] = []
    problems: list[str] = []
    for label in labels:
        in_group = grp == label
        positives = in_group & (true == 1)
        negatives = in_group & (true == 0)
        if positives.any():
            tprs.append(float(pred[positives].mean()))
        else:
            problems.append(f"group {label} has no actual positives (TPR undefined)")
        if negatives.any():
            fprs.append(float(pred[negatives].mean()))
        else:
            problems.append(f"group {label} has no actual negatives (FPR undefined)")

    if problems:
        warn_undefined("Equalized odds difference is undefined: " + "; ".join(problems) + ".")
        return float("nan")
    return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))
