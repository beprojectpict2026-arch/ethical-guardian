"""Proxy strength: how well one column predicts a protected attribute."""

from __future__ import annotations

import numpy.typing as npt
import pandas as pd

from eguard.metrics._validation import as_group_array, warn_unstable

DEFAULT_BINS = 10
MIN_ROWS_PER_LEVEL = 20


def proxy_strength(
    feature: npt.ArrayLike, attribute: npt.ArrayLike, *, bins: int = DEFAULT_BINS
) -> float:
    """How much knowing `feature` improves guessing `attribute`.

    accuracy = share of rows whose attribute equals the most common attribute among rows
               with the same feature value
    baseline = share of the most common attribute overall
    strength = (accuracy - baseline) / (1 - baseline)

    Numeric features with more than `bins` distinct values are first split into `bins`
    equal-count bins. Missing feature values form their own category.

    Returns a value in [0, 1]: 0 = no predictive power, 1 = feature determines the attribute.
    Emits an EguardWarning below 20 rows per feature level, since the score is then
    inflated by chance.
    """
    feat = pd.Series(feature).reset_index(drop=True)
    if len(feat) == 0:
        raise ValueError("`feature` must not be empty.")
    if bins < 2:
        raise ValueError("`bins` must be at least 2.")
    groups, _ = as_group_array(attribute, expected_length=len(feat))

    if pd.api.types.is_numeric_dtype(feat) and feat.nunique(dropna=True) > bins:
        levels = pd.qcut(feat.rank(method="first"), q=bins, labels=False).astype(float)
        levels = levels.fillna(-1.0)
    else:
        levels = feat.astype("object").where(feat.notna(), "<missing>")

    table = pd.crosstab(levels.to_numpy(), groups)
    n_rows, n_levels = len(feat), table.shape[0]
    if n_rows < MIN_ROWS_PER_LEVEL * n_levels:
        warn_unstable(
            f"Proxy strength uses {n_levels} feature levels but only {n_rows} rows "
            f"(fewer than {MIN_ROWS_PER_LEVEL} rows per level); the score may be inflated."
        )

    accuracy = table.to_numpy().max(axis=1).sum() / n_rows
    baseline = pd.Series(groups).value_counts(normalize=True).max()
    return float(max(0.0, (accuracy - baseline) / (1.0 - baseline)))
