"""Shared input validation and warning helpers for metric functions (internal)."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from eguard.exceptions import EguardWarning

MIN_GROUP_SIZE = 30


def as_1d_float_array(values: npt.ArrayLike, name: str) -> np.ndarray:
    """Return a finite, non-empty, one-dimensional float array, or raise ValueError."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must contain only numbers.") from exc
    if arr.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"`{name}` must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` must not contain NaN or infinite values.")
    return arr


def as_binary_array(values: npt.ArrayLike, name: str) -> np.ndarray:
    """Return an integer array containing only 0 and 1, or raise ValueError."""
    arr = as_1d_float_array(values, name)
    if not np.all(np.isin(arr, (0.0, 1.0))):
        raise ValueError(f"`{name}` must contain only 0 and 1.")
    return arr.astype(int)


def as_group_array(groups: npt.ArrayLike, expected_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate group labels. Returns (labels per sample, sorted distinct labels)."""
    arr = np.asarray(groups)
    if arr.ndim != 1:
        raise ValueError(f"`groups` must be one-dimensional, got shape {arr.shape}.")
    if arr.shape[0] != expected_length:
        raise ValueError(f"`groups` has length {arr.shape[0]}, expected {expected_length}.")
    if pd.isna(arr).any():
        raise ValueError("`groups` must not contain missing labels.")
    try:
        labels = np.unique(arr)
    except TypeError as exc:
        raise ValueError("`groups` labels must all be of one comparable type.") from exc
    if labels.size < 2:
        raise ValueError(f"`groups` must contain at least 2 distinct groups, got {labels.size}.")
    return arr, labels


def check_same_length(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    """Raise ValueError if two arrays differ in length."""
    if a.size != b.size:
        raise ValueError(
            f"`{name_a}` and `{name_b}` must have the same length ({a.size} != {b.size})."
        )


def warn_undefined(message: str) -> None:
    """Warn that a metric is undefined for valid input. Call directly from a metric function."""
    warnings.warn(message, EguardWarning, stacklevel=3)


def warn_small_groups(
    groups: np.ndarray, labels: np.ndarray, min_size: int = MIN_GROUP_SIZE
) -> None:
    """Warn if any group has fewer than `min_size` members. Call directly from a metric."""
    small = {str(label): int(np.sum(groups == label)) for label in labels}
    small = {label: size for label, size in small.items() if size < min_size}
    if small:
        details = ", ".join(f"{label}: {size}" for label, size in small.items())
        warnings.warn(
            f"Groups smaller than {min_size} give unstable rates ({details}).",
            EguardWarning,
            stacklevel=3,
        )
