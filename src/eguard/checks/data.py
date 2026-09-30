"""Data-phase checks: inspect historical or training data before any agent is built.

For each protected attribute listed in the agent manifest:
    representation  is any group much smaller than the others?
    label parity    do historical outcomes already differ between groups?
    proxy           can another column stand in for the protected attribute?

Definitions and thresholds: docs/metric_definitions.md sections 2, 3a and 3b.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

from eguard.checks.thresholds import get_threshold, with_config
from eguard.config import GuardConfig
from eguard.manifest import AgentProfile, Dimension
from eguard.metrics import demographic_parity_ratio, proxy_strength
from eguard.results import CheckResult, Phase, Report, Status

PHASE = Phase.DATA


@with_config
def check_data(
    profile: AgentProfile,
    data: pd.DataFrame,
    target: str,
    *,
    exclude: Sequence[str] = (),
    config: GuardConfig | None = None,
) -> Report:
    """Run data-phase checks on historical data.

    Args:
        profile: agent manifest; its `protected_attributes` must be columns of `data`.
        data: one row per past case (e.g. applicant), including the outcome column.
        target: binary outcome column, e.g. "hired".
        exclude: columns never tested as proxies (e.g. columns you know are harmless).
        config: scoring configuration; defaults to GuardConfig().
    """
    _validate_columns(profile, data, target, exclude)
    protected = list(profile.protected_attributes)
    proxy_columns = _proxy_candidates(data, target, protected, exclude)

    results: list[CheckResult] = []
    for attribute in protected:
        groups = data[attribute]
        results.append(_representation(attribute, groups))
        parity = _label_parity(attribute, groups, data[target], target)
        results.append(parity)
        label_disparity = parity.risk if parity.risk is not None else 1.0
        results.append(_proxy(attribute, groups, data, proxy_columns, label_disparity))

    notes: dict[str, Any] = {"target": target, "rows": len(data), "proxy_columns": proxy_columns}
    generator = data.attrs.get("generator")
    if generator:
        notes["data_generator"] = {
            "seed": generator["seed"],
            "bias_strength": generator["bias_strength"],
        }
    return Report.create(profile, PHASE, results, config=config, notes=notes)


# ----------------------------------------------------------------------------- checks


def _representation(attribute: str, groups: pd.Series) -> CheckResult:
    shares = groups.value_counts(normalize=True)
    ratio = float(shares.min() / shares.max())
    smallest, largest = shares.idxmin(), shares.idxmax()
    breakdown = ", ".join(f"{group} {share:.0%}" for group, share in shares.items())
    return CheckResult.from_value(
        f"equity.representation_{_slug(attribute)}",
        Dimension.EQUITY,
        PHASE,
        ratio,
        get_threshold("representation"),
        risk=None,
        message=(
            f"Smallest {attribute} group ({smallest}) is {ratio:.0%} the size of the largest "
            f"({largest}): {breakdown}."
        ),
        evidence={
            "group_shares": {str(g): float(s) for g, s in shares.items()},
            "group_sizes": {str(g): int(c) for g, c in groups.value_counts().items()},
        },
        remediation=[
            "Collect more data for under-represented groups before training.",
            "Report results per group, including group sizes.",
        ],
    )


def _label_parity(attribute: str, groups: pd.Series, labels: pd.Series, target: str) -> CheckResult:
    ratio = demographic_parity_ratio(labels, groups)
    rates = labels.groupby(groups).mean()
    low, high = rates.idxmin(), rates.idxmax()
    if math.isnan(ratio):
        message = f"Label parity for {attribute} is undefined: no row has a positive '{target}'."
    else:
        message = (
            f"Historical '{target}' rate for {attribute}={low} is {ratio:.0%} of the rate for "
            f"{high} ({rates[low]:.1%} vs {rates[high]:.1%})."
        )
    return CheckResult.from_value(
        f"equity.label_parity_{_slug(attribute)}",
        Dimension.EQUITY,
        PHASE,
        ratio,
        get_threshold("label_parity"),
        risk=None if math.isnan(ratio) else 1.0 - ratio,
        message=message,
        evidence={"positive_rates": {str(g): float(r) for g, r in rates.items()}},
        remediation=[
            "Do not train directly on this label without correction: a model will learn "
            "the historical disparity.",
            "Investigate why past outcomes differ between groups (criteria, process, reviewers).",
            "Consider reweighting or resampling so groups have comparable label rates, then "
            "re-run this check.",
        ],
    )


def _proxy(
    attribute: str,
    groups: pd.Series,
    data: pd.DataFrame,
    columns: list[str],
    label_disparity: float,
) -> CheckResult:
    check_id = f"equity.proxy_{_slug(attribute)}"
    if not columns:
        return CheckResult.not_applicable(
            check_id,
            Dimension.EQUITY,
            PHASE,
            f"No other columns to test as proxies for {attribute}.",
        )
    scores = {col: proxy_strength(data[col], groups) for col in columns}
    ranked = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    top, strength = next(iter(ranked.items()))
    threshold = get_threshold("proxy")
    if threshold.evaluate(strength) is Status.PASS:
        message = (
            f"No column strongly predicts {attribute}; the strongest is '{top}' "
            f"(proxy strength {strength:.2f})."
        )
    else:
        message = (
            f"'{top}' predicts {attribute} (proxy strength {strength:.2f}): a model could use "
            f"it to reproduce group differences even if '{attribute}' is removed."
        )
    return CheckResult.from_value(
        check_id,
        Dimension.EQUITY,
        PHASE,
        strength,
        threshold,
        risk=strength * label_disparity,
        message=message,
        evidence={
            "strongest_proxy": top,
            "proxy_strength": {col: round(s, 4) for col, s in ranked.items()},
            "label_disparity": round(label_disparity, 4),
        },
        remediation=[
            f"Remove or coarsen '{top}' before training, or document why it is needed.",
            "Run counterfactual tests in the testing phase to confirm the agent does not rely "
            "on it.",
        ],
    )


# ---------------------------------------------------------------------------- helpers


def _validate_columns(
    profile: AgentProfile, data: pd.DataFrame, target: str, exclude: Sequence[str]
) -> None:
    if target not in data.columns:
        raise ValueError(f"target column '{target}' not found in data.")
    if target in profile.protected_attributes:
        raise ValueError(f"target '{target}' cannot also be a protected attribute.")
    missing = [a for a in profile.protected_attributes if a not in data.columns]
    if missing:
        raise ValueError(f"protected attribute column(s) missing from data: {', '.join(missing)}.")
    unknown = [c for c in exclude if c not in data.columns]
    if unknown:
        raise ValueError(f"`exclude` names unknown column(s): {', '.join(unknown)}.")


def _proxy_candidates(
    data: pd.DataFrame, target: str, protected: list[str], exclude: Sequence[str]
) -> list[str]:
    skip = {target, *protected, *exclude}
    return [c for c in data.columns if c not in skip and not _is_identifier(data[c])]


def _is_identifier(column: pd.Series) -> bool:
    """Integer or text columns with a unique value per row, such as IDs, are not features."""
    if not column.is_unique:
        return False
    return pd.api.types.is_integer_dtype(column) or not pd.api.types.is_numeric_dtype(column)


def _slug(name: str) -> str:
    """Make an attribute name safe for a check_id, e.g. 'Age Group' -> 'age_group'."""
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
