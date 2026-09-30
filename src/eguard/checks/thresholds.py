"""Default thresholds for built-in checks. Provisional: see docs/metric_definitions.md."""

from eguard.results import Threshold

DEFAULT_THRESHOLDS: dict[str, Threshold] = {
    "representation": Threshold(warn_at=0.5, fail_at=0.25, direction="lower_is_worse"),
    "label_parity": Threshold(warn_at=0.9, fail_at=0.8, direction="lower_is_worse"),
    "proxy": Threshold(warn_at=0.3, fail_at=0.9, direction="higher_is_worse"),
    "demographic_parity": Threshold(warn_at=0.9, fail_at=0.8, direction="lower_is_worse"),
    "equalized_odds": Threshold(warn_at=0.05, fail_at=0.1, direction="higher_is_worse"),
    "counterfactual_flip": Threshold(warn_at=0.01, fail_at=0.05, direction="higher_is_worse"),
    "job_displacement": Threshold(warn_at=20, fail_at=50, direction="higher_is_worse"),
    "documentation": Threshold(warn_at=1.0, fail_at=0.5, direction="lower_is_worse"),
}
