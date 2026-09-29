"""Default thresholds for built-in checks. Provisional: see docs/metric_definitions.md."""

from eguard.results import Threshold

DEFAULT_THRESHOLDS: dict[str, Threshold] = {
    "representation": Threshold(warn_at=0.5, fail_at=0.25, direction="lower_is_worse"),
    "label_parity": Threshold(warn_at=0.9, fail_at=0.8, direction="lower_is_worse"),
    "proxy": Threshold(warn_at=0.3, fail_at=0.9, direction="higher_is_worse"),
}
