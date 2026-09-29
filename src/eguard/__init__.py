"""Ethical Guardian Suite: lifecycle-adaptive ethical and economic evaluation of AI agents."""

from importlib.metadata import PackageNotFoundError, version

from eguard.checks import check, evaluate
from eguard.config import GuardConfig
from eguard.exceptions import AgentError, EguardWarning, EvaluationFailed, ManifestError
from eguard.manifest import AgentProfile
from eguard.results import CheckResult, Phase, Report, Status, Threshold

try:
    __version__ = version("ethical-guardian")
except PackageNotFoundError:  # package not installed, e.g. running from a raw checkout
    __version__ = "0.0.0"

__all__ = [
    "AgentError",
    "AgentProfile",
    "CheckResult",
    "EguardWarning",
    "EvaluationFailed",
    "GuardConfig",
    "ManifestError",
    "Phase",
    "Report",
    "Status",
    "Threshold",
    "check",
    "evaluate",
    "__version__",
]
