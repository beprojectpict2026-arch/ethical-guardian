"""Custom warnings raised by eguard."""


class EguardWarning(UserWarning):
    """A result is valid but needs attention, e.g. undefined or statistically unstable."""


class ManifestError(ValueError):
    """An agent manifest could not be read or is invalid."""


class EvaluationFailed(AssertionError):
    """A report contains failing or undefined checks. Raised by Report.raise_for_status()."""


class AgentError(ValueError):
    """An agent crashed or returned output that breaks its interface contract."""


class ConfigError(ValueError):
    """A configuration file could not be read or is invalid."""
