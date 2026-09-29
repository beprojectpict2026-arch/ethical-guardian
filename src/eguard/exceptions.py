"""Custom warnings raised by eguard."""


class EguardWarning(UserWarning):
    """A result is valid but needs attention, e.g. undefined or statistically unstable."""


class ManifestError(ValueError):
    """An agent manifest could not be read or is invalid."""
