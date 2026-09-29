"""Ethical Guardian Suite: lifecycle-adaptive ethical and economic evaluation of AI agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ethical-guardian")
except PackageNotFoundError:  # package not installed, e.g. running from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
