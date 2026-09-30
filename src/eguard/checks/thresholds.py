"""Threshold lookup for checks, honouring the configuration of the current check call.

Checks call ``get_threshold(name)``. Public check functions are decorated with
``@with_config``, which makes that call's ``config`` argument active while it runs, so
overrides reach every check without passing the configuration through each helper.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextvars import ContextVar
from typing import ParamSpec, TypeVar

from eguard.config import DEFAULT_THRESHOLDS, GuardConfig
from eguard.scoring import Threshold

__all__ = ["DEFAULT_THRESHOLDS", "get_threshold", "with_config"]

_ACTIVE: ContextVar[GuardConfig | None] = ContextVar("eguard_active_config", default=None)

P = ParamSpec("P")
R = TypeVar("R")


def get_threshold(name: str) -> Threshold:
    """The threshold for `name` under the active configuration (defaults if none)."""
    config = _ACTIVE.get()
    if config is None:
        return DEFAULT_THRESHOLDS[name]
    return config.threshold(name)


def with_config(func: Callable[P, R]) -> Callable[P, R]:
    """Make the call's `config` keyword argument active while the function runs."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _ACTIVE.set(kwargs.get("config"))  # type: ignore[arg-type]
        try:
            return func(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    return wrapper
