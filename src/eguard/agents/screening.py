"""Interface for hiring (screening) agents.

An agent is anything with ``decide(applicants) -> decisions``: one yes/no per applicant row.
Plain functions with the same signature are accepted and wrapped automatically. The library
always calls agents through ``screen``, which protects the input data and validates output.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from eguard.exceptions import AgentError

DECISION_NAME = "shortlisted"

Decisions = pd.Series | Sequence[bool] | Sequence[int]


@runtime_checkable
class ScreeningAgent(Protocol):
    """A hiring agent: decides which applicants to shortlist."""

    def decide(self, applicants: pd.DataFrame) -> Decisions:
        """Return one decision per row of `applicants`: True/1 = shortlist, False/0 = reject."""
        ...


class FunctionAgent:
    """Wraps a plain function ``fn(applicants) -> decisions`` as a ScreeningAgent."""

    def __init__(self, fn: Callable[[pd.DataFrame], Decisions], name: str | None = None):
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "function_agent")

    def decide(self, applicants: pd.DataFrame) -> Decisions:
        return self._fn(applicants)


def as_screening_agent(agent: Any) -> ScreeningAgent:
    """Return `agent` itself if it has ``decide``, wrap it if it is a function, else fail."""
    if isinstance(agent, ScreeningAgent):
        return agent
    if callable(agent):
        return FunctionAgent(agent)
    raise TypeError(
        f"{type(agent).__name__} is not a screening agent: it needs a "
        "`decide(applicants)` method, or must be a function taking the applicants."
    )


def agent_name(agent: Any) -> str:
    """A readable name for reports: the agent's `name` attribute, else its class name."""
    return str(getattr(agent, "name", None) or type(agent).__name__)


def screen(agent: Any, applicants: pd.DataFrame) -> pd.Series:
    """Ask an agent for decisions and return them validated.

    The agent receives a copy of `applicants`, so it cannot alter the caller's data.

    Returns:
        Boolean Series named "shortlisted", indexed like `applicants`.

    Raises:
        TypeError: if `agent` is neither a ScreeningAgent nor a function.
        AgentError: if the agent raises an exception or returns invalid decisions.
    """
    wrapped = as_screening_agent(agent)
    name = agent_name(wrapped)
    try:
        output = wrapped.decide(applicants.copy())
    except Exception as exc:
        raise AgentError(f"agent '{name}' raised {type(exc).__name__}: {exc}") from exc
    return _validate(output, applicants.index, name)


def _validate(output: Any, index: pd.Index, name: str) -> pd.Series:
    if output is None:
        raise AgentError(f"agent '{name}' returned None instead of decisions.")

    if isinstance(output, pd.Series):
        if len(output) != len(index) or not output.index.sort_values().equals(index.sort_values()):
            raise AgentError(
                f"agent '{name}' returned decisions for different applicants than it was "
                f"given ({len(output)} decisions for {len(index)} applicants)."
            )
        decisions = output.reindex(index)
    else:
        values = list(output)
        if len(values) != len(index):
            raise AgentError(
                f"agent '{name}' returned {len(values)} decisions for {len(index)} applicants."
            )
        decisions = pd.Series(values, index=index)

    if decisions.isna().any():
        raise AgentError(f"agent '{name}' returned missing decisions.")
    if not (pd.api.types.is_bool_dtype(decisions) or pd.api.types.is_numeric_dtype(decisions)):
        raise AgentError(f"agent '{name}' must return True/False or 1/0 decisions.")
    if not decisions.isin([0, 1]).all():
        raise AgentError(f"agent '{name}' must return True/False or 1/0 decisions.")
    return decisions.astype(bool).rename(DECISION_NAME)
