"""Agent interfaces: how an agent under evaluation plugs into the suite."""

from eguard.agents.screening import (
    DECISION_NAME,
    FunctionAgent,
    ScreeningAgent,
    agent_name,
    as_screening_agent,
    screen,
)

__all__ = [
    "DECISION_NAME",
    "FunctionAgent",
    "ScreeningAgent",
    "agent_name",
    "as_screening_agent",
    "screen",
]
