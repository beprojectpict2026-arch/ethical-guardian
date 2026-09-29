"""Agent manifest: the structured description of an agent that a company fills in
at the planning phase. It tells the suite what the agent decides, what it automates,
and who it affects, so the right checks and metrics can be applied."""

from eguard.manifest.schema import (
    DEFAULT_DIMENSIONS,
    AgentProfile,
    AgentType,
    AutomatedTask,
    DeploymentScale,
    Dimension,
    ModelInfo,
)

__all__ = [
    "DEFAULT_DIMENSIONS",
    "AgentProfile",
    "AgentType",
    "AutomatedTask",
    "DeploymentScale",
    "Dimension",
    "ModelInfo",
]
