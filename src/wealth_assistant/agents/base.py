"""BaseAgent: shared foundation for all pipeline agents.

Agents never call raw implementations; the ToolRegistry is injected via
the constructor and is the only way an agent reaches a tool.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from ..context import RunContext
from ..logging_setup import RunLogger, get_logger
from ..tools.registry import ToolRegistry


@dataclass
class AgentResult:
    """Outcome of one agent run.

    Attributes:
        status: "ok", "degraded" (completed with warnings), or "failed".
        message: Human-readable summary of the outcome.
    """

    status: Literal["ok", "degraded", "failed"]
    message: str = ""


class BaseAgent(ABC):
    """Abstract agent with an injected tool registry."""

    def __init__(self, registry: ToolRegistry) -> None:
        """Create the agent.

        Args:
            registry: ToolRegistry used for every tool call.
        """
        self.registry = registry

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name, used in logs and trace entries."""
        ...

    @abstractmethod
    def run(self, context: RunContext) -> AgentResult:
        """Execute the agent's pipeline step against the run context."""
        ...

    def _log(self, context: RunContext) -> RunLogger:
        """Structured logger bound to this run and agent."""
        return get_logger(
            f"wealth_assistant.agents.{self.name}", run_id=context.run_id
        ).bind(agent=self.name)
