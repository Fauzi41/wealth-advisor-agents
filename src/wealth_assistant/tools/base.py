"""Tool foundation: the ToolResult envelope and the Tool abstract base class.

Agents never call raw implementations; they call tools through a
ToolRegistry. Every tool returns a ToolResult and never raises to its
caller -- the base class guarantees this by routing run() through
_execute() inside a try/except.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """Uniform envelope returned by every tool.

    Attributes:
        ok: True when the tool completed successfully.
        data: Payload produced by the tool (shape is tool-specific).
        error: Human-readable message when ok is False, else None.
        meta: Auxiliary info (counts, paths, retry attempts, ...).
    """

    ok: bool
    data: Any = None
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for all tools.

    Subclasses provide a ``name`` and implement ``_execute``. The public
    ``run`` method wraps ``_execute`` so a tool never raises to its
    caller: unexpected exceptions become ``ToolResult(ok=False, ...)``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique registry name for this tool."""
        ...

    @abstractmethod
    def _execute(self, **kwargs: Any) -> ToolResult:
        """Tool logic. May raise; ``run`` converts it into a ToolResult."""
        ...

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, never raising to the caller."""
        try:
            result = self._execute(**kwargs)
        except Exception as exc:  # noqa: BLE001 - envelope guarantee
            return ToolResult(ok=False, error=f"{self.name} failed: {exc}")
        if not isinstance(result, ToolResult):
            return ToolResult(
                ok=False,
                error=f"{self.name} returned {type(result).__name__}, "
                "expected ToolResult",
            )
        return result
