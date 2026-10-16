"""ToolRegistry: agents look tools up by name instead of importing implementations."""
from __future__ import annotations

from typing import Dict, List

from .anomaly_detector import AnomalyDetectorTool
from .base import Tool
from .client_loader import ClientLoaderTool
from .crm import CRMTool
from .portfolio_metrics import PortfolioMetricsTool


class ToolRegistry:
    """Maps tool names to Tool instances."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance under its ``name``.

        Re-registering the same name replaces the previous tool.
        """
        if not isinstance(tool, Tool):
            raise TypeError(
                f"expected Tool instance, got {type(tool).__name__}"
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool registered as ``name``.

        Raises:
            KeyError: naming the unknown tool and listing registered names.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(
                f"unknown tool: {name!r}. Registered tools: {self.list()}"
            ) from None

    def list(self) -> List[str]:
        """Sorted names of all registered tools."""
        return sorted(self._tools)


def create_default_registry() -> ToolRegistry:
    """Build a registry with all built-in tools registered."""
    registry = ToolRegistry()
    for tool in (
        ClientLoaderTool(),
        CRMTool(),
        PortfolioMetricsTool(),
        AnomalyDetectorTool(),
    ):
        registry.register(tool)
    return registry
