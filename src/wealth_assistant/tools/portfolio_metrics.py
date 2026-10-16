"""Portfolio metrics: total value, allocation, drift and concentration."""
from __future__ import annotations

from typing import Any, Dict

from ..models import Client
from .base import Tool, ToolResult


def compute_portfolio_metrics(client: Client) -> Dict[str, Any]:
    """Compute deterministic portfolio metrics for a client.

    Returns:
        total_value: sum of holding market values (0 when no holdings).
        allocation: asset class -> fraction of total value (empty when 0).
        drift: asset class -> (actual - target) in percentage points
            (empty when the client has no target_allocation).
        concentration: {"symbol", "fraction"} of the largest holding
            (empty when there are no holdings).

    Never raises on missing/empty data.
    """
    holdings = client.holdings or []
    total_value = sum(h.market_value for h in holdings)

    allocation: Dict[str, float] = {}
    if total_value > 0:
        by_class: Dict[str, float] = {}
        for holding in holdings:
            by_class[holding.asset_class] = (
                by_class.get(holding.asset_class, 0.0) + holding.market_value
            )
        allocation = {
            asset_class: value / total_value for asset_class, value in by_class.items()
        }

    drift: Dict[str, float] = {}
    if client.target_allocation:
        for asset_class, target in client.target_allocation.items():
            drift[asset_class] = (allocation.get(asset_class, 0.0) - target) * 100.0

    concentration: Dict[str, Any] = {}
    if holdings and total_value > 0:
        top = max(holdings, key=lambda h: h.market_value)
        concentration = {
            "symbol": top.symbol,
            "fraction": top.market_value / total_value,
        }

    return {
        "total_value": total_value,
        "allocation": allocation,
        "drift": drift,
        "concentration": concentration,
    }


class PortfolioMetricsTool(Tool):
    """Tool wrapper around compute_portfolio_metrics."""

    name = "portfolio_metrics"

    def _execute(self, client: Any) -> ToolResult:
        typed = client if isinstance(client, Client) else Client(**client)
        metrics = compute_portfolio_metrics(typed)
        return ToolResult(
            ok=True, data=metrics, meta={"client_id": typed.client_id}
        )
