"""AnalyzerAgent: ANALYZE step -- metrics, anomalies and risk score."""
from __future__ import annotations

from ..context import RunContext
from .base import AgentResult, BaseAgent


class AnalyzerAgent(BaseAgent):
    """Compute portfolio metrics, run anomaly detection and score risk.

    Writes ``portfolio_summary``, ``anomalies`` and ``risk_score`` to the
    context. When the client has no transactions, the transaction rules
    naturally yield nothing and a warning is recorded.
    """

    name = "analyzer"

    def run(self, context: RunContext) -> AgentResult:
        """Analyze the client in the context."""
        log = self._log(context)
        log.info("analyze_start", f"analyzing client {context.client_id}")

        client = context.get("client")
        if client is None:
            return AgentResult(
                status="failed", message="no client in context; run data_fetcher first"
            )

        if not client.transactions:
            message = "no transactions; transaction anomaly rules skipped"
            context.add_warning(message)
            log.warning("no_transactions", message)

        metrics_result = self.registry.get("portfolio_metrics").run(client=client)
        if not metrics_result.ok:
            log.error("metrics_failed", metrics_result.error or "unknown error")
            return AgentResult(status="failed", message="portfolio metrics failed")
        metrics = metrics_result.data or {}
        context.set("portfolio_summary", metrics)
        log.info(
            "metrics_done",
            f"total_value={metrics.get('total_value', 0):,.0f}",
        )

        detector_result = self.registry.get("anomaly_detector").run(
            client=client,
            crm=context.get("crm"),
            prior_balances=context.get("prior_balances"),
        )
        if not detector_result.ok:
            log.error("detection_failed", detector_result.error or "unknown error")
            return AgentResult(status="failed", message="anomaly detection failed")
        data = detector_result.data or {}
        anomalies = data.get("anomalies", [])
        risk_score = data.get("risk_score", 0)
        context.set("anomalies", anomalies)
        context.set("risk_score", risk_score)
        log.info(
            "analyze_end",
            f"{len(anomalies)} anomalies, risk_score={risk_score}",
            extra={"anomaly_count": len(anomalies), "risk_score": risk_score},
        )
        return AgentResult(
            status="ok",
            message=f"{len(anomalies)} anomalies, risk score {risk_score}",
        )
