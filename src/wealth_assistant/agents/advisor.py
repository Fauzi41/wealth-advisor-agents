"""AdvisorAgent: RECOMMEND step -- deterministic advice + narrative.

Builds recommendations from findings with fixed rules (no LLM involved in
the numbers), then asks the llm_summarizer tool for prose. This module must
not import anything from the llm/ package; the summarizer tool owns all
LLM contact.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..context import RunContext
from ..models import CRMRecord, Recommendation
from .base import AgentResult, BaseAgent


def _severity(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("severity", ""))
    return str(getattr(item, "severity", ""))


def _rule(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("rule", item.get("type", "")))
    return str(getattr(item, "rule", ""))


def _evidence(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item.get("evidence", {}) or {})
    return dict(getattr(item, "evidence", {}) or {})


class AdvisorAgent(BaseAgent):
    """Turn analyzed findings into recommendations and a narrative."""

    name = "advisor"

    def run(self, context: RunContext) -> AgentResult:
        """Build recommendations, then generate the narrative."""
        log = self._log(context)
        log.info("advise_start", f"advising client {context.client_id}")

        anomalies: List[Any] = context.get("anomalies", []) or []
        crm: Optional[CRMRecord] = context.get("crm")

        recommendations = self._build_recommendations(anomalies)
        context.set("recommendations", recommendations)
        log.info(
            "recommendations_done",
            f"{len(recommendations)} recommendations",
            extra={"count": len(recommendations)},
        )

        findings = self._build_findings(context, anomalies, recommendations)
        summarizer = self.registry.get("llm_summarizer")
        llm_expected = getattr(summarizer, "client", None) is not None
        result = summarizer.run(findings=findings)
        if not result.ok:
            # The tool promises never to raise, but stay defensive.
            log.error("summarize_failed", result.error or "unknown error")
            return AgentResult(status="failed", message="narrative failed")

        narrative = result.data or {}
        summary_source = (result.meta or {}).get("summary_source", "template")
        context.set("narrative", narrative)
        context.set("summary_source", summary_source)
        log.info("narrative_done", f"summary_source={summary_source}")

        if summary_source == "template" and llm_expected:
            message = "LLM narrative unavailable; used template fallback"
            context.add_warning(message)
            log.warning("llm_fallback", message)

        log.info("advise_end", f"source={summary_source}")
        return AgentResult(
            status="ok",
            message=f"{len(recommendations)} recommendations, "
            f"narrative via {summary_source}",
        )

    def _build_findings(
        self,
        context: RunContext,
        anomalies: List[Any],
        recommendations: List[Recommendation],
    ) -> Dict[str, Any]:
        """Assemble the findings dict for the summarizer (no PII)."""
        crm: Optional[CRMRecord] = context.get("crm")
        crm_flags: Dict[str, Any] = {}
        if crm is not None:
            crm_flags = {
                "kyc_status": crm.kyc_status,
                "kyc_expiry": crm.kyc_expiry.isoformat(),
                "last_contact_date": crm.last_contact_date.isoformat(),
                "life_events": list(crm.life_events),
            }
        return {
            "client_id": context.client_id,
            "risk_score": context.get("risk_score", 0),
            "anomalies": anomalies,
            "portfolio_summary": context.get("portfolio_summary", {}) or {},
            "recommendations": [r.model_dump() for r in recommendations],
            "crm_flags": crm_flags,
            "data_quality_warnings": list(context.warnings),
        }

    def _build_recommendations(
        self, anomalies: List[Any]
    ) -> List[Recommendation]:
        """Map findings to deterministic recommendations.

        Money-moving or otherwise high-impact actions require approval.
        """
        by_rule: Dict[str, List[Any]] = {}
        for anomaly in anomalies:
            by_rule.setdefault(_rule(anomaly), []).append(anomaly)

        recommendations: List[Recommendation] = []

        if "allocation_drift" in by_rule:
            classes = sorted(
                {
                    str(_evidence(a).get("asset_class", "?"))
                    for a in by_rule["allocation_drift"]
                }
            )
            recommendations.append(
                Recommendation(
                    action="Rebalance portfolio toward target allocation",
                    rationale="Allocation drifted more than 5pp from target in: "
                    + ", ".join(classes) + ".",
                    priority="medium",
                    requires_approval=True,
                )
            )

        if "concentration" in by_rule:
            symbols = sorted(
                {
                    str(_evidence(a).get("symbol", "?"))
                    for a in by_rule["concentration"]
                }
            )
            recommendations.append(
                Recommendation(
                    action="Reduce single-holding concentration",
                    rationale=", ".join(symbols)
                    + " exceeds 30% of portfolio value; diversify to lower "
                    "idiosyncratic risk.",
                    priority="high",
                    requires_approval=True,
                )
            )

        if "profile_mismatch" in by_rule:
            recommendations.append(
                Recommendation(
                    action="Review risk profile with client",
                    rationale="Portfolio equity exposure does not match the "
                    "client's stated risk profile.",
                    priority="high",
                    requires_approval=True,
                )
            )

        expired_kyc = [
            a
            for a in by_rule.get("crm_kyc", [])
            if _evidence(a).get("expired")
        ]
        if expired_kyc:
            recommendations.append(
                Recommendation(
                    action="Refresh KYC documentation immediately",
                    rationale="KYC expired "
                    + str(_evidence(expired_kyc[0]).get("kyc_expiry", ""))
                    + "; account restrictions may apply until renewed.",
                    priority="urgent",
                    requires_approval=False,
                )
            )

        high = [a for a in anomalies if _severity(a) == "high"]
        if high:
            rules = sorted({_rule(a) for a in high})
            recommendations.append(
                Recommendation(
                    action="Contact client to verify flagged activity",
                    rationale=f"{len(high)} high-severity finding(s): "
                    + ", ".join(rules) + ".",
                    priority="high",
                    requires_approval=True,
                )
            )

        if "crm_stale_contact" in by_rule:
            days = _evidence(by_rule["crm_stale_contact"][0]).get(
                "days_since_contact", "?"
            )
            recommendations.append(
                Recommendation(
                    action="Schedule advisor review meeting",
                    rationale=f"No advisor contact for {days} days; "
                    "re-engage the client.",
                    priority="low",
                    requires_approval=False,
                )
            )

        return recommendations
