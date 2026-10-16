"""Orchestrator: declarative agent pipeline, per client, with HITL and memory."""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agents.advisor import AdvisorAgent
from .agents.analyzer import AnalyzerAgent
from .agents.base import AgentResult, BaseAgent
from .agents.data_fetcher import DataFetcherAgent
from .context import RunContext
from .hitl import checkpoint
from .llm.factory import build_llm_client
from .logging_setup import get_logger
from .memory import LongTermMemory
from .models import HumanReview, Narrative, Recommendation, Report
from .tools.client_loader import ClientLoaderTool
from .tools.llm_summarizer import LLMSummarizerTool
from .tools.registry import ToolRegistry, create_default_registry

_logger = logging.getLogger(__name__)


def _llm_disabled_reason() -> str:
    """Best-effort reason for logging when no LLM client could be built."""
    provider = (os.getenv("LLM_PROVIDER") or "none").strip().lower()
    if provider == "none":
        return "LLM_PROVIDER not set"
    if provider not in ("anthropic", "openai"):
        return f"unknown provider {provider!r}"
    key_name = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[
        provider
    ]
    if not os.getenv(key_name):
        return f"{key_name} not set"
    if not (os.getenv("LLM_MODEL") or "").strip():
        return "LLM_MODEL not set"
    return "client unavailable (see warnings)"


def resolve_client_ids(data_path: str, which: str) -> List[str]:
    """Resolve which client ids to process.

    Args:
        data_path: Path to clients.json.
        which: "all" or a single client id.

    Raises:
        RuntimeError: when the data file cannot be loaded at all.
    """
    if which != "all":
        return [which]
    result = ClientLoaderTool(path=data_path).run()
    if not result.ok:
        raise RuntimeError(f"cannot load {data_path}: {result.error}")
    data = result.data or {}
    ids = [c.client_id for c in data.get("clients", [])]
    for error in data.get("errors", []):
        if error.get("client_id"):
            ids.append(error["client_id"])
    return ids


class Orchestrator:
    """Run the INGEST -> ANALYZE -> RECOMMEND -> HITL -> REPORT pipeline."""

    def __init__(
        self,
        data_path: str = "data/clients.json",
        output_dir: str = "outputs",
        memory_path: str = "memory_store/memory.json",
        no_llm: bool = False,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        """Create the orchestrator.

        The LLM client is built exactly once here (unless no_llm or a
        registry is injected, e.g. in tests).
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.memory = LongTermMemory(memory_path)

        if registry is None:
            registry = create_default_registry()
            registry.register(ClientLoaderTool(path=data_path))
            self.llm_client = None
            self.llm_mode = ""
            if no_llm:
                self.llm_mode = "LLM mode: disabled (--no-llm flag)"
            else:
                self.llm_client = build_llm_client()
                if self.llm_client is not None:
                    self.llm_mode = (
                        f"LLM mode: {self.llm_client.provider}/"
                        f"{self.llm_client.model}"
                    )
                else:
                    self.llm_mode = (
                        f"LLM mode: disabled ({_llm_disabled_reason()})"
                    )
            registry.register(LLMSummarizerTool(self.llm_client))
            get_logger("wealth_assistant.orchestrator").info(
                "llm_mode", self.llm_mode
            )
        else:
            self.llm_client = None
            self.llm_mode = "LLM mode: disabled (injected registry)"
        self.registry = registry

        self.steps: List[Dict[str, Any]] = [
            {"name": "ingest", "agent": DataFetcherAgent(self.registry),
             "critical": True},
            {"name": "analyze", "agent": AnalyzerAgent(self.registry),
             "critical": True},
            {"name": "recommend", "agent": AdvisorAgent(self.registry),
             "critical": False},
        ]

    # -- public API ---------------------------------------------------------

    def run(
        self, client_ids: List[str], auto_approve: bool = False
    ) -> List[Dict[str, Any]]:
        """Process every client independently; one failure never stops the batch."""
        return [
            self.run_client(client_id, auto_approve=auto_approve)
            for client_id in client_ids
        ]

    def run_client(
        self, client_id: str, auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Run the full pipeline for one client and write its reports."""
        context = RunContext(client_id=client_id)
        log = get_logger("wealth_assistant.orchestrator",
                         run_id=context.run_id).bind(agent="orchestrator")
        log.info("run_start", f"processing client {client_id}")
        try:
            return self._run_client(context, auto_approve, log)
        except Exception as exc:  # never let one client kill the batch
            log.exception("run_crashed", f"unexpected error: {exc}")
            context.add_trace("orchestrator", "orchestrator", "failed", 0)
            report = self._failed_report(context, "failed",
                                         f"unexpected error: {exc}")
            return self._finish(context, report, log)

    # -- internals -----------------------------------------------------------

    def _run_client(
        self, context: RunContext, auto_approve: bool, log: Any
    ) -> Dict[str, Any]:
        # Long-term memory feeds balance_drop before analysis runs.
        last = self.memory.get_last_summary(context.client_id)
        if last and last.get("total_balance") is not None:
            context.set(
                "prior_balances",
                {context.client_id: last["total_balance"]},
            )

        status = "success"
        for step in self.steps:
            agent: BaseAgent = step["agent"]
            start = time.perf_counter()
            try:
                result = agent.run(context)
                step_status = result.status
                log.info("step_end",
                         f"{step['name']}: {step_status} - {result.message}")
            except Exception as exc:
                log.exception(
                    "step_exception", f"{step['name']} raised: {exc}"
                )
                step_status = "failed"
            duration_ms = int((time.perf_counter() - start) * 1000)
            context.add_trace(step["name"], agent.name, step_status,
                              duration_ms)

            if step_status == "degraded" and status == "success":
                status = "degraded"
            if step_status == "failed":
                if step["critical"]:
                    status = "failed"
                    log.warning("run_stopped",
                                f"critical step {step['name']} failed")
                    break
                status = "degraded"

        if status == "failed":
            report = self._failed_report(
                context, status, "pipeline failed; see trace"
            )
            return self._finish(context, report, log)

        report = self._build_report(context, status)

        current = {
            "anomalies": [
                {"type": a.type, "id": a.id, "severity": a.severity}
                for a in report.anomalies
            ]
        }
        report.changes_since_last_review = self.memory.diff_with_last(
            context.client_id, current
        )

        # checkpoint() mutates its input dict in place, so keep the dumped
        # copy to merge reviewer edits (priority changes, deletions) back.
        hitl_input = report.model_dump(mode="json")
        review = checkpoint(hitl_input, auto_approve=auto_approve)
        report.human_review = review
        edited = {r["action"]: r for r in hitl_input["recommendations"]}
        report.recommendations = [
            Recommendation(**edited[r.action])
            for r in report.recommendations
            if r.action in edited
        ]
        log.info("hitl_done", f"decision={review.decision}")

        if review.decision != "rejected":
            client = context.get("client")
            self.memory.save_summary(
                context.client_id,
                {
                    "risk_score": report.risk_score,
                    "anomalies": current["anomalies"],
                    "recommendations": [r.action for r in report.recommendations],
                    "total_balance": (
                        sum(a.balance for a in client.accounts)
                        if client else 0
                    ),
                },
            )
        else:
            log.warning("hitl_rejected",
                        "report rejected; not saved to memory")
        return self._finish(context, report, log)

    def _finish(
        self, context: RunContext, report: Report, log: Any
    ) -> Dict[str, Any]:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"report_{context.client_id}.json"
        json_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        md_path = out / f"report_{context.client_id}.md"
        md_path.write_text(self._render_markdown(report) + "\n",
                           encoding="utf-8")
        log.info("run_end",
                 f"client={context.client_id} status={report.status} "
                 f"report={json_path}")
        return {
            "client_id": context.client_id,
            "status": report.status,
            "risk_score": report.risk_score,
            "anomaly_count": len(report.anomalies),
            "report_path": str(json_path),
        }

    def _failed_report(
        self, context: RunContext, status: str, message: str
    ) -> Report:
        _logger.warning("report_failed: %s (%s)", context.client_id, message)
        return Report(
            run_id=context.run_id,
            client_id=context.client_id,
            status=status,  # type: ignore[arg-type]
            data_quality_warnings=list(context.warnings),
            risk_score=0,
            narrative=Narrative(
                executive_summary="",
                talking_points=[],
                caveats=[],
                summary_source="template",
            ),
            human_review=HumanReview(
                required=False, decision="not_required", note=""
            ),
            trace=list(context.trace),
        )

    def _build_report(self, context: RunContext, status: str) -> Report:
        anomalies = context.get("anomalies", []) or []
        recommendations = context.get("recommendations", []) or []
        narrative = context.get("narrative", {}) or {}
        crm = context.get("crm")
        return Report(
            run_id=context.run_id,
            client_id=context.client_id,
            status=status,  # type: ignore[arg-type]
            data_quality_warnings=list(context.warnings),
            portfolio_summary=context.get("portfolio_summary", {}) or {},
            anomalies=anomalies,
            risk_score=int(context.get("risk_score", 0) or 0),
            recommendations=recommendations,
            crm_context=crm.model_dump(mode="json") if crm else {},
            narrative=Narrative(
                executive_summary=narrative.get("executive_summary", ""),
                talking_points=narrative.get("talking_points", []),
                caveats=narrative.get("caveats", []),
                summary_source=context.get("summary_source", "template"),
            ),
            human_review=HumanReview(
                required=False, decision="not_required", note=""
            ),
            trace=list(context.trace),
        )

    def _render_markdown(self, report: Report) -> str:
        """Readable summary with tables."""
        lines = [
            f"# Wealth Advisor Report — {report.client_id}",
            "",
            f"- Run ID: `{report.run_id}`",
            f"- Generated: {report.generated_at.isoformat()}",
            f"- Status: **{report.status}**",
            f"- Risk score: **{report.risk_score}/100**",
            "",
            "## Portfolio summary",
            "",
            f"- Total value: "
            f"{report.portfolio_summary.get('total_value', 0):,.2f}",
        ]
        allocation = report.portfolio_summary.get("allocation", {})
        if allocation:
            lines += ["", "| Asset class | Allocation |",
                      "|---|---|"]
            for cls, frac in sorted(allocation.items()):
                lines.append(f"| {cls} | {frac:.1%} |")
        drift = report.portfolio_summary.get("drift", {})
        drifted = {k: v for k, v in drift.items() if abs(v) > 0.01}
        if drifted:
            lines += ["", "| Asset class | Drift vs target |",
                      "|---|---|"]
            for cls, pp in sorted(drifted.items()):
                lines.append(f"| {cls} | {pp:+.1f}pp |")
        concentration = report.portfolio_summary.get("concentration", {})
        if concentration.get("symbol"):
            lines.append(
                f"- Top holding: {concentration['symbol']} "
                f"({concentration.get('fraction', 0):.1%} of value)"
            )

        lines += ["", "## Anomalies", ""]
        if report.anomalies:
            lines += ["| Severity | Rule | Description |",
                      "|---|---|---|"]
            for anomaly in report.anomalies:
                lines.append(
                    f"| {anomaly.severity} | {anomaly.rule} | "
                    f"{anomaly.description} |"
                )
        else:
            lines.append("No anomalies detected.")

        lines += ["", "## Recommendations", ""]
        if report.recommendations:
            lines += ["| Priority | Action | Rationale | Approval |",
                      "|---|---|---|---|"]
            for rec in report.recommendations:
                approval = "yes" if rec.requires_approval else "no"
                lines.append(
                    f"| {rec.priority} | {rec.action} | {rec.rationale} | "
                    f"{approval} |"
                )
        else:
            lines.append("No recommendations.")

        if report.crm_context:
            lines += ["", "## CRM context", ""]
            for key, value in report.crm_context.items():
                lines.append(f"- {key}: {value}")

        lines += ["", "## Narrative",
                  f"({report.narrative.summary_source})", "",
                  report.narrative.executive_summary, ""]
        if report.narrative.talking_points:
            lines.append("Talking points:")
            lines += [f"- {point}"
                      for point in report.narrative.talking_points]
        if report.narrative.caveats:
            lines += ["", "Caveats:"]
            lines += [f"- {caveat}" for caveat in report.narrative.caveats]

        lines += ["", "## Human review", "",
                  f"- Required: {report.human_review.required}",
                  f"- Decision: {report.human_review.decision}"]
        if report.human_review.note:
            lines.append(f"- Note: {report.human_review.note}")

        changes = report.changes_since_last_review
        if any(changes.values()):
            lines += ["", "## Changes since last review", ""]
            for key in ("new", "resolved", "worsened"):
                items = changes.get(key, [])
                lines.append(f"- {key}: "
                             + (", ".join(items) if items else "none"))

        if report.data_quality_warnings:
            lines += ["", "## Data-quality warnings", ""]
            lines += [f"- {w}" for w in report.data_quality_warnings]

        lines += ["", "## Trace", "",
                  "| Step | Agent | Status | Duration (ms) |",
                  "|---|---|---|---|"]
        for entry in report.trace:
            lines.append(
                f"| {entry.step} | {entry.agent} | {entry.status} | "
                f"{entry.duration_ms} |"
            )
        return "\n".join(lines)
