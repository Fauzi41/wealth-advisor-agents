"""Prompts for LLM narrative generation.

The LLM only writes prose from structured findings. Only the client_id and
aggregated findings are sent -- never the client name and never raw
transactions.
"""
from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT = """You are a writing assistant to a human wealth advisor. Your job is to turn
structured portfolio findings into a short, plain-language narrative for the
advisor to use with their client.

Hard rules:
- Use ONLY the findings provided below. Never invent numbers, holdings,
  transactions, names, dates, or events.
- If a figure is not in the findings, say it is unavailable. Do not estimate.
- Never promise, imply, or suggest guaranteed returns or future performance.
- State uncertainty explicitly. Distinguish observed facts from interpretation.
- Keep it concise and professional. Explain jargon instead of assuming it.
- Write for the advisor, not the end client ("discuss with the client...",
  "consider raising...").

Output:
- Respond with JSON ONLY. No markdown fences, no commentary, no extra keys.
- Schema: {"executive_summary": str, "talking_points": [str], "caveats": [str]}
- executive_summary: 3-5 sentences.
- talking_points: short bullets the advisor can raise in the meeting.
- caveats: limitations, data-quality issues, and uncertainties the advisor
  must keep in mind."""


def _severity(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("severity", ""))
    return str(getattr(item, "severity", ""))


def _rule(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("rule", item.get("type", "")))
    return str(getattr(item, "rule", ""))


def _description(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("description", ""))
    return str(getattr(item, "description", ""))


def _anomaly_lines(anomalies: List[Any]) -> List[str]:
    return [
        f"- [{_severity(a) or 'n/a'}] {_rule(a)}: {_description(a)}"
        for a in anomalies
    ]


def build_user_prompt(findings: Dict[str, Any]) -> str:
    """Render the user prompt from structured findings.

    Expected keys (all optional): client_id, risk_score, anomalies (Anomaly
    models or dicts), portfolio_summary {allocation, drift, concentration},
    recommendations, crm_flags, data_quality_warnings. The client name and
    raw transactions are never included.
    """
    lines = [
        "Write the narrative for the following portfolio findings.",
        "",
        f"Client ID: {findings.get('client_id', 'unknown')}",
        f"Risk score: {findings.get('risk_score', 'n/a')}/100",
        "",
    ]

    anomalies = findings.get("anomalies") or []
    lines.append(f"Anomalies ({len(anomalies)}):")
    lines.extend(_anomaly_lines(anomalies) or ["- none"])
    lines.append("")

    summary = findings.get("portfolio_summary") or {}
    allocation = summary.get("allocation") or {}
    if allocation:
        alloc_str = ", ".join(
            f"{cls}: {frac:.0%}" for cls, frac in allocation.items()
        )
        lines.append(f"Allocation: {alloc_str}")
    drift = summary.get("drift") or {}
    drifted = {k: v for k, v in drift.items() if abs(v) > 0.01}
    if drifted:
        drift_str = ", ".join(f"{cls}: {v:+.1f}pp" for cls, v in drifted.items())
        lines.append(f"Allocation drift vs target: {drift_str}")
    concentration = summary.get("concentration") or {}
    if concentration.get("symbol"):
        lines.append(
            f"Top holding concentration: {concentration['symbol']} "
            f"({concentration.get('fraction', 0):.0%} of value)"
        )
    lines.append("")

    recommendations = findings.get("recommendations") or []
    lines.append(f"Recommendations ({len(recommendations)}):")
    for rec in recommendations:
        if isinstance(rec, dict):
            lines.append(
                f"- [{rec.get('priority', 'n/a')}] {rec.get('action', '')}: "
                f"{rec.get('rationale', '')}"
            )
        else:
            lines.append(
                f"- [{getattr(rec, 'priority', 'n/a')}] "
                f"{getattr(rec, 'action', '')}: {getattr(rec, 'rationale', '')}"
            )
    if not recommendations:
        lines.append("- none")
    lines.append("")

    crm_flags = findings.get("crm_flags") or {}
    if crm_flags:
        lines.append("CRM flags:")
        for key, value in crm_flags.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    warnings = findings.get("data_quality_warnings") or []
    if warnings:
        lines.append("Data-quality warnings:")
        lines.extend(f"- {w}" for w in warnings)
        lines.append("")

    lines.append(
        'Respond with JSON only, no markdown fences: '
        '{"executive_summary": str, "talking_points": [str], "caveats": [str]}'
    )
    return "\n".join(lines)
