"""LLMSummarizerTool: narrative from structured findings, LLM or template."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from .. import config
from ..llm.base import LLMClient, LLMError, LLMResponse
from ..llm.prompts import SYSTEM_PROMPT, build_user_prompt
from .base import Tool, ToolResult

#: Nudge appended to the user prompt when the first LLM output is invalid.
RETRY_NUDGE = (
    "IMPORTANT: return valid JSON only, no markdown fences, no commentary."
)


class NarrativePayload(BaseModel):
    """Validated shape of the LLM's JSON narrative."""

    executive_summary: str = Field(min_length=1)
    talking_points: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)


def _severity(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("severity", ""))
    return str(getattr(item, "severity", ""))


def _description(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("description", ""))
    return str(getattr(item, "description", ""))


def template_summary(findings: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback narrative built only from findings."""
    anomalies = findings.get("anomalies") or []
    risk = findings.get("risk_score", 0)
    high = sum(1 for a in anomalies if _severity(a) == "high")
    summary = (
        f"Portfolio review for client {findings.get('client_id', 'unknown')}: "
        f"risk score {risk}/100 with {len(anomalies)} finding(s), "
        f"{high} of them high severity."
    )
    talking_points = [
        f"[{_severity(a).upper() or 'N/A'}] {_description(a)}" for a in anomalies[:5]
    ]
    caveats = [
        "Generated from a template without an LLM; all figures are "
        "deterministic outputs of the analysis."
    ]
    for warning in findings.get("data_quality_warnings") or []:
        caveats.append(f"Data quality: {warning}")
    return {
        "executive_summary": summary,
        "talking_points": talking_points,
        "caveats": caveats,
    }


def _strip_code_fences(text: str) -> str:
    """Remove surrounding markdown code fences, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_narrative(text: str) -> Optional[Dict[str, Any]]:
    """Parse and validate an LLM narrative; None when invalid."""
    try:
        data = json.loads(_strip_code_fences(text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return NarrativePayload(**data).model_dump()
    except ValidationError:
        return None


class LLMSummarizerTool(Tool):
    """Generate the report narrative from structured findings.

    With a client: calls the LLM, strips code fences, validates the JSON
    with pydantic, and retries once with a nudge on invalid output.
    Without a client, on LLMError, or when the output stays invalid: falls
    back to the deterministic template. Never raises.
    """

    name = "llm_summarizer"

    def __init__(self, client: Optional[LLMClient] = None) -> None:
        """Create the tool.

        Args:
            client: Configured LLM client, or None to always use the
                template fallback.
        """
        self.client = client

    def _template_result(
        self, findings: Dict[str, Any], reason: str
    ) -> ToolResult:
        return ToolResult(
            ok=True,
            data=template_summary(findings),
            meta={"summary_source": "template", "reason": reason},
        )

    def _execute(self, findings: Dict[str, Any]) -> ToolResult:
        if self.client is None:
            return self._template_result(findings, reason="no_llm_client")

        user_prompt = build_user_prompt(findings)
        try:
            response = self.client.complete(
                SYSTEM_PROMPT, user_prompt, max_tokens=config.LLM_MAX_TOKENS
            )
        except LLMError as exc:
            return self._template_result(findings, reason=f"llm_error: {exc}")

        payload = _parse_narrative(response.text)
        if payload is None and config.LLM_MAX_RETRIES >= 1:
            try:
                response = self.client.complete(
                    SYSTEM_PROMPT,
                    user_prompt + "\n\n" + RETRY_NUDGE,
                    max_tokens=config.LLM_MAX_TOKENS,
                )
            except LLMError as exc:
                return self._template_result(
                    findings, reason=f"llm_error_on_retry: {exc}"
                )
            payload = _parse_narrative(response.text)

        if payload is None:
            return self._template_result(findings, reason="invalid_llm_output")

        return ToolResult(
            ok=True,
            data=payload,
            meta={
                "summary_source": f"llm:{response.provider}:{response.model}",
                "latency_ms": response.latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
