"""Human-in-the-loop checkpoint before a report is finalized."""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List

from . import config
from .models import HumanReview

_logger = logging.getLogger(__name__)

_VALID_PRIORITIES = ("low", "medium", "high", "urgent")


def _severity(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("severity", ""))
    return str(getattr(item, "severity", ""))


def _describe_anomaly(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("description", item.get("rule", "")))
    return str(getattr(item, "description", ""))


def _priority(rec: Any) -> str:
    if isinstance(rec, dict):
        return str(rec.get("priority", ""))
    return str(getattr(rec, "priority", ""))


def _action(rec: Any) -> str:
    if isinstance(rec, dict):
        return str(rec.get("action", ""))
    return str(getattr(rec, "action", ""))


def _set_priority(rec: Any, value: str) -> None:
    if isinstance(rec, dict):
        rec["priority"] = value
    else:
        rec.priority = value


def review_required(report: Dict[str, Any]) -> bool:
    """A human must review when any anomaly is high severity or risk >= 70."""
    anomalies = report.get("anomalies", []) or []
    if any(_severity(a) == "high" for a in anomalies):
        return True
    return (report.get("risk_score", 0) or 0) >= config.HITL_RISK_THRESHOLD


def checkpoint(
    report: Dict[str, Any], auto_approve: bool = False
) -> HumanReview:
    """Run the HITL checkpoint for a report dict.

    Returns a HumanReview. When not required, decision is "not_required".
    With auto_approve (or a non-TTY stdin), decision is "auto".
    An override mutates ``report["recommendations"]`` in place.
    """
    if not review_required(report):
        return HumanReview(required=False, decision="not_required", note="")

    if auto_approve:
        return HumanReview(
            required=True, decision="auto", note="auto-approved (flag)"
        )
    if not sys.stdin.isatty():
        _logger.warning(
            "hitl_non_tty: stdin is not a TTY; auto-approving review for %s",
            report.get("client_id", "?"),
        )
        return HumanReview(
            required=True, decision="auto", note="auto-approved (non-TTY)"
        )
    return _interactive_checkpoint(report)


def _interactive_checkpoint(report: Dict[str, Any]) -> HumanReview:
    """Prompt the reviewer on the terminal."""
    print("\n=== Human review required ===")
    print(
        f"Client: {report.get('client_id', '?')}  "
        f"Risk: {report.get('risk_score', '?')}/100"
    )
    for anomaly in report.get("anomalies", []) or []:
        print(f"  [{_severity(anomaly)}] {_describe_anomaly(anomaly)}")
    print()

    while True:
        choice = input("[a]pprove / [o]verride / [r]eject > ").strip().lower()
        if choice == "a":
            return HumanReview(required=True, decision="approved", note="")
        if choice == "r":
            note = _required_input("Rejection note (required) > ")
            return HumanReview(required=True, decision="rejected", note=note)
        if choice == "o":
            return _interactive_override(report)
        print("Please answer a, o or r.")


def _required_input(prompt: str) -> str:
    """Prompt until a non-empty value is given."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("A note is mandatory.")


def _interactive_override(report: Dict[str, Any]) -> HumanReview:
    """Let the reviewer change a recommendation priority or remove it."""
    recommendations: List[Any] = report.get("recommendations", []) or []
    if not recommendations:
        print("No recommendations to override.")
        return HumanReview(
            required=True, decision="approved", note="override skipped: none"
        )
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. [{_priority(rec)}] {_action(rec)}")

    while True:
        raw = input("Recommendation number > ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(recommendations):
            index = int(raw) - 1
            break
        print(f"Enter a number between 1 and {len(recommendations)}.")

    while True:
        op = input("[p] change priority / [d] delete > ").strip().lower()
        if op in ("p", "d"):
            break
        print("Please answer p or d.")

    if op == "p":
        while True:
            new_priority = (
                input(f"New priority {_VALID_PRIORITIES} > ").strip().lower()
            )
            if new_priority in _VALID_PRIORITIES:
                break
            print(f"Priority must be one of {_VALID_PRIORITIES}.")
        _set_priority(recommendations[index], new_priority)
        action = f"priority of recommendation {index + 1} -> {new_priority}"
    else:
        del recommendations[index]
        action = f"removed recommendation {index + 1}"

    note = _required_input("Override note (required) > ")
    return HumanReview(
        required=True, decision="overridden", note=f"{action}; {note}"
    )
