"""HITL checkpoint tests: gating, auto-approve, interactive flows."""
import sys

from wealth_assistant.hitl import checkpoint, review_required


def _report(**overrides):
    base = {"client_id": "C1", "risk_score": 10, "anomalies": [],
            "recommendations": []}
    base.update(overrides)
    return base


def _anomaly(severity):
    return {"id": "a1", "type": "t", "severity": severity,
            "description": "d", "evidence": {}, "rule": "t"}


def test_not_required_when_calm():
    review = checkpoint(_report())
    assert review.required is False
    assert review.decision == "not_required"


def test_required_on_high_severity():
    assert review_required(_report(anomalies=[_anomaly("high")])) is True


def test_required_on_risk_threshold():
    assert review_required(_report(risk_score=70)) is True
    assert review_required(_report(risk_score=69)) is False


def test_auto_approve():
    review = checkpoint(_report(risk_score=90), auto_approve=True)
    assert review.required is True
    assert review.decision == "auto"


def test_non_tty_behaves_like_auto(monkeypatch, caplog):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with caplog.at_level("WARNING"):
        review = checkpoint(_report(risk_score=90))
    assert review.decision == "auto"
    assert any("not a TTY" in r.message for r in caplog.records)


def _tty(monkeypatch, answers):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))


def test_interactive_approve(monkeypatch):
    _tty(monkeypatch, ["a"])
    review = checkpoint(_report(risk_score=90))
    assert review.decision == "approved"


def test_interactive_reject_requires_note(monkeypatch):
    _tty(monkeypatch, ["r", "", "too risky"])
    review = checkpoint(_report(risk_score=90))
    assert review.decision == "rejected"
    assert review.note == "too risky"


def test_interactive_override_priority(monkeypatch):
    report = _report(
        risk_score=90,
        recommendations=[{"action": "Rebalance", "rationale": "r",
                          "priority": "medium", "requires_approval": True}],
    )
    _tty(monkeypatch, ["o", "1", "p", "low", "client asked"])
    review = checkpoint(report)
    assert review.decision == "overridden"
    assert report["recommendations"][0]["priority"] == "low"
    assert "client asked" in review.note


def test_interactive_override_delete(monkeypatch):
    report = _report(
        risk_score=90,
        recommendations=[{"action": "Rebalance", "rationale": "r",
                          "priority": "medium", "requires_approval": True}],
    )
    _tty(monkeypatch, ["o", "1", "d", "not needed"])
    review = checkpoint(report)
    assert review.decision == "overridden"
    assert report["recommendations"] == []
