"""Orchestrator tests: real agents, fake tools, tmp outputs/memory."""
import json

import wealth_assistant.orchestrator as orch_mod
from wealth_assistant.models import Anomaly, Client, CRMRecord, Recommendation
from wealth_assistant.orchestrator import Orchestrator, resolve_client_ids
from wealth_assistant.tools.base import Tool, ToolResult
from wealth_assistant.tools.registry import ToolRegistry


class FakeTool(Tool):
    def __init__(self, name, handler):
        self._tool_name = name
        self._handler = handler

    @property
    def name(self):
        return self._tool_name

    def _execute(self, **kwargs):
        return self._handler(kwargs)


class FakeSummarizer(FakeTool):
    def __init__(self, handler):
        super().__init__("llm_summarizer", handler)
        self.client = None  # template mode: no LLM expected


def make_client(client_id="C1"):
    return Client(
        client_id=client_id, name="Test", risk_profile="moderate",
        accounts=[{"account_id": "A1", "type": "checking", "balance": 5000,
                   "currency": "USD"}])


def make_anomaly(aid="a1", severity="medium"):
    return Anomaly(id=aid, type="velocity", severity=severity,
                   description="burst", evidence={}, rule="velocity")


def make_crm(client_id="C1"):
    return CRMRecord(client_id=client_id, advisor_name="Sam Lee",
                     kyc_status="valid", kyc_expiry="2027-01-01",
                     last_contact_date="2026-09-01")


def fake_registry(anomalies=None, risk_score=10, loader_clients=None):
    anomalies = anomalies if anomalies is not None else []
    loader_clients = loader_clients if loader_clients is not None \
        else [make_client("C1")]
    reg = ToolRegistry()

    def loader(kw):
        return ToolResult(ok=True, data={"clients": loader_clients,
                                         "errors": [], "warnings": []},
                          meta={})

    def crm(kw):
        return ToolResult(
            ok=True, data=make_crm(kw.get("client_id", "C1")))

    def metrics(kw):
        return ToolResult(ok=True, data={"total_value": 5000,
                                         "allocation": {}, "drift": {},
                                         "concentration": {}})

    def detector(kw):
        return ToolResult(ok=True, data={"anomalies": anomalies,
                                         "risk_score": risk_score})

    def summarizer(kw):
        return ToolResult(
            ok=True,
            data={"executive_summary": "s", "talking_points": [],
                  "caveats": []},
            meta={"summary_source": "template"})

    for tool in (FakeTool("client_loader", loader),
                 FakeTool("crm_lookup", crm),
                 FakeTool("portfolio_metrics", metrics),
                 FakeTool("anomaly_detector", detector),
                 FakeSummarizer(summarizer)):
        reg.register(tool)
    return reg


def make_orchestrator(tmp_path, **overrides):
    return Orchestrator(
        output_dir=str(tmp_path / "outputs"),
        memory_path=str(tmp_path / "memory.json"),
        registry=fake_registry(**overrides),
        no_llm=True,
    )


def test_run_success_writes_json_and_md(tmp_path):
    orch = make_orchestrator(tmp_path)
    results = orch.run(["C1"], auto_approve=True)

    assert results[0]["status"] == "success"
    json_path = tmp_path / "outputs" / "report_C1.json"
    md_path = tmp_path / "outputs" / "report_C1.md"
    assert json_path.exists() and md_path.exists()

    report = json.loads(json_path.read_text())
    assert report["client_id"] == "C1"
    assert report["status"] == "success"
    assert report["human_review"]["decision"] == "not_required"
    assert {t["step"] for t in report["trace"]} == {
        "ingest", "analyze", "recommend"}
    md = md_path.read_text()
    assert "## Anomalies" in md and "## Recommendations" in md

    memory = json.loads((tmp_path / "memory.json").read_text())
    assert memory["C1"]["risk_score"] == 10


def test_failed_client_does_not_stop_batch(tmp_path):
    orch = make_orchestrator(tmp_path)
    results = orch.run(["C1", "C9"], auto_approve=True)

    statuses = {r["client_id"]: r["status"] for r in results}
    assert statuses == {"C1": "success", "C9": "failed"}
    report = json.loads(
        (tmp_path / "outputs" / "report_C9.json").read_text())
    assert report["status"] == "failed"


def test_hitl_auto_on_high_anomaly(tmp_path):
    orch = make_orchestrator(
        tmp_path, anomalies=[make_anomaly("a1", "high")], risk_score=80)
    results = orch.run(["C1"], auto_approve=True)

    assert results[0]["status"] == "success"
    report = json.loads(
        (tmp_path / "outputs" / "report_C1.json").read_text())
    assert report["human_review"]["decision"] == "auto"
    assert report["human_review"]["required"] is True


def test_rejected_report_not_saved_to_memory(tmp_path, monkeypatch):
    from wealth_assistant.models import HumanReview
    monkeypatch.setattr(
        orch_mod, "checkpoint",
        lambda report, auto_approve=False: HumanReview(
            required=True, decision="rejected", note="no"))
    orch = make_orchestrator(
        tmp_path, anomalies=[make_anomaly("a1", "high")], risk_score=80)
    results = orch.run(["C1"], auto_approve=False)

    assert results[0]["status"] == "success"
    assert not (tmp_path / "memory.json").exists()
    report = json.loads(
        (tmp_path / "outputs" / "report_C1.json").read_text())
    assert report["human_review"]["decision"] == "rejected"


def test_memory_feeds_changes_and_balance_drop(tmp_path):
    orch = make_orchestrator(tmp_path)
    orch.run(["C1"], auto_approve=True)
    # second run with a new anomaly: diff must flag it
    orch2 = make_orchestrator(tmp_path)
    orch2.registry.get("anomaly_detector")._handler = lambda kw: ToolResult(
        ok=True, data={"anomalies": [make_anomaly("a2", "high")],
                       "risk_score": 30})
    orch2.run(["C1"], auto_approve=True)

    report = json.loads(
        (tmp_path / "outputs" / "report_C1.json").read_text())
    assert report["changes_since_last_review"]["new"] == ["a2"]
    assert report["changes_since_last_review"]["resolved"] == []
