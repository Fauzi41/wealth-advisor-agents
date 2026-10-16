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


# ======================================================================
# Phase 7: end-to-end tests with the REAL registry, tools and data files.
# No network access, no API keys: no_llm=True, CRM_FAILURE_MODE via env.
# ======================================================================
import sys
from pathlib import Path

from wealth_assistant.llm.base import LLMClient, LLMResponse
from wealth_assistant.memory import LongTermMemory
from wealth_assistant.tools.llm_summarizer import LLMSummarizerTool

DATA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "clients.json")


def real_orchestrator(tmp_path, name="e2e", **kwargs):
    """Orchestrator with the real default registry and real data files."""
    return Orchestrator(
        data_path=DATA_PATH,
        output_dir=str(tmp_path / name / "outputs"),
        memory_path=str(tmp_path / name / "memory.json"),
        no_llm=True,
        **kwargs,
    )


def read_report(tmp_path, name, client_id):
    return json.loads(
        (tmp_path / name / "outputs" / f"report_{client_id}.json").read_text())


def test_e2e_c001_happy_path(tmp_path):
    orch = real_orchestrator(tmp_path, "t1")
    results = orch.run(["C001"], auto_approve=True)

    assert results[0]["status"] == "success"
    report = read_report(tmp_path, "t1", "C001")
    assert report["status"] == "success"
    assert report["risk_score"] == 0
    assert report["anomalies"] == []
    assert report["human_review"]["required"] is False
    assert report["human_review"]["decision"] == "not_required"
    assert (tmp_path / "t1" / "outputs" / "report_C001.md").exists()


def test_e2e_c002_anomaly_path(tmp_path):
    orch = real_orchestrator(tmp_path, "t2")
    results = orch.run(["C002"], auto_approve=True)

    assert results[0]["status"] == "success"
    report = read_report(tmp_path, "t2", "C002")
    rules = {a["rule"] for a in report["anomalies"]}
    assert {"outlier_transaction", "duplicates", "velocity",
            "round_number_withdrawal", "new_merchant_large",
            "large_vs_income"} <= rules
    assert any(a["severity"] == "high" for a in report["anomalies"])
    assert report["human_review"]["required"] is True
    assert report["human_review"]["decision"] == "auto"


def test_e2e_crm_timeout_degraded(tmp_path, monkeypatch):
    monkeypatch.setenv("CRM_FAILURE_MODE", "timeout")
    monkeypatch.setattr(
        "wealth_assistant.tools.crm.time.sleep", lambda s: None)
    orch = real_orchestrator(tmp_path, "t3")
    results = orch.run(["C001"], auto_approve=True)

    assert results[0]["status"] == "degraded"
    report = read_report(tmp_path, "t3", "C001")
    assert report["status"] == "degraded"
    # pipeline still completes all three steps
    assert {t["step"] for t in report["trace"]} == {
        "ingest", "analyze", "recommend"}
    assert any("CRM" in w for w in report["data_quality_warnings"])


def test_e2e_malformed_c005_failed_batch_continues(tmp_path):
    orch = real_orchestrator(tmp_path, "t4")
    results = orch.run(["C005", "C001"], auto_approve=True)

    by_id = {r["client_id"]: r["status"] for r in results}
    assert by_id == {"C005": "failed", "C001": "success"}
    bad = read_report(tmp_path, "t4", "C005")
    assert bad["status"] == "failed"
    assert bad["trace"][0]["step"] == "ingest"
    assert bad["trace"][0]["status"] == "failed"


def test_e2e_c004_no_crm_no_transactions(tmp_path):
    orch = real_orchestrator(tmp_path, "t5")
    results = orch.run(["C004"], auto_approve=True)

    assert results[0]["status"] == "degraded"
    report = read_report(tmp_path, "t5", "C004")
    assert report["status"] == "degraded"
    assert report["anomalies"] == []
    assert report["risk_score"] == 0
    assert report["human_review"]["decision"] == "not_required"


def test_e2e_hitl_override_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = iter(["o", "1", "p", "low", "downgraded after verification call"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    orch = real_orchestrator(tmp_path, "t6")
    orch.run(["C002"], auto_approve=False)

    report = read_report(tmp_path, "t6", "C002")
    assert report["human_review"]["required"] is True
    assert report["human_review"]["decision"] == "overridden"
    assert "downgraded after verification call" in report["human_review"]["note"]
    assert len(report["recommendations"]) == 1
    assert report["recommendations"][0]["priority"] == "low"


def test_e2e_second_run_changes_populated(tmp_path):
    orch = real_orchestrator(tmp_path, "t7")
    orch.run(["C002"], auto_approve=True)
    first = read_report(tmp_path, "t7", "C002")
    first_ids = {a["id"] for a in first["anomalies"]}
    dup_id = next(a["id"] for a in first["anomalies"]
                  if a["rule"] == "duplicates")
    vel_id = next(a["id"] for a in first["anomalies"]
                  if a["rule"] == "velocity")
    # first run: everything is new
    assert first["changes_since_last_review"]["new"] == sorted(first_ids)

    # simulate an older, different history in memory (same instance the
    # orchestrator uses, as in a real single-process run)
    orch.memory.save_summary("C002", {
        "risk_score": 20,
        "anomalies": [
            {"type": "velocity", "id": vel_id, "severity": "low"},
            {"type": "duplicates", "id": dup_id, "severity": "medium"},
            {"type": "old_rule", "id": "old:1", "severity": "low"},
        ],
        "recommendations": [],
        "total_balance": 12000.0,
    })

    orch.run(["C002"], auto_approve=True)
    second = read_report(tmp_path, "t7", "C002")
    changes = second["changes_since_last_review"]
    assert changes["resolved"] == ["old:1"]
    assert changes["worsened"] == [vel_id]  # low -> high
    assert changes["new"] == sorted(first_ids - {vel_id, dup_id})


class FakeLLMClient(LLMClient):
    """Offline LLM double returning a valid JSON summary."""

    def __init__(self):
        self.provider = "fake"
        self.model = "fake-model"
        self.calls = 0

    def complete(self, system, user, max_tokens):
        self.calls += 1
        return LLMResponse(
            text='{"executive_summary": "LLM summary.", '
                 '"talking_points": ["t1"], "caveats": ["c1"]}',
            provider=self.provider, model=self.model, latency_ms=3,
            input_tokens=5, output_tokens=7)

    def ping(self):
        return True


def _deterministic_fields(report):
    return {
        "anomalies": report["anomalies"],
        "risk_score": report["risk_score"],
        "recommendations": report["recommendations"],
    }


def test_e2e_llm_vs_template_deterministic_fields_identical(tmp_path):
    orch_llm = real_orchestrator(tmp_path, "t8a")
    fake = FakeLLMClient()
    orch_llm.registry.register(LLMSummarizerTool(fake))
    orch_llm.run(["C002"], auto_approve=True)
    llm_report = read_report(tmp_path, "t8a", "C002")
    assert fake.calls == 1
    assert llm_report["narrative"]["summary_source"].startswith("llm:")

    orch_tpl = real_orchestrator(tmp_path, "t8b")
    orch_tpl.run(["C002"], auto_approve=True)
    tpl_report = read_report(tmp_path, "t8b", "C002")
    assert tpl_report["narrative"]["summary_source"] == "template"

    assert _deterministic_fields(llm_report) == _deterministic_fields(tpl_report)
