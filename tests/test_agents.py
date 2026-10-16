"""Agent tests. Fake tools are registered in a real ToolRegistry."""
from pathlib import Path

from wealth_assistant.agents.advisor import AdvisorAgent
from wealth_assistant.agents.analyzer import AnalyzerAgent
from wealth_assistant.agents.data_fetcher import DataFetcherAgent
from wealth_assistant.context import RunContext
from wealth_assistant.models import Anomaly, Client, CRMRecord
from wealth_assistant.tools.base import Tool, ToolResult
from wealth_assistant.tools.registry import ToolRegistry


class FakeTool(Tool):
    """Scriptable fake: records calls, returns canned ToolResults."""

    def __init__(self, name, handler):
        self._tool_name = name
        self._handler = handler
        self.calls = []

    @property
    def name(self):
        return self._tool_name

    def _execute(self, **kwargs):
        self.calls.append(kwargs)
        return self._handler(kwargs)


class FakeSummarizer(FakeTool):
    """Fake llm_summarizer with a client attribute like the real tool."""

    def __init__(self, handler, client=None):
        super().__init__("llm_summarizer", handler)
        self.client = client


def make_client(client_id="C1", **overrides):
    base = {
        "client_id": client_id, "name": "Test", "risk_profile": "moderate",
        "monthly_income": 8000,
        "accounts": [{"account_id": "A1", "type": "checking",
                      "balance": 10000, "currency": "USD"}],
        "holdings": [],
        "transactions": [
            {"id": "T1", "date": "2026-09-01", "amount": 100.0,
             "type": "debit", "category": "groceries", "merchant": "S",
             "account_id": "A1"},
        ],
    }
    base.update(overrides)
    return Client(**base)


def make_crm(client_id="C1"):
    return CRMRecord(client_id=client_id, advisor_name="Sam Lee",
                     kyc_status="valid", kyc_expiry="2027-01-01",
                     last_contact_date="2026-09-01")


def make_anomaly(rule, severity="medium", evidence=None, aid="a1"):
    return Anomaly(id=aid, type=rule, severity=severity,
                   description=f"{rule} found", evidence=evidence or {},
                   rule=rule)


def registry_with(**tools):
    reg = ToolRegistry()
    for tool in tools.values():
        reg.register(tool)
    return reg


# --- DataFetcherAgent ----------------------------------------------------------

def _loader_handler(clients, warnings=()):
    def handler(kwargs):
        return ToolResult(
            ok=True,
            data={"clients": clients, "errors": [], "warnings": list(warnings)},
            meta={},
        )

    return handler


def test_fetch_ok_records_client_crm_and_warnings():
    loader = FakeTool("client_loader", _loader_handler(
        [make_client("C1"), make_client("C2")],
        [{"client_id": "C1", "missing_fields": ["age"]}],
    ))
    crm = FakeTool("crm_lookup",
                   lambda kw: ToolResult(ok=True, data=make_crm("C1")))
    agent = DataFetcherAgent(registry_with(loader=loader, crm=crm))
    ctx = RunContext(client_id="C1")

    result = agent.run(ctx)

    assert result.status == "ok"
    assert ctx.get("client").client_id == "C1"
    assert ctx.get("crm").client_id == "C1"
    assert any("age" in w for w in ctx.warnings)
    assert loader.calls and crm.calls


def test_fetch_crm_failure_degrades():
    loader = FakeTool("client_loader", _loader_handler([make_client("C1")]))
    crm = FakeTool("crm_lookup",
                   lambda kw: ToolResult(ok=False, error="timeout"))
    agent = DataFetcherAgent(registry_with(loader=loader, crm=crm))
    ctx = RunContext(client_id="C1")

    result = agent.run(ctx)

    assert result.status == "degraded"
    assert ctx.get("crm") is None
    assert any("CRM lookup failed" in w for w in ctx.warnings)


def test_fetch_crm_not_found_degrades():
    loader = FakeTool("client_loader", _loader_handler([make_client("C1")]))
    crm = FakeTool("crm_lookup",
                   lambda kw: ToolResult(ok=True, data=None))
    agent = DataFetcherAgent(registry_with(loader=loader, crm=crm))
    ctx = RunContext(client_id="C1")

    result = agent.run(ctx)

    assert result.status == "degraded"
    assert any("no CRM record" in w for w in ctx.warnings)


def test_fetch_unknown_client_fails():
    loader = FakeTool("client_loader", _loader_handler([make_client("C1")]))
    crm = FakeTool("crm_lookup",
                   lambda kw: ToolResult(ok=True, data=None))
    agent = DataFetcherAgent(registry_with(loader=loader, crm=crm))

    result = agent.run(RunContext(client_id="C9"))

    assert result.status == "failed"


def test_fetch_loader_failure_fails():
    loader = FakeTool("client_loader",
                      lambda kw: ToolResult(ok=False, error="no file"))
    agent = DataFetcherAgent(registry_with(loader=loader))
    ctx = RunContext(client_id="C1")

    result = agent.run(ctx)

    assert result.status == "failed"


def test_fetch_loads_client_list_once():
    loader = FakeTool("client_loader", _loader_handler([make_client("C1")]))
    crm = FakeTool("crm_lookup",
                   lambda kw: ToolResult(ok=True, data=make_crm("C1")))
    agent = DataFetcherAgent(registry_with(loader=loader, crm=crm))
    ctx = RunContext(client_id="C1")

    agent.run(ctx)
    agent.run(ctx)

    assert len(loader.calls) == 1  # cached in the context


# --- AnalyzerAgent ---------------------------------------------------------------

def _analyzer_registry(metrics_ok=True, detect_ok=True):
    metrics = FakeTool(
        "portfolio_metrics",
        lambda kw: ToolResult(
            ok=metrics_ok, data={"total_value": 10000} if metrics_ok else None,
            error=None if metrics_ok else "boom"),
    )
    detector = FakeTool(
        "anomaly_detector",
        lambda kw: ToolResult(
            ok=detect_ok,
            data={"anomalies": [], "risk_score": 5} if detect_ok else None,
            error=None if detect_ok else "boom"),
    )
    return registry_with(metrics=metrics, detector=detector)


def test_analyze_ok_writes_context():
    agent = AnalyzerAgent(_analyzer_registry())
    ctx = RunContext(client_id="C1")
    ctx.set("client", make_client("C1"))

    result = agent.run(ctx)

    assert result.status == "ok"
    assert ctx.get("portfolio_summary") == {"total_value": 10000}
    assert ctx.get("anomalies") == []
    assert ctx.get("risk_score") == 5


def test_analyze_no_transactions_warns():
    agent = AnalyzerAgent(_analyzer_registry())
    ctx = RunContext(client_id="C1")
    ctx.set("client", make_client("C1", transactions=[]))

    result = agent.run(ctx)

    assert result.status == "ok"
    assert any("no transactions" in w for w in ctx.warnings)


def test_analyze_no_client_fails():
    agent = AnalyzerAgent(_analyzer_registry())

    result = agent.run(RunContext(client_id="C1"))

    assert result.status == "failed"


def test_analyze_metrics_failure_fails():
    agent = AnalyzerAgent(_analyzer_registry(metrics_ok=False))
    ctx = RunContext(client_id="C1")
    ctx.set("client", make_client("C1"))

    result = agent.run(ctx)

    assert result.status == "failed"


# --- AdvisorAgent ------------------------------------------------------------------

def _advisor_anomalies():
    return [
        make_anomaly("allocation_drift", "medium",
                     {"asset_class": "equity"}, "a1"),
        make_anomaly("concentration", "medium", {"symbol": "AAPL"}, "a2"),
        make_anomaly("profile_mismatch", "high", {}, "a3"),
        make_anomaly("crm_kyc", "high",
                     {"expired": True, "kyc_expiry": "2024-01-01"}, "a4"),
        make_anomaly("crm_stale_contact", "low",
                     {"days_since_contact": 200}, "a5"),
    ]


def _advisor_registry(summarizer):
    return registry_with(summarizer=summarizer)


def _advisor_context():
    ctx = RunContext(client_id="C1")
    ctx.set("anomalies", _advisor_anomalies())
    ctx.set("risk_score", 80)
    ctx.set("portfolio_summary", {"total_value": 50000})
    ctx.set("crm", make_crm("C1"))
    return ctx


def _ok_summarizer(client=None, source="llm:fake:model"):
    narrative = {"executive_summary": "s", "talking_points": ["t"],
                 "caveats": ["c"]}
    return FakeSummarizer(
        lambda kw: ToolResult(ok=True, data=narrative,
                              meta={"summary_source": source}),
        client=client,
    )


def test_recommendation_mapping():
    agent = AdvisorAgent(_advisor_registry(_ok_summarizer()))
    ctx = _advisor_context()

    result = agent.run(ctx)

    assert result.status == "ok"
    recs = {r.action: r for r in ctx.get("recommendations")}
    assert recs["Rebalance portfolio toward target allocation"].priority == "medium"
    assert recs["Rebalance portfolio toward target allocation"].requires_approval is True
    assert recs["Reduce single-holding concentration"].priority == "high"
    assert recs["Review risk profile with client"].requires_approval is True
    urgent = recs["Refresh KYC documentation immediately"]
    assert urgent.priority == "urgent"
    assert urgent.requires_approval is False
    contact = recs["Contact client to verify flagged activity"]
    assert contact.priority == "high" and contact.requires_approval is True
    assert recs["Schedule advisor review meeting"].priority == "low"
    assert len(recs) == 6


def test_narrative_and_source_stored():
    agent = AdvisorAgent(_advisor_registry(_ok_summarizer()))
    ctx = _advisor_context()

    agent.run(ctx)

    assert ctx.get("narrative")["executive_summary"] == "s"
    assert ctx.get("summary_source") == "llm:fake:model"


def test_template_fallback_warns_only_when_llm_expected():
    agent = AdvisorAgent(
        _advisor_registry(_ok_summarizer(client=object(), source="template")))
    ctx = _advisor_context()

    result = agent.run(ctx)

    assert result.status == "ok"  # status unchanged
    assert any("template fallback" in w for w in ctx.warnings)


def test_template_without_llm_expected_no_warning():
    agent = AdvisorAgent(_advisor_registry(_ok_summarizer(client=None,
                                                         source="template")))
    ctx = _advisor_context()

    result = agent.run(ctx)

    assert result.status == "ok"
    assert ctx.warnings == []


def test_advisor_does_not_import_llm_package():
    source = Path(__file__).parent.parent.joinpath(
        "src", "wealth_assistant", "agents", "advisor.py").read_text()
    assert "from ..llm" not in source
    assert "from wealth_assistant.llm" not in source
    assert "import llm" not in source
