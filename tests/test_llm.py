"""LLM layer tests. No network: a FakeLLMClient stands in for providers."""
import pytest

from wealth_assistant.llm import factory
from wealth_assistant.llm.base import LLMClient, LLMError, LLMResponse
from wealth_assistant.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from wealth_assistant.tools.llm_summarizer import LLMSummarizerTool

VALID_JSON = (
    '{"executive_summary": "Client shows elevated risk.", '
    '"talking_points": ["Discuss concentration"], "caveats": ["Limited history"]}'
)


class FakeLLMClient(LLMClient):
    """Scripted fake: queued responses, optional error, controllable ping."""

    def __init__(self, responses=None, error=None, ping_ok=True):
        self.provider = "fake"
        self.model = "fake-model"
        self._responses = list(responses or [])
        self._error = error
        self._ping_ok = ping_ok
        self.calls = []

    def complete(self, system, user, max_tokens):
        self.calls.append({"system": system, "user": user,
                           "max_tokens": max_tokens})
        if self._error is not None:
            raise self._error
        text = self._responses.pop(0) if self._responses else "{}"
        return LLMResponse(text=text, provider=self.provider,
                           model=self.model, latency_ms=5,
                           input_tokens=10, output_tokens=20)

    def ping(self):
        return self._ping_ok


@pytest.fixture()
def findings():
    return {
        "client_id": "C001",
        "risk_score": 42,
        "anomalies": [
            {"id": "a1", "type": "concentration", "severity": "medium",
             "description": "Holding AAPL is 60% of portfolio value.",
             "rule": "concentration"},
        ],
        "portfolio_summary": {
            "allocation": {"equity": 0.6, "bond": 0.4},
            "drift": {"equity": 0.0},
            "concentration": {"symbol": "AAPL", "fraction": 0.6},
        },
        "recommendations": [
            {"action": "Rebalance", "rationale": "Reduce concentration",
             "priority": "medium"},
        ],
        "crm_flags": {"kyc": "valid"},
        "data_quality_warnings": ["missing age"],
    }


@pytest.fixture(autouse=True)
def _fresh_llm_cache():
    factory.reset_llm_client_cache()
    yield
    factory.reset_llm_client_cache()


# --- summarizer ---------------------------------------------------------------

def test_valid_json_narrative(findings):
    tool = LLMSummarizerTool(FakeLLMClient(responses=[VALID_JSON]))
    result = tool.run(findings=findings)
    assert result.ok is True
    assert result.data["executive_summary"] == "Client shows elevated risk."
    assert result.data["talking_points"] == ["Discuss concentration"]
    assert result.meta["summary_source"] == "llm:fake:fake-model"
    assert result.meta["latency_ms"] == 5
    assert result.meta["input_tokens"] == 10


def test_json_in_code_fences_is_stripped(findings):
    fenced = "```json\n" + VALID_JSON + "\n```"
    tool = LLMSummarizerTool(FakeLLMClient(responses=[fenced]))
    result = tool.run(findings=findings)
    assert result.ok is True
    assert result.data["executive_summary"] == "Client shows elevated risk."


def test_invalid_then_valid_on_retry(findings):
    fake = FakeLLMClient(responses=["not json at all", VALID_JSON])
    tool = LLMSummarizerTool(fake)
    result = tool.run(findings=findings)
    assert result.ok is True
    assert result.meta["summary_source"] == "llm:fake:fake-model"
    assert len(fake.calls) == 2
    assert "valid JSON only" in fake.calls[1]["user"]


def test_invalid_twice_falls_back_to_template(findings):
    fake = FakeLLMClient(responses=["garbage", "still garbage"])
    tool = LLMSummarizerTool(fake)
    result = tool.run(findings=findings)
    assert result.ok is True
    assert result.meta["summary_source"] == "template"
    assert "C001" in result.data["executive_summary"]
    assert "42/100" in result.data["executive_summary"]


def test_llm_error_falls_back_to_template(findings):
    tool = LLMSummarizerTool(FakeLLMClient(error=LLMError("boom")))
    result = tool.run(findings=findings)
    assert result.ok is True
    assert result.meta["summary_source"] == "template"
    assert result.meta["reason"].startswith("llm_error")


def test_no_client_uses_template(findings):
    tool = LLMSummarizerTool(None)
    result = tool.run(findings=findings)
    assert result.ok is True
    assert result.meta["summary_source"] == "template"
    assert result.meta["reason"] == "no_llm_client"


def test_tool_never_raises_on_unexpected_error(findings):
    class Exploding(FakeLLMClient):
        def complete(self, system, user, max_tokens):
            raise RuntimeError("unexpected")

    result = LLMSummarizerTool(Exploding()).run(findings=findings)
    assert result.ok is False
    assert "llm_summarizer failed" in result.error


# --- factory -------------------------------------------------------------------

def _set_llm_env(monkeypatch, provider="anthropic", key="sk-test",
                 model="test-model"):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setenv("LLM_MODEL", model)


def test_factory_none_when_key_missing(monkeypatch, caplog):
    _set_llm_env(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with caplog.at_level("WARNING"):
        assert factory.build_llm_client() is None
    assert any("ANTHROPIC_API_KEY" in r.message for r in caplog.records)
    assert not any("sk-test" in r.message for r in caplog.records)


def test_factory_none_when_ping_fails(monkeypatch):
    _set_llm_env(monkeypatch)

    class DeadClient:
        def __init__(self, api_key, model):
            pass

        def ping(self):
            return False

    monkeypatch.setitem(factory._CLIENTS, "anthropic",
                        ("ANTHROPIC_API_KEY", DeadClient))
    assert factory.build_llm_client() is None


def test_factory_none_for_unknown_provider(monkeypatch):
    _set_llm_env(monkeypatch, provider="mystery")
    assert factory.build_llm_client() is None


def test_factory_none_when_disabled(monkeypatch):
    _set_llm_env(monkeypatch, provider="none")
    assert factory.build_llm_client() is None


def test_factory_caches_result(monkeypatch):
    _set_llm_env(monkeypatch)

    class FakeClient:
        def __init__(self, api_key, model):
            self.api_key = api_key

        def ping(self):
            return True

    monkeypatch.setitem(factory._CLIENTS, "anthropic",
                        ("ANTHROPIC_API_KEY", FakeClient))
    first = factory.build_llm_client()
    second = factory.build_llm_client()
    assert first is second
    assert first.api_key == "sk-test"


# --- prompts ---------------------------------------------------------------------

def test_system_prompt_guardrails():
    text = SYSTEM_PROMPT.lower()
    assert "only" in text and "provided" in text
    assert "guaranteed" in text
    assert "json only" in text


def test_user_prompt_includes_findings_excludes_pii(findings):
    findings = dict(findings, name="Jane Doe",
                    transactions=[{"id": "T1", "amount": 999}])
    prompt = build_user_prompt(findings)
    assert "42/100" in prompt
    assert "concentration" in prompt
    assert "AAPL" in prompt
    assert "Rebalance" in prompt
    assert "Jane Doe" not in prompt
    assert '"T1"' not in prompt and "999" not in prompt
