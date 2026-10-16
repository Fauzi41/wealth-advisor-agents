"""CRM tool tests: retry behavior on transient failures, no-retry on not_found."""
import time

from wealth_assistant.models import CRMRecord
from wealth_assistant.tools.crm import CRMTool, MockCRMClient


def test_timeout_retries_three_times_then_fails(monkeypatch):
    monkeypatch.setenv("CRM_FAILURE_MODE", "timeout")
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    result = CRMTool().run(client_id="C001")
    assert result.ok is False
    assert "3 attempts" in result.error
    assert result.meta["attempts"] == 3
    assert sleeps == [0.5, 1.0]  # exponential backoff: 0.5 * 2**n


def test_server_error_retries_then_fails(monkeypatch):
    monkeypatch.setenv("CRM_FAILURE_MODE", "server_error")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    result = CRMTool().run(client_id="C001")
    assert result.ok is False
    assert result.meta["attempts"] == 3


def test_not_found_does_not_retry(monkeypatch):
    monkeypatch.setenv("CRM_FAILURE_MODE", "not_found")
    calls = []
    original = MockCRMClient.get_record

    def counting(self, client_id):
        calls.append(client_id)
        return original(self, client_id)

    monkeypatch.setattr(MockCRMClient, "get_record", counting)
    result = CRMTool().run(client_id="C001")
    assert result.ok is True
    assert result.data is None
    assert result.meta["found"] is False
    assert len(calls) == 1  # exactly one attempt, no retries


def test_success_returns_record(monkeypatch):
    monkeypatch.setenv("CRM_FAILURE_MODE", "none")
    result = CRMTool().run(client_id="C001")
    assert result.ok is True
    assert isinstance(result.data, CRMRecord)
    assert result.data.client_id == "C001"
    assert result.meta["found"] is True


def test_unknown_client_returns_none_not_error(monkeypatch):
    monkeypatch.setenv("CRM_FAILURE_MODE", "none")
    result = CRMTool().run(client_id="ZZZ")
    assert result.ok is True
    assert result.data is None
    assert result.meta["found"] is False


def test_tool_never_raises(monkeypatch):
    monkeypatch.setenv("CRM_FAILURE_MODE", "bogus")
    result = CRMTool().run(client_id="C001")
    assert result.ok is False
    assert "CRM_FAILURE_MODE" in result.error
