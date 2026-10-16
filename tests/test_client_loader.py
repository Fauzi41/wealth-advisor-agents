"""Client loader tests: per-client validation, bad file handling, warnings."""
import json

from wealth_assistant.models import Client
from wealth_assistant.tools.client_loader import ClientLoaderTool


def _valid_client_dict():
    return {
        "client_id": "C001",
        "name": "Jane Doe",
        "risk_profile": "moderate",
        "accounts": [{"account_id": "A1", "type": "checking",
                      "balance": 2500, "currency": "USD"}],
    }


def _write(tmp_path, payload):
    path = tmp_path / "clients.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_mixed_valid_and_invalid_clients(tmp_path):
    bad = {"name": "No ID", "risk_profile": "moderate",  # missing client_id
           "accounts": [{"account_id": "A9", "type": "savings",
                         "balance": 1, "currency": "USD"}]}
    path = _write(tmp_path, [_valid_client_dict(), bad,
                             _valid_client_dict() | {"client_id": "C003"}])
    result = ClientLoaderTool().run(path=path)
    assert result.ok is True
    assert len(result.data["clients"]) == 2
    assert all(isinstance(c, Client) for c in result.data["clients"])
    assert len(result.data["errors"]) == 1
    assert result.data["errors"][0]["index"] == 1
    assert result.meta == {"path": path, "total": 3, "valid": 2, "invalid": 1}


def test_missing_file_returns_not_ok():
    result = ClientLoaderTool().run(path="does/not/exist.json")
    assert result.ok is False
    assert "not found" in result.error


def test_invalid_json_returns_not_ok(tmp_path):
    path = tmp_path / "clients.json"
    path.write_text("{oops", encoding="utf-8")
    result = ClientLoaderTool().run(path=str(path))
    assert result.ok is False
    assert "invalid JSON" in result.error


def test_top_level_not_list_returns_not_ok(tmp_path):
    path = _write(tmp_path, {"client_id": "C001"})
    result = ClientLoaderTool().run(path=path)
    assert result.ok is False
    assert "expected a list" in result.error


def test_missing_optional_fields_produce_warnings(tmp_path):
    path = _write(tmp_path, [_valid_client_dict()])
    result = ClientLoaderTool().run(path=path)
    assert result.ok is True
    warnings = result.data["warnings"]
    assert len(warnings) == 1
    assert warnings[0]["client_id"] == "C001"
    assert set(warnings[0]["missing_fields"]) == {
        "age", "monthly_income", "target_allocation", "holdings",
        "transactions",
    }
