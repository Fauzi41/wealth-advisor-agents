"""Model contract tests.

Covers the client input contract: a fully-populated client parses, a
missing required field is rejected, and omitted optional fields fall back
to safe defaults.
"""
import pytest
from pydantic import ValidationError

from wealth_assistant.models import Client


def _valid_client_payload() -> dict:
    """Full client record matching the data/clients.json contract."""
    return {
        "client_id": "C001",
        "name": "Jane Doe",
        "age": 52,
        "risk_profile": "moderate",
        "monthly_income": 8000,
        "target_allocation": {"equity": 0.6, "bond": 0.3, "cash": 0.1},
        "accounts": [
            {
                "account_id": "A1",
                "type": "checking",
                "balance": 25000,
                "currency": "USD",
            }
        ],
        "holdings": [
            {
                "symbol": "AAPL",
                "asset_class": "equity",
                "quantity": 100,
                "market_value": 18000,
            }
        ],
        "transactions": [
            {
                "id": "T1",
                "date": "2026-09-01",
                "amount": 120.5,
                "type": "debit",
                "category": "groceries",
                "merchant": "Whole Foods",
                "account_id": "A1",
            }
        ],
    }


def test_valid_client_parses() -> None:
    """A fully-populated client record validates cleanly."""
    client = Client(**_valid_client_payload())
    assert client.client_id == "C001"
    assert client.name == "Jane Doe"
    assert client.accounts[0].balance == 25000
    assert client.holdings[0].symbol == "AAPL"
    assert client.transactions[0].date.isoformat() == "2026-09-01"


def test_missing_required_field_fails() -> None:
    """Dropping a required field (client_id) raises ValidationError."""
    payload = _valid_client_payload()
    del payload["client_id"]
    with pytest.raises(ValidationError):
        Client(**payload)


def test_optional_fields_get_safe_defaults() -> None:
    """A minimal client (required fields only) gets safe defaults."""
    client = Client(
        client_id="C002",
        name="John Smith",
        risk_profile="conservative",
        accounts=[
            {
                "account_id": "A9",
                "type": "savings",
                "balance": 1000,
                "currency": "USD",
            }
        ],
    )
    assert client.age is None
    assert client.monthly_income is None
    assert client.target_allocation is None
    assert client.holdings == []
    assert client.transactions == []
