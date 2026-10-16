"""Pydantic v2 data models for the wealth-advisor data contracts.

Mirrors the spec contracts exactly:
- client input records (data/clients.json)
- CRM records (data/crm_mock.json)
- the final report (outputs/report_<client_id>.json)
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

RiskProfile = Literal["conservative", "moderate", "aggressive"]
AccountType = Literal["checking", "savings", "brokerage"]
AssetClass = Literal["equity", "bond", "cash", "other"]
TransactionType = Literal["debit", "credit"]
Severity = Literal["low", "medium", "high"]
Priority = Literal["low", "medium", "high", "urgent"]
ReportStatus = Literal["success", "degraded", "failed"]
ReviewDecision = Literal["approved", "overridden", "rejected", "auto", "not_required"]
KycStatus = Literal["valid", "expired", "pending"]


class Account(BaseModel):
    """A single client account (checking, savings or brokerage)."""

    account_id: str
    type: AccountType
    balance: float
    currency: str


class Holding(BaseModel):
    """A single security holding."""

    symbol: str
    asset_class: AssetClass
    quantity: float
    market_value: float


class Transaction(BaseModel):
    """A single account transaction. ``category`` and ``merchant`` are optional."""

    id: str
    date: date
    amount: float
    type: TransactionType
    category: Optional[str] = None
    merchant: Optional[str] = None
    account_id: str


class Client(BaseModel):
    """Client input record (data/clients.json).

    Required: client_id, name, risk_profile, accounts.
    Optional (safe defaults here; missing values raise a data-quality
    warning in the data-fetcher agent): age, monthly_income,
    target_allocation, holdings, transactions.
    """

    client_id: str
    name: str
    age: Optional[int] = None
    risk_profile: RiskProfile
    monthly_income: Optional[float] = None
    target_allocation: Optional[Dict[str, float]] = None
    accounts: List[Account] = Field(min_length=1)
    holdings: List[Holding] = Field(default_factory=list)
    transactions: List[Transaction] = Field(default_factory=list)


class CRMRecord(BaseModel):
    """CRM record matched by client_id (data/crm_mock.json)."""

    client_id: str
    advisor_name: str
    kyc_status: KycStatus
    kyc_expiry: date
    last_contact_date: date
    risk_tolerance_updated: Optional[str] = None
    life_events: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class Anomaly(BaseModel):
    """One detected anomaly, as it appears in the final report."""

    id: str
    type: str
    severity: Severity
    description: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    rule: str


class Recommendation(BaseModel):
    """One advisor recommendation, as it appears in the final report."""

    action: str
    rationale: str
    priority: Priority
    requires_approval: bool


class Narrative(BaseModel):
    """Human-readable narrative. Numbers stay deterministic; only the
    prose may come from an LLM (see summary_source)."""

    executive_summary: str
    talking_points: List[str] = Field(default_factory=list)
    caveats: List[str] = Field(default_factory=list)
    summary_source: str = "template"

    @field_validator("summary_source")
    @classmethod
    def _validate_source(cls, value: str) -> str:
        """summary_source must be 'template' or 'llm:<provider>:<model>'."""
        if value == "template" or value.startswith("llm:"):
            return value
        raise ValueError("summary_source must be 'template' or 'llm:<provider>:<model>'")


class HumanReview(BaseModel):
    """Outcome of the human-in-the-loop checkpoint."""

    required: bool
    decision: ReviewDecision
    note: str = ""


class TraceEntry(BaseModel):
    """One pipeline step record."""

    step: str
    agent: str
    status: str
    duration_ms: int


class Report(BaseModel):
    """Final report (outputs/report_<client_id>.json)."""

    run_id: str
    client_id: str
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: ReportStatus
    data_quality_warnings: List[str] = Field(default_factory=list)
    portfolio_summary: Dict[str, Any] = Field(default_factory=dict)
    anomalies: List[Anomaly] = Field(default_factory=list)
    risk_score: int = Field(ge=0, le=100)
    recommendations: List[Recommendation] = Field(default_factory=list)
    crm_context: Dict[str, Any] = Field(default_factory=dict)
    narrative: Narrative
    human_review: HumanReview
    changes_since_last_review: Dict[str, List[str]] = Field(
        default_factory=lambda: {"new": [], "resolved": [], "worsened": []}
    )
    trace: List[TraceEntry] = Field(default_factory=list)
