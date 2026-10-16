"""Anomaly detection: 14 deterministic rules, run_all() and compute_risk_score().

Each rule is a separate small function returning ``list[Anomaly]``. Rules
that lack the data they need (no income, no target allocation, no CRM
record, ...) skip cleanly and contribute nothing -- they never crash.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from statistics import median
from typing import Any, Dict, List, Mapping, Optional

from .. import config
from ..models import Anomaly, Client, CRMRecord
from .base import Tool, ToolResult
from .portfolio_metrics import compute_portfolio_metrics

#: Sort order for run_all(): high severity first, stable within a level.
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

RISK_WEIGHTS = {
    "high": config.RISK_WEIGHT_HIGH,
    "medium": config.RISK_WEIGHT_MEDIUM,
    "low": config.RISK_WEIGHT_LOW,
}


def _anomaly(
    anomaly_id: str,
    rule: str,
    severity: str,
    description: str,
    evidence: Optional[Dict[str, Any]] = None,
) -> Anomaly:
    """Build an Anomaly with the rule name as both type and rule."""
    return Anomaly(
        id=anomaly_id,
        type=rule,
        severity=severity,  # type: ignore[arg-type]
        description=description,
        evidence=evidence or {},
        rule=rule,
    )


# --- transaction rules --------------------------------------------------------


def outlier_transactions(client: Client) -> List[Anomaly]:
    """Flag transactions whose robust z-score within their category is high.

    Uses the modified z-score ``0.6745 * (x - median) / MAD`` and flags
    ``|z| > 3.5``. Categories with fewer than 5 transactions are skipped;
    a zero MAD (identical amounts) cannot produce an outlier and is skipped.
    """
    by_category: Dict[str, List] = {}
    for txn in client.transactions or []:
        if txn.category:
            by_category.setdefault(txn.category, []).append(txn)

    findings: List[Anomaly] = []
    for category, txns in by_category.items():
        if len(txns) < config.OUTLIER_MIN_TXNS_PER_CATEGORY:
            continue
        amounts = [t.amount for t in txns]
        med = median(amounts)
        mad = median([abs(a - med) for a in amounts])
        if mad == 0:
            continue
        for txn in txns:
            z = 0.6745 * (txn.amount - med) / mad
            if abs(z) > config.OUTLIER_ZSCORE_THRESHOLD:
                findings.append(
                    _anomaly(
                        f"outlier_transaction:{txn.id}",
                        "outlier_transaction",
                        config.SEVERITY_OUTLIER_TRANSACTION,
                        f"Transaction {txn.id} of {txn.amount:,.2f} is an outlier "
                        f"in category '{category}' (robust z={z:.1f}).",
                        {
                            "transaction_id": txn.id,
                            "category": category,
                            "amount": txn.amount,
                            "median": med,
                            "mad": mad,
                            "z_score": round(z, 2),
                            "threshold": config.OUTLIER_ZSCORE_THRESHOLD,
                        },
                    )
                )
    return findings


def large_vs_income(client: Client) -> List[Anomaly]:
    """Flag single debits exceeding 50% of monthly income.

    Skipped when monthly income is missing or not positive.
    """
    income = client.monthly_income
    if not income or income <= 0:
        return []
    findings: List[Anomaly] = []
    for txn in client.transactions or []:
        if txn.type == "debit" and txn.amount > config.LARGE_VS_INCOME_RATIO * income:
            findings.append(
                _anomaly(
                    f"large_vs_income:{txn.id}",
                    "large_vs_income",
                    config.SEVERITY_LARGE_VS_INCOME,
                    f"Debit {txn.id} of {txn.amount:,.2f} exceeds 50% of "
                    f"monthly income ({income:,.2f}).",
                    {
                        "transaction_id": txn.id,
                        "amount": txn.amount,
                        "monthly_income": income,
                        "ratio": round(txn.amount / income, 3),
                    },
                )
            )
    return findings


def new_merchant_large(client: Client) -> List[Anomaly]:
    """Flag large debits at a merchant seen for the first time.

    Merchants are tracked chronologically, so only the first transaction at
    a merchant can fire. Debit-only by design: a large first-time credit
    (payroll, transfers) is normal, not suspicious. Skipped without monthly
    income; transactions without a merchant are skipped.
    """
    income = client.monthly_income
    if not income or income <= 0:
        return []
    threshold = config.NEW_MERCHANT_LARGE_RATIO * income
    seen = set()
    findings: List[Anomaly] = []
    ordered = sorted(client.transactions or [], key=lambda t: (t.date, t.id))
    for txn in ordered:
        merchant = txn.merchant
        first_seen = merchant is not None and merchant not in seen
        if merchant is not None:
            seen.add(merchant)
        if first_seen and txn.type == "debit" and txn.amount > threshold:
            findings.append(
                _anomaly(
                    f"new_merchant_large:{txn.id}",
                    "new_merchant_large",
                    config.SEVERITY_NEW_MERCHANT_LARGE,
                    f"First transaction at new merchant '{merchant}' is a large "
                    f"debit of {txn.amount:,.2f}.",
                    {
                        "transaction_id": txn.id,
                        "merchant": merchant,
                        "amount": txn.amount,
                        "monthly_income": income,
                        "threshold": threshold,
                    },
                )
            )
    return findings


def velocity(client: Client) -> List[Anomaly]:
    """Flag accounts with more than 5 transactions in any 24h window.

    Transaction dates have day granularity, so each date is treated as
    midnight for windowing. One anomaly per affected account.
    """
    by_account: Dict[str, List[datetime]] = {}
    for txn in client.transactions or []:
        by_account.setdefault(txn.account_id, []).append(
            datetime.combine(txn.date, time.min)
        )

    findings: List[Anomaly] = []
    window = timedelta(hours=config.VELOCITY_WINDOW_HOURS)
    for account_id, stamps in by_account.items():
        stamps.sort()
        start = 0
        for end in range(len(stamps)):
            while stamps[end] - stamps[start] > window:
                start += 1
            if end - start + 1 > config.VELOCITY_MAX_TXNS:
                findings.append(
                    _anomaly(
                        f"velocity:{account_id}",
                        "velocity",
                        config.SEVERITY_VELOCITY,
                        f"Account {account_id} had {end - start + 1} transactions "
                        "within 24h.",
                        {
                            "account_id": account_id,
                            "count": end - start + 1,
                            "window_start": stamps[start].date().isoformat(),
                            "window_end": stamps[end].date().isoformat(),
                        },
                    )
                )
                break
    return findings


def duplicates(client: Client) -> List[Anomaly]:
    """Flag (amount, merchant, date) combinations appearing 2+ times."""
    groups: Dict[tuple, List] = {}
    for txn in client.transactions or []:
        groups.setdefault((txn.amount, txn.merchant, txn.date), []).append(txn)

    findings: List[Anomaly] = []
    for (amount, merchant, day), txns in groups.items():
        if len(txns) >= 2:
            findings.append(
                _anomaly(
                    f"duplicates:{txns[0].id}",
                    "duplicates",
                    config.SEVERITY_DUPLICATES,
                    f"{len(txns)} transactions share amount {amount:,.2f}, "
                    f"merchant '{merchant}', date {day}.",
                    {
                        "transaction_ids": [t.id for t in txns],
                        "amount": amount,
                        "merchant": merchant,
                        "date": day.isoformat(),
                        "count": len(txns),
                    },
                )
            )
    return findings


def round_number_withdrawals(client: Client) -> List[Anomaly]:
    """Flag debits divisible by 1000 and >= 2000."""
    findings: List[Anomaly] = []
    for txn in client.transactions or []:
        if (
            txn.type == "debit"
            and txn.amount >= config.ROUND_WITHDRAWAL_MIN_AMOUNT
            and txn.amount % config.ROUND_WITHDRAWAL_DIVISOR == 0
        ):
            findings.append(
                _anomaly(
                    f"round_number_withdrawal:{txn.id}",
                    "round_number_withdrawal",
                    config.SEVERITY_ROUND_NUMBER_WITHDRAWAL,
                    f"Debit {txn.id} of {txn.amount:,.2f} is a round-number "
                    "withdrawal.",
                    {"transaction_id": txn.id, "amount": txn.amount},
                )
            )
    return findings


def spending_trend(client: Client) -> List[Anomaly]:
    """Flag when the latest month's debits exceed 130% of the previous average.

    Skipped with fewer than 2 distinct months or a zero previous average.
    """
    monthly: Dict[str, float] = {}
    for txn in client.transactions or []:
        if txn.type == "debit":
            key = txn.date.strftime("%Y-%m")
            monthly[key] = monthly.get(key, 0.0) + txn.amount
    if len(monthly) < 2:
        return []
    latest = max(monthly)
    previous = [total for month, total in monthly.items() if month != latest]
    avg_previous = sum(previous) / len(previous)
    if avg_previous == 0:
        return []
    latest_total = monthly[latest]
    if latest_total > config.SPENDING_TREND_RATIO * avg_previous:
        return [
            _anomaly(
                f"spending_trend:{latest}",
                "spending_trend",
                config.SEVERITY_SPENDING_TREND,
                f"Debits in {latest} ({latest_total:,.2f}) are "
                f"{latest_total / avg_previous:.0%} of the previous monthly "
                f"average ({avg_previous:,.2f}).",
                {
                    "month": latest,
                    "month_total": latest_total,
                    "previous_average": round(avg_previous, 2),
                    "ratio": round(latest_total / avg_previous, 3),
                },
            )
        ]
    return []


def balance_drop(
    client: Client, prior_balances: Optional[Mapping[str, float]] = None
) -> List[Anomaly]:
    """Flag a >40% drop in total account balance vs long-term memory.

    Skipped when no prior balance is stored for this client (or it is not
    positive).
    """
    if not prior_balances or client.client_id not in prior_balances:
        return []
    prior = prior_balances[client.client_id]
    if prior is None or prior <= 0:
        return []
    current = sum(a.balance for a in client.accounts or [])
    drop = (prior - current) / prior
    if drop > config.BALANCE_DROP_RATIO:
        return [
            _anomaly(
                "balance_drop:total",
                "balance_drop",
                config.SEVERITY_BALANCE_DROP,
                f"Total balance fell {drop:.0%} (from {prior:,.2f} to "
                f"{current:,.2f}).",
                {
                    "prior_total": prior,
                    "current_total": current,
                    "drop_ratio": round(drop, 3),
                },
            )
        ]
    return []


# --- portfolio rules ----------------------------------------------------------


def allocation_drift(client: Client, metrics: Dict[str, Any]) -> List[Anomaly]:
    """Flag asset classes differing from target by > 5 percentage points.

    One anomaly per drifted class. Skipped without a target allocation.
    """
    if not client.target_allocation:
        return []
    allocation = metrics.get("allocation", {}) or {}
    findings: List[Anomaly] = []
    for asset_class, target in client.target_allocation.items():
        actual = allocation.get(asset_class, 0.0)
        drift_pp = (actual - target) * 100.0
        if abs(drift_pp) > config.ALLOCATION_DRIFT_PP:
            direction = "above" if drift_pp > 0 else "below"
            findings.append(
                _anomaly(
                    f"allocation_drift:{asset_class}",
                    "allocation_drift",
                    config.SEVERITY_ALLOCATION_DRIFT,
                    f"{asset_class} allocation is {actual:.0%}, "
                    f"{abs(drift_pp):.1f}pp {direction} target {target:.0%}.",
                    {
                        "asset_class": asset_class,
                        "actual": round(actual, 4),
                        "target": target,
                        "drift_pp": round(drift_pp, 2),
                    },
                )
            )
    return findings


def concentration(client: Client, metrics: Dict[str, Any]) -> List[Anomaly]:
    """Flag any single holding worth > 30% of total holdings value."""
    holdings = client.holdings or []
    total = metrics.get("total_value", 0) or 0
    if total <= 0:
        return []
    findings: List[Anomaly] = []
    for holding in holdings:
        fraction = holding.market_value / total
        if fraction > config.CONCENTRATION_RATIO:
            findings.append(
                _anomaly(
                    f"concentration:{holding.symbol}",
                    "concentration",
                    config.SEVERITY_CONCENTRATION,
                    f"Holding {holding.symbol} is {fraction:.0%} of portfolio "
                    "value.",
                    {
                        "symbol": holding.symbol,
                        "market_value": holding.market_value,
                        "total_value": total,
                        "fraction": round(fraction, 4),
                    },
                )
            )
    return findings


def profile_mismatch(client: Client, metrics: Dict[str, Any]) -> List[Anomaly]:
    """Flag conservative clients over 70% equity or aggressive under 30%."""
    allocation = metrics.get("allocation", {}) or {}
    equity = allocation.get("equity", 0.0)
    mismatch = (
        client.risk_profile == "conservative"
        and equity > config.PROFILE_CONSERVATIVE_MAX_EQUITY
    ) or (
        client.risk_profile == "aggressive"
        and equity < config.PROFILE_AGGRESSIVE_MIN_EQUITY
    )
    if not mismatch:
        return []
    return [
        _anomaly(
            "profile_mismatch:equity",
            "profile_mismatch",
            config.SEVERITY_PROFILE_MISMATCH,
            f"Risk profile '{client.risk_profile}' but equity allocation is "
            f"{equity:.0%}.",
            {
                "risk_profile": client.risk_profile,
                "equity_fraction": round(equity, 4),
            },
        )
    ]


# --- CRM rules ----------------------------------------------------------------


def crm_kyc(
    crm: Optional[CRMRecord], today: Optional[date] = None
) -> List[Anomaly]:
    """Flag expired KYC (high) or KYC expiring within 30 days (medium).

    Skipped without a CRM record.
    """
    if crm is None:
        return []
    today = today or date.today()
    expired = crm.kyc_status == "expired" or crm.kyc_expiry < today
    if expired:
        return [
            _anomaly(
                "crm_kyc:expiry",
                "crm_kyc",
                config.SEVERITY_CRM_KYC_EXPIRED,
                f"KYC is expired (expiry {crm.kyc_expiry.isoformat()}, status "
                f"'{crm.kyc_status}').",
                {
                    "kyc_status": crm.kyc_status,
                    "kyc_expiry": crm.kyc_expiry.isoformat(),
                    "expired": True,
                },
            )
        ]
    days_left = (crm.kyc_expiry - today).days
    if days_left <= config.KYC_EXPIRY_WARNING_DAYS:
        return [
            _anomaly(
                "crm_kyc:expiry",
                "crm_kyc",
                config.SEVERITY_CRM_KYC_EXPIRING,
                f"KYC expires in {days_left} days ({crm.kyc_expiry.isoformat()}).",
                {
                    "kyc_status": crm.kyc_status,
                    "kyc_expiry": crm.kyc_expiry.isoformat(),
                    "expired": False,
                    "days_until_expiry": days_left,
                },
            )
        ]
    return []


def crm_stale_contact(
    crm: Optional[CRMRecord], today: Optional[date] = None
) -> List[Anomaly]:
    """Flag no advisor contact for more than 180 days. Skipped without CRM."""
    if crm is None:
        return []
    today = today or date.today()
    days = (today - crm.last_contact_date).days
    if days > config.STALE_CONTACT_DAYS:
        return [
            _anomaly(
                "crm_stale_contact:last_contact",
                "crm_stale_contact",
                config.SEVERITY_CRM_STALE_CONTACT,
                f"No advisor contact for {days} days (last "
                f"{crm.last_contact_date.isoformat()}).",
                {
                    "last_contact_date": crm.last_contact_date.isoformat(),
                    "days_since_contact": days,
                },
            )
        ]
    return []


def crm_life_event(crm: Optional[CRMRecord]) -> List[Anomaly]:
    """Report any recorded life events (low severity, informational)."""
    if crm is None or not crm.life_events:
        return []
    return [
        _anomaly(
            "crm_life_event:events",
            "crm_life_event",
            config.SEVERITY_CRM_LIFE_EVENT,
            f"Client has {len(crm.life_events)} recorded life event(s).",
            {"life_events": list(crm.life_events)},
        )
    ]


# --- orchestration helpers ----------------------------------------------------


def run_all(
    client: Client,
    crm: Optional[CRMRecord] = None,
    metrics: Optional[Dict[str, Any]] = None,
    prior_balances: Optional[Mapping[str, float]] = None,
    today: Optional[date] = None,
) -> List[Anomaly]:
    """Run all 14 rules, returning anomalies sorted high -> medium -> low.

    Rules lacking the data they need skip cleanly and contribute nothing.
    """
    m = metrics if metrics is not None else compute_portfolio_metrics(client)
    findings: List[Anomaly] = []
    findings += outlier_transactions(client)
    findings += large_vs_income(client)
    findings += new_merchant_large(client)
    findings += velocity(client)
    findings += duplicates(client)
    findings += round_number_withdrawals(client)
    findings += spending_trend(client)
    findings += balance_drop(client, prior_balances)
    findings += allocation_drift(client, m)
    findings += concentration(client, m)
    findings += profile_mismatch(client, m)
    findings += crm_kyc(crm, today)
    findings += crm_stale_contact(crm, today)
    findings += crm_life_event(crm)
    findings.sort(key=lambda a: SEVERITY_ORDER[a.severity])
    return findings


def compute_risk_score(anomalies: List[Anomaly]) -> int:
    """Compute the 0-100 risk score.

    Capped weighted sum (high=25, medium=10, low=3), plus 10 when KYC is
    expired.
    """
    score = sum(RISK_WEIGHTS[a.severity] for a in anomalies)
    if any(a.rule == "crm_kyc" and a.evidence.get("expired") for a in anomalies):
        score += config.RISK_KYC_EXPIRED_BONUS
    return min(score, config.RISK_SCORE_MAX)


class AnomalyDetectorTool(Tool):
    """Tool wrapper: run all anomaly rules and the risk score for a client."""

    name = "anomaly_detector"

    def _execute(
        self,
        client: Any,
        crm: Any = None,
        prior_balances: Optional[Mapping[str, float]] = None,
    ) -> ToolResult:
        typed_client = client if isinstance(client, Client) else Client(**client)
        typed_crm = None
        if crm is not None:
            typed_crm = crm if isinstance(crm, CRMRecord) else CRMRecord(**crm)
        anomalies = run_all(typed_client, crm=typed_crm, prior_balances=prior_balances)
        return ToolResult(
            ok=True,
            data={
                "anomalies": anomalies,
                "risk_score": compute_risk_score(anomalies),
            },
            meta={
                "client_id": typed_client.client_id,
                "anomaly_count": len(anomalies),
            },
        )
