"""One positive and one negative test per anomaly rule, plus run_all/score tests."""
from datetime import date

from wealth_assistant.models import Client, CRMRecord
from wealth_assistant.tools import anomaly_detector as ad
from wealth_assistant.tools.portfolio_metrics import compute_portfolio_metrics

TODAY = date(2026, 10, 16)


def make_client(**overrides) -> Client:
    base = {
        "client_id": "C1",
        "name": "Test",
        "risk_profile": "moderate",
        "monthly_income": 8000,
        "accounts": [
            {"account_id": "A1", "type": "checking", "balance": 10000,
             "currency": "USD"}
        ],
        "holdings": [],
        "transactions": [],
    }
    base.update(overrides)
    return Client(**base)


def txn(tid, day, amount, type="debit", category="groceries",
        merchant="Store", account_id="A1"):
    return {"id": tid, "date": day, "amount": amount, "type": type,
            "category": category, "merchant": merchant,
            "account_id": account_id}


def make_crm(**overrides) -> CRMRecord:
    base = {"client_id": "C1", "advisor_name": "Sam Lee",
            "kyc_status": "valid", "kyc_expiry": "2027-06-01",
            "last_contact_date": "2026-09-15",
            "risk_tolerance_updated": "moderate", "life_events": [],
            "notes": ""}
    base.update(overrides)
    return CRMRecord(**base)


def metrics_for(client) -> dict:
    return compute_portfolio_metrics(client)


# --- 1. outlier_transaction ---------------------------------------------------

def test_outlier_transaction_positive():
    txns = [txn(f"T{i}", f"2026-09-{i + 1:02d}", amt)
            for i, amt in enumerate([50, 52, 48, 51, 49, 500])]
    out = ad.outlier_transactions(make_client(transactions=txns))
    assert len(out) == 1
    assert out[0].rule == "outlier_transaction"
    assert out[0].evidence["transaction_id"] == "T5"
    assert out[0].severity == "medium"


def test_outlier_transaction_negative():
    txns = [txn(f"T{i}", f"2026-09-{i + 1:02d}", amt)
            for i, amt in enumerate([50, 52, 48, 51, 49, 53])]
    assert ad.outlier_transactions(make_client(transactions=txns)) == []
    few = [txn(f"T{i}", "2026-09-01", 50.0 + i) for i in range(4)]
    assert ad.outlier_transactions(make_client(transactions=few)) == []


# --- 2. large_vs_income --------------------------------------------------------

def test_large_vs_income_positive():
    out = ad.large_vs_income(make_client(
        transactions=[txn("T1", "2026-09-01", 5000.0)]))
    assert len(out) == 1 and out[0].severity == "high"


def test_large_vs_income_negative():
    assert ad.large_vs_income(make_client(
        transactions=[txn("T1", "2026-09-01", 3000.0)])) == []
    assert ad.large_vs_income(make_client(monthly_income=None,
        transactions=[txn("T1", "2026-09-01", 9000.0)])) == []


# --- 3. new_merchant_large -----------------------------------------------------

def test_new_merchant_large_positive():
    txns = [txn("T1", "2026-09-01", 2000.0, merchant="NewStore")]
    out = ad.new_merchant_large(make_client(transactions=txns))
    assert len(out) == 1 and out[0].severity == "high"


def test_new_merchant_large_negative():
    txns = [txn("T1", "2026-09-01", 100.0, merchant="NewStore"),
            txn("T2", "2026-09-02", 2000.0, merchant="NewStore")]
    assert ad.new_merchant_large(make_client(transactions=txns)) == []


# --- 4. velocity ----------------------------------------------------------------

def test_velocity_positive():
    txns = [txn(f"T{i}", "2026-09-01", 10.0 + i) for i in range(6)]
    out = ad.velocity(make_client(transactions=txns))
    assert len(out) == 1 and out[0].evidence["count"] == 6


def test_velocity_negative():
    txns = [txn(f"T{i}", "2026-09-01", 10.0 + i) for i in range(5)]
    assert ad.velocity(make_client(transactions=txns)) == []


# --- 5. duplicates ---------------------------------------------------------------

def test_duplicates_positive():
    txns = [txn("T1", "2026-09-01", 120.5, merchant="Whole Foods"),
            txn("T2", "2026-09-01", 120.5, merchant="Whole Foods"),
            txn("T3", "2026-09-02", 50.0)]
    out = ad.duplicates(make_client(transactions=txns))
    assert len(out) == 1
    assert out[0].evidence["transaction_ids"] == ["T1", "T2"]


def test_duplicates_negative():
    txns = [txn("T1", "2026-09-01", 120.5), txn("T2", "2026-09-01", 121.5)]
    assert ad.duplicates(make_client(transactions=txns)) == []


# --- 6. round_number_withdrawal ----------------------------------------------------

def test_round_number_withdrawal_positive():
    out = ad.round_number_withdrawals(make_client(
        transactions=[txn("T1", "2026-09-01", 2000.0)]))
    assert len(out) == 1 and out[0].severity == "medium"


def test_round_number_withdrawal_negative():
    assert ad.round_number_withdrawals(make_client(
        transactions=[txn("T1", "2026-09-01", 1500.0)])) == []
    assert ad.round_number_withdrawals(make_client(
        transactions=[txn("T1", "2026-09-01", 3000.0, type="credit")])) == []


# --- 7. spending_trend ---------------------------------------------------------------

def test_spending_trend_positive():
    txns = [txn("T1", "2026-08-05", 1000.0), txn("T2", "2026-10-05", 1500.0)]
    out = ad.spending_trend(make_client(transactions=txns))
    assert len(out) == 1 and out[0].severity == "low"


def test_spending_trend_negative():
    txns = [txn("T1", "2026-08-05", 1000.0), txn("T2", "2026-10-05", 1200.0)]
    assert ad.spending_trend(make_client(transactions=txns)) == []
    assert ad.spending_trend(make_client(
        transactions=[txn("T1", "2026-10-05", 5000.0)])) == []


# --- 8. balance_drop ------------------------------------------------------------------

def test_balance_drop_positive():
    out = ad.balance_drop(make_client(), {"C1": 100000.0})
    assert len(out) == 1 and out[0].severity == "high"


def test_balance_drop_negative():
    assert ad.balance_drop(make_client(), {"C1": 12000.0}) == []
    assert ad.balance_drop(make_client(), None) == []
    assert ad.balance_drop(make_client(), {"OTHER": 1.0}) == []


# --- 9. allocation_drift -----------------------------------------------------------------

def test_allocation_drift_positive():
    client = make_client(
        target_allocation={"equity": 0.6, "bond": 0.4},
        holdings=[{"symbol": "A", "asset_class": "equity", "quantity": 1,
                   "market_value": 10000}])
    out = ad.allocation_drift(client, metrics_for(client))
    assert len(out) == 2  # equity +40pp, bond -40pp
    assert all(a.severity == "medium" for a in out)


def test_allocation_drift_negative():
    client = make_client(
        target_allocation={"equity": 0.6, "bond": 0.4},
        holdings=[{"symbol": "A", "asset_class": "equity", "quantity": 1,
                   "market_value": 6000},
                  {"symbol": "B", "asset_class": "bond", "quantity": 1,
                   "market_value": 4000}])
    assert ad.allocation_drift(client, metrics_for(client)) == []
    client2 = make_client(holdings=[{"symbol": "A", "asset_class": "equity",
                                     "quantity": 1, "market_value": 10000}])
    assert ad.allocation_drift(client2, metrics_for(client2)) == []


# --- 10. concentration ---------------------------------------------------------------------

def test_concentration_positive():
    client = make_client(holdings=[
        {"symbol": "AAPL", "asset_class": "equity", "quantity": 1,
         "market_value": 9000},
        {"symbol": "BND", "asset_class": "bond", "quantity": 1,
         "market_value": 1000}])
    out = ad.concentration(client, metrics_for(client))
    assert len(out) == 1 and out[0].evidence["symbol"] == "AAPL"


def test_concentration_negative():
    client = make_client(holdings=[
        {"symbol": s, "asset_class": "equity", "quantity": 1,
         "market_value": 2500} for s in ("A", "B", "C", "D")])
    assert ad.concentration(client, metrics_for(client)) == []


# --- 11. profile_mismatch --------------------------------------------------------------------

def test_profile_mismatch_positive():
    cons = make_client(risk_profile="conservative", holdings=[
        {"symbol": "A", "asset_class": "equity", "quantity": 1,
         "market_value": 10000}])
    out = ad.profile_mismatch(cons, metrics_for(cons))
    assert len(out) == 1 and out[0].severity == "high"
    aggr = make_client(risk_profile="aggressive", holdings=[
        {"symbol": "B", "asset_class": "bond", "quantity": 1,
         "market_value": 10000}])
    assert len(ad.profile_mismatch(aggr, metrics_for(aggr))) == 1


def test_profile_mismatch_negative():
    mod = make_client(risk_profile="moderate", holdings=[
        {"symbol": "A", "asset_class": "equity", "quantity": 1,
         "market_value": 10000}])
    assert ad.profile_mismatch(mod, metrics_for(mod)) == []
    cons = make_client(risk_profile="conservative", holdings=[
        {"symbol": "A", "asset_class": "equity", "quantity": 1,
         "market_value": 5000},
        {"symbol": "B", "asset_class": "bond", "quantity": 1,
         "market_value": 5000}])
    assert ad.profile_mismatch(cons, metrics_for(cons)) == []


# --- 12. crm_kyc -------------------------------------------------------------------------------

def test_crm_kyc_expired_positive():
    out = ad.crm_kyc(make_crm(kyc_expiry="2025-01-01",
                             kyc_status="expired"), TODAY)
    assert len(out) == 1
    assert out[0].severity == "high"
    assert out[0].evidence["expired"] is True


def test_crm_kyc_expiring_positive():
    out = ad.crm_kyc(make_crm(kyc_expiry="2026-10-30"), TODAY)
    assert len(out) == 1
    assert out[0].severity == "medium"
    assert out[0].evidence["expired"] is False


def test_crm_kyc_negative():
    assert ad.crm_kyc(make_crm(kyc_expiry="2027-06-01"), TODAY) == []
    assert ad.crm_kyc(None, TODAY) == []


# --- 13. crm_stale_contact -------------------------------------------------------------------------

def test_crm_stale_contact_positive():
    out = ad.crm_stale_contact(make_crm(last_contact_date="2026-01-01"), TODAY)
    assert len(out) == 1 and out[0].severity == "low"


def test_crm_stale_contact_negative():
    assert ad.crm_stale_contact(
        make_crm(last_contact_date="2026-09-15"), TODAY) == []
    assert ad.crm_stale_contact(None, TODAY) == []


# --- 14. crm_life_event --------------------------------------------------------------------------------

def test_crm_life_event_positive():
    out = ad.crm_life_event(make_crm(life_events=["retirement planned 2028"]))
    assert len(out) == 1 and out[0].severity == "low"


def test_crm_life_event_negative():
    assert ad.crm_life_event(make_crm(life_events=[])) == []
    assert ad.crm_life_event(None) == []


# --- run_all / compute_risk_score -----------------------------------------------------------------------

def test_run_all_sorts_by_severity():
    client = make_client(
        risk_profile="conservative",
        transactions=[txn("T1", "2026-09-01", 5000.0, merchant="New")],
        holdings=[{"symbol": "A", "asset_class": "equity", "quantity": 1,
                   "market_value": 10000}])
    out = ad.run_all(client)
    severities = [a.severity for a in out]
    assert severities == sorted(severities,
                               key=lambda s: {"high": 0, "medium": 1,
                                              "low": 2}[s])
    assert "high" in severities


def test_compute_risk_score_weights():
    anomalies = ad.run_all(make_client(
        transactions=[txn("T1", "2026-09-01", 4500.0, merchant="New")]))
    # large_vs_income (high=25) + new_merchant_large (high=25)
    assert ad.compute_risk_score(anomalies) == 50


def test_compute_risk_score_kyc_bonus():
    kyc = ad.crm_kyc(make_crm(kyc_expiry="2025-01-01",
                             kyc_status="expired"), TODAY)
    assert ad.compute_risk_score(kyc) == 25 + 10


def test_compute_risk_score_capped():
    many = [ad._anomaly(f"h{i}", "velocity", "high", "x") for i in range(10)]
    assert ad.compute_risk_score(many) == 100
