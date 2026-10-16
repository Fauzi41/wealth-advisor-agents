"""Central configuration: thresholds, severities, retry and LLM settings.

Every magic number used by the anomaly / risk rules lives here so the
rules stay small and the trade-offs stay visible in one place.
"""
from __future__ import annotations

# --- anomaly rule thresholds -------------------------------------------------
OUTLIER_ZSCORE_THRESHOLD = 3.5  # robust z-score (median/MAD) of amount within category
OUTLIER_MIN_TXNS_PER_CATEGORY = 5  # skip categories with fewer transactions
LARGE_VS_INCOME_RATIO = 0.5  # single debit > 50% of monthly_income
NEW_MERCHANT_LARGE_RATIO = 0.2  # first-seen merchant AND amount > 20% of monthly_income
VELOCITY_MAX_TXNS = 5  # more than 5 transactions ...
VELOCITY_WINDOW_HOURS = 24  # ... within any 24h window on the same account
ROUND_WITHDRAWAL_DIVISOR = 1000  # debit amount divisible by 1000 ...
ROUND_WITHDRAWAL_MIN_AMOUNT = 2000  # ... and >= 2000
SPENDING_TREND_RATIO = 1.3  # latest month debits > 130% of previous months' average
BALANCE_DROP_RATIO = 0.4  # total account balance dropped > 40% vs long-term memory
ALLOCATION_DRIFT_PP = 5.0  # any asset class differs from target by > 5 percentage points
CONCENTRATION_RATIO = 0.3  # single holding > 30% of total holdings value
PROFILE_CONSERVATIVE_MAX_EQUITY = 0.7  # conservative with equity > 70% ...
PROFILE_AGGRESSIVE_MIN_EQUITY = 0.3  # ... or aggressive with equity < 30%
KYC_EXPIRY_WARNING_DAYS = 30  # KYC expiring within 30 days
STALE_CONTACT_DAYS = 180  # no contact for > 180 days

# --- severity per rule --------------------------------------------------------
SEVERITY_OUTLIER_TRANSACTION = "medium"
SEVERITY_LARGE_VS_INCOME = "high"
SEVERITY_NEW_MERCHANT_LARGE = "high"
SEVERITY_VELOCITY = "high"
SEVERITY_DUPLICATES = "medium"
SEVERITY_ROUND_NUMBER_WITHDRAWAL = "medium"
SEVERITY_SPENDING_TREND = "low"
SEVERITY_BALANCE_DROP = "high"
SEVERITY_ALLOCATION_DRIFT = "medium"
SEVERITY_CONCENTRATION = "medium"
SEVERITY_PROFILE_MISMATCH = "high"
SEVERITY_CRM_KYC_EXPIRED = "high"
SEVERITY_CRM_KYC_EXPIRING = "medium"
SEVERITY_CRM_STALE_CONTACT = "low"
SEVERITY_CRM_LIFE_EVENT = "low"

# --- risk score ----------------------------------------------------------------
RISK_WEIGHT_HIGH = 25
RISK_WEIGHT_MEDIUM = 10
RISK_WEIGHT_LOW = 3
RISK_SCORE_MAX = 100
RISK_KYC_EXPIRED_BONUS = 10  # added when KYC is expired

# --- retry (CRM timeouts / 5xx) -------------------------------------------------
RETRY_COUNT = 3
BACKOFF_BASE_S = 0.5

# --- LLM (optional; the system runs fully without these) -------------------------
LLM_TIMEOUT_S = 20
LLM_MAX_TOKENS = 600
LLM_MAX_RETRIES = 1
