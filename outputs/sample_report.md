# Wealth Advisor Report — C002

- Run ID: `42c51587-d12e-45ac-b942-e2c596bfe6bc`
- Generated: 2026-10-16T01:34:50.783722+00:00
- Status: **success**
- Risk score: **100/100**

## Portfolio summary

- Total value: 20,000.00

| Asset class | Allocation |
|---|---|
| bond | 20.0% |
| cash | 10.0% |
| equity | 70.0% |
- Top holding: VTI (30.0% of value)

## Anomalies

| Severity | Rule | Description |
|---|---|---|
| high | large_vs_income | Debit C002-T27 of 5,000.00 exceeds 50% of monthly income (9,000.00). |
| high | new_merchant_large | First transaction at new merchant 'Crypto Exchange' is a large debit of 5,000.00. |
| high | velocity | Account B1 had 6 transactions within 24h. |
| medium | outlier_transaction | Transaction C002-T17 of 1,200.00 is an outlier in category 'groceries' (robust z=61.4). |
| medium | duplicates | 2 transactions share amount 85.40, merchant 'Trader Joes', date 2026-09-12. |
| medium | round_number_withdrawal | Debit C002-T27 of 5,000.00 is a round-number withdrawal. |

## Recommendations

| Priority | Action | Rationale | Approval |
|---|---|---|---|
| high | Contact client to verify flagged activity | 3 high-severity finding(s): large_vs_income, new_merchant_large, velocity. | yes |

## CRM context

- client_id: C002
- advisor_name: Sam Lee
- kyc_status: valid
- kyc_expiry: 2027-01-15
- last_contact_date: 2026-08-25
- risk_tolerance_updated: moderate
- life_events: []
- notes: Flagged unusual activity for review.

## Narrative
(template)

Portfolio review for client C002: risk score 100/100 with 6 finding(s), 3 of them high severity.

Talking points:
- [HIGH] Debit C002-T27 of 5,000.00 exceeds 50% of monthly income (9,000.00).
- [HIGH] First transaction at new merchant 'Crypto Exchange' is a large debit of 5,000.00.
- [HIGH] Account B1 had 6 transactions within 24h.
- [MEDIUM] Transaction C002-T17 of 1,200.00 is an outlier in category 'groceries' (robust z=61.4).
- [MEDIUM] 2 transactions share amount 85.40, merchant 'Trader Joes', date 2026-09-12.

Caveats:
- Generated from a template without an LLM; all figures are deterministic outputs of the analysis.

## Human review

- Required: True
- Decision: auto
- Note: auto-approved (flag)

## Changes since last review

- new: duplicates:C002-T18, large_vs_income:C002-T27, new_merchant_large:C002-T27, outlier_transaction:C002-T17, round_number_withdrawal:C002-T27, velocity:B1
- resolved: none
- worsened: none

## Trace

| Step | Agent | Status | Duration (ms) |
|---|---|---|---|
| ingest | data_fetcher | ok | 0 |
| analyze | analyzer | ok | 0 |
| recommend | advisor | ok | 0 |
