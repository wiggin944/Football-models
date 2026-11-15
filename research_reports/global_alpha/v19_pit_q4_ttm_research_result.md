# Global Alpha v0.19 — original filing Q4/TTM feature quality result (15 November 2026)

**Status: successful engineering/data-quality gate, not a backtest. NO VERIFIED ALPHA and NO LIVE TRADING.**

## Reproducible executions

- SEC as-filed source including 6m and 9m YTD: https://github.com/wiggin944/Football-models/actions/runs/43766604118
- Frozen 4,617 company-month Q4/TTM/industry-peer features: https://github.com/wiggin944/Football-models/actions/runs/43766787123
- Code: `research/global_alpha/v19_pit_ttm_peer_features.py` on `research/global-alpha-v19-pit-quarterly-ttm`.

## Data and anti-leakage controls

- 81 SEC-mapped issuers from 100 deterministically selected historical US stock folders, 28 contemporary SIC two-digit groups.
- 76,595 original SEC financial numeric observations, retaining filing accession, date, acceptance timestamp and numeric statement period.
- 4,617 monthly issuer observations from January 2022 to September 2026.
- Synthetic tests passed for Q4 = annual FY less 9-month YTD, Q3 = 9m less 6m, Q2 = 6m less Q1, later restatements not entering earlier dates, signed negative profit retained and point-in-time SIC lookup.
- Source availability = original filing date + five calendar days (conservative lag; not a substitute for exchange-specific trading calendars).
- Standalone cash-flow Q2/Q3/Q4 derived only when the exact same XBRL concept tag, relevant economic period and original filing vintage are available. No cross-concept addition.
- SIC groups are inferred from as-of SEC filing records; the historical stock-folder-to-CIK mapping still uses a 2026 snapshot and is not a full historical entity master.
- Peer comparisons require at least three *other* sample companies with the same SEC SIC 2-digit industry and an available comparable metric, and exclude the company itself. They do not represent full industry competitive sets.

## Coverage (August 2025 to September 2026; 81-company accounting sample)

| Signal | Availability |
|---|---:|
| TTM revenue | 73.99% |
| TTM revenue growth YoY | 73.99% |
| TTM earnings | 77.78% |
| TTM net margin | 72.93% |
| TTM operating cash flow | **75.84%** |
| TTM operating cash flow/assets | 54.50% |
| Liabilities/assets (non-financial sector where comparable) | 44.44% |
| Cash/assets | 100.00% |
| SIC2 peer-relative revenue growth | 47.35% |
| SIC2 peer-relative margin | 47.35% |
| SIC2 peer-relative cash-flow/assets | 33.77% |
| SIC2 peer-relative cash/assets | 64.20% |
| SIC2 peer-relative leverage | 18.52% |

The previous CFO TTM pipeline exposed only ~0.97% of data because it skipped qtrs=2 and 3 cumulative reports. Correct reconstruction increased usable TTM CFO to **75.84%**. In the current evaluation section 776 company-months use a synthetically reconstructed revenue Q4 and 827 an earnings Q4. These are counts of months, not unique companies or unique fiscal years.

## Interpretation and limits

- The large cash-flow availability improvement is a **data parsing repair, not predictive alpha**.
- 81 issuers cannot establish a global investment universe, peer competitive exposure or risk-control model.
- SIC2 is an approximate industry grouping and risks mixing economically distinct businesses.
- TTM concepts that change accounting tags are marked missing; financial-sector leverage and cashflow/assets are excluded because concepts are not automatically comparable.
- Coverage is for the accounting panel, not for the entire tradable security population.
- Price-adjustment quality, corporate actions, delistings, past ticker identities, cross-border reporting and valuation denominators require separate audits.
- The monthly 2025–2026 research period has already been viewed repeatedly. No return-optimised model should be promoted on it.
- The external 2026 company-map dataset has separate licensing constraints; obtain production rights or replace it with an appropriately licensed historical identifier master before deployment.

## Next research gate

1. Secure historically correct company/security/industry identifiers and a larger cross-region accounting sample.
2. Recover balance-sheet leverage for industrial issuers with Assets / Equity accounting identities, with accounting-quality checks.
3. Audit common currency, adjusted/unadjusted prices, stock-split handling, cash-flow and valuation denominators before constructing price-to-free-cash-flow and other valuations.
4. Freeze an economic-context model including TTM revenue/earnings/operating cash flow, industry-relative features, peer competition, macro regimes, dividends and event signals.
5. Validate on an independent period and realistic transaction/FX/slippage, salary contributions, portfolio cash exposure and downside-risk constraints.
6. Keep LIVE TRADING DISABLED until a robust economic and operational gate passes.
