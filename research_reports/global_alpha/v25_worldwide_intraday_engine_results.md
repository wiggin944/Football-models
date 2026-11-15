# Global Alpha v0.25 — worldwide intraday and paper-execution engineering checkpoint

**Date:** 15 November 2026. **Outcome:** Engineering tests passed. **NO TRADING LIVE. NO VERIFIED ALPHA.**

## User's actual product goal

The user deposits GBP whenever they like (often monthly salary), and the
algorithm continuously discovers and evaluates **all listed global equities**,
across developed/emerging markets and market capitalisations. A trained
context-aware ML portfolio policy autonomously buys/sells/holds/reallocates
cash intraday when expected risk-adjusted utility exceeds total execution
costs. Deposit cadence does not control trading frequency.

## Completed, reproducible deliverables

- `research/global_alpha/global_security_master.py`: stable security identity
  distinct from issuer identity/ticker, exchange MIC, dated historical listing
  endpoints and order eligibility gate.
- `research/global_alpha/v25_global_intraday_paper_engine.py`: GBP-denominated,
  **broker-disconnected** intraday paper engine consuming external target
  weights, with buy/sell fills at bid/ask, local-per-GBP FX, explicit fees,
  cash reserve, issuer position cap, portfolio equity cap, lot sizes, market
  status, forecast/FX/quote age, liquidity participation and kill switch.
- `research/global_alpha/v25_global_intraday_data_audit.py`: recent *historical*
  five-minute OHLC audit across 11 geographically dispersed listed equities.
  Requires monotonic, timestamped, internally valid price bars and >=2
  trading sessions. Source: Yahoo historical OHLC via yfinance; **not executable
  intraday real-time bid/ask quotes**.
- `research/global_alpha/v25_worldwide_listings_adapter.py`: exchange-wide
  candidate discovery using EODHD's exchanges list and *separate active and
  delisted* per-exchange endpoints. No country whitelist or market-cap filter.
  **Only mocked endpoint tests run**; authenticated live catalog needs the
  user's own EODHD API key and appropriate rights. Ticker conflicts are
  flagged, not treated as identical permanent instruments. Discovery records
  are not mistakenly promoted to historically eligible trading securities.

### Tested results

Reproducible passing GitHub Actions workflow:
https://github.com/wiggin944/Football-models/actions/runs/43773522566

- Eleven of eleven country examples passed the recent 5m OHLC data quality audit:
  US AAPL (390 bars), UK HSBA.L (510), Germany SAP.DE (510), Japan 7203.T (326),
  Hong Kong 0700.HK (335), India RELIANCE.NS (369), South Korea 005930.KS (360),
  Australia BHP.AX (293), Brazil PETR4.SA (417), Canada SHOP.TO (390), South
  Africa NPN.JO (470); **4,370 total bars** from 4–5 recent trading sessions each.
  This establishes *11 probe symbols*, not worldwide intraday feed completeness.
- Synthetic GBP paper cash test: £1,000 deposit, synthetic US buy at $100.05
  offer, sell at $105 bid, modelled fee+FX spread, £1,002.80 resulting cash.
  **This is a fixture, NOT a model-produced return.**
- Paper engine explicitly rejected stale data, no broker contract,
  a closed exchange, and its kill switch; 0 broker API interactions.
- Exchange-catalog tests passed for two fixture exchanges with 3 active
  and 3 delisted sample tickers, including a duplicated/reused symbol and
  an unfiltered small company; **no real EODHD API token used**.
- Intraday bars do not contain actual bid/ask depth. The paper engine was
  tested with labelled simulated bid/ask; do not use OHLC midpoints as
  fillable prices without spread/slippage modelling.

## Architectural implication

Discovery is low-frequency/batch across every supported exchange, including
delisted history. Paid real-time quote entitlement is concentrated on a
dynamic risk-screened **watchlist**, not all global instruments simultaneously.
True historical minute/tick data across all markets and legally usable
security/delisting masters are nontrivial commercial sourcing requirements.
A symbol/ISIN snapshot is not point-in-time historical membership. FX quotes,
venue calendars and data permissions are separate prerequisites.

IBKR's standard Web API account has 100 initial market-data lines and exchange
subscription gates. Broker paper connectivity, entitlements, risk controls
and a user-authorised account are still absent; no orders can be sent by
these scripts. EODHD currently lists roughly 70 supported exchanges and
active+delisted symbol endpoints. These are **vendor coverage claims**,
not a demonstration of complete data for every listed global company.

## Next milestone

1. Build provider-neutral exchange-listing registry with durable issuer/
   instrument identity, active/delisted events and source provenance; run
   it on a **licensed authenticated** catalog spanning many exchange MICs.
   Quantify missing small-cap and historical corporate-action coverage.
2. Research actual licensed 1m/tick quote availability, FX, real-time
   top-of-book and fees on a few pilot venues with data subscriptions,
   expanding only on verified data quality.
3. Build a multi-market historical **event-time replay** from original
   quotes/trades with exchange calendars, asynchronous observation and
   publication timestamps, margin/cash and GBP execution.
4. Wire in a trained ML **forecast distribution** and portfolio optimiser,
   not a fixed toy rule, then test against passive, momentum and risk-adjusted
   controls with truly untouched global data.
5. Keep brokerage live connectivity **OFF** until long prospective
   paper-track results and user approval.

**Decision:** good engineering progress; no trading or recommendation.
