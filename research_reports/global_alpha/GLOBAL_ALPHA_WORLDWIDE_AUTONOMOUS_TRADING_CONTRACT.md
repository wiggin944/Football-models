# Global Alpha — global all-listed-equities and autonomous intraday design contract
Version: 0.24 · 15 November 2026 · research only / live orders disabled

## Objective — the user's actual request

A user deposits GBP whenever they want (normally from salary), and the autonomous system chooses:
**which global equities to buy, hold, reduce, or sell; exactly when and in what size; when to keep cash.**
It is NOT an S&P 500 index strategy, NOT a monthly-only rebalance, NOT a US-only stock universe,
and NOT restricted to large-cap companies or to price-only momentum. It must seek incremental
*out-of-sample, risk-adjusted, net-of-cost* alpha under a downside-risk budget.

### Discovery universe — all countries, all listed common shares
- Build an exchange-wide **global security master**, not a hard-coded ticker list. Include US/Canada,
  EU/UK/Switzerland, Japan, mainland China, Hong Kong, Taiwan, Korea, India, ASEAN, Middle East,
  Australia/NZ, Latin America, and African exchanges where licensed feeds, legal and broker
  access permit. Add other country/exchange codes dynamically; this is not a region whitelist.
- Discover all historical and present equity listings, including delisted entities, small and
  micro-caps, secondary share classes, and exchange transfers. Avoid look-ahead membership,
  survivorship and stale identifier bugs. Maintain issuer, instrument and listing separately.
- Canonical **instrument identity** is NOT ticker: use date-valid MIC + exchange-specific
  permanent security ID + ISIN/FIGI (where available); link to issuer and original regulatory
  document identifiers with effective-from/effective-to dates. Never use 2026 mapping to
  resolve an earlier trading session without historical validity proof.
- Duplicate share classes and dual listings remain separate instruments linked to the same
  issuer; avoid duplicate risk exposures. Do not treat ADR and primary listing as different firms.
- **Screening not exclusion from discovery:** investigate every listing; order eligibility requires
  proof of live broker permissions, local regulatory status, bid/ask liquidity, tradeable lot sizes,
  price quality, exchange status, borrowing restrictions if relevant, and data freshness.
  Ineligible instruments stay visible with reason codes; do not silently remove stocks with bad outcomes.
- Open-ended market and feed registry: a new market can be added without rewriting the model.

### Temporal resolution and trading
- Store and reconcile **intraday bid/ask quotes, trades, and 1m/5m/hourly bars** where licensed;
  daily adjusted/raw OHLCV, splits, dividends, rights, mergers, delistings, auctions and suspensions.
- Historical intraday data is a separate licensed *research requirement*. A provider claiming
  'global intraday' is not presumed to offer real-time tick streaming in every country.
- Observation timestamps: exchange local timestamp, UTC exchange event time, **first received
  at** timestamp, effective availability, publication time, corporate action treatment.
- Cross-venue UTC schedules, holidays, auctions, halts, settlement, FX conversion, DST;
  market calendar required. No forward-filled prices through halted or closed markets.
- Multiple model horizons: event/microstructure minute-hours, short-term days, medium-term
  weeks-months, long-term structural years. Learn each holding-period utility separately.
  The user's monthly contribution cadence is **not** the signal or trading cadence.
- Event loop chooses risk-adjusted actions continuously or upon fresh signals, but may choose
  not to trade. Penalise churn by spread, FX, commissions, market impact and slippage.

### Economic-context model
Historical company price path / drawdown / reversal / trend at multiple timescales, volatility,
microstructure/order-book quality, peer competitors, upstream/downstream exposures, TTM financials,
balance sheet, value and dividend stream (with RAW share-price denominator and valid share counts),
analyst revisions where licensed, filings/news surprise with original publication timestamps,
credit/funding exposure, macro, rates, local FX, cross-market risk regime, political risk.
Use true firm competitors where known, **do not call correlation neighbours competitors**.

### Optimiser and trade implementation
- Expected net outcome **distribution** per instrument / holding horizon (uncertainty + tail loss),
  calibrated over time and across markets; abstain where evidence is poor.
- Portfolio optimizer combines forecasts, asset correlations, country/industry/currency exposure,
  issuer duplication and liquidity with an explicit downside risk target and GBP cash budget.
- Actions: BUY, ADD, HOLD, REDUCE, SELL, STAY IN CASH; limit and risk-aware order sizing.
  Prevent trading while market status / FX / stale price / broker permission / feature freshness
  are unknown. Emergency kill switch, manual override, reconciliation and immutable audit trail.
- Contributions enter cash ledger asynchronously; automated trades can occur when justified,
  not necessarily on deposit day. Account type (UK ISA vs taxable) determines eligible products,
  reporting and tax constraints; no universal assumption that every worldwide stock can be
  purchased by one UK account.
- **Research-only now.** Paper orders and shadow allocations before any live brokerage
  authorisation. No broker or user funds connected.

### Independent evidence and fail-closed requirements
- Globally diversified *historical investable membership* and exchange-specific original
  as-of security listing must be validated first. Index membership is an independent subset
  diagnostic, not the live universe definition.
- Source vendors' adjustment factors and dividend treatment audited; delistings retained
  including cash/stock consideration, no silent dropped names.
- Purged rolling-origin splits, embargo around overlapping labels, untouched evaluation
  after pre-registration of model features/hyperparameters/risk policy; block bootstrap,
  factor-adjusted and regional benchmark returns, liquidity/capacity stress, all net of costs.
- Report missing price **as invalid returns**, not as profit or zero loss. Record universe
  available in each country on each trading day and missing/rejected reasons.
- Guard against data leakage from modern ticker maps, report end dates, earnings restatements,
  future index constituents, stale fundamentals and incorrectly scaled splits.
- No single historical 18-month test can certify live alpha. Track paper results for long enough
  to see changing regimes. Default to passive/cash until evidence passes.

## Initial evidence / sources confirmed 15 November 2026
- Existing v0.23 US sample: **no validated alpha**; official-ish price archives and original
  as-filed SEC accounting are still not globally comprehensive.
- Interactive Brokers TWS API offers real-time quotes, intraday/historical bars and broker
  execution on supported global instruments, subject to permissions/market-data licences.
- EODHD advertises global EOD/intraday stocks, but real-time WebSocket coverage is more
  limited (US stocks, Europe as offered, FX, crypto). Historical intraday coverage depends
  on plan and exchange; verify per MIC before claiming universal data.
- OpenFIGI provides global instrument identifier mappings, including optional unlisted
  equities, but is **not** an investable-universe, tick-data or delisting-return provider.
- None of these three sources individually provides a verified all-world, all-microcap
  historical as-of security master + full intraday + corporate actions + point-in-time
  fundamentals for free.

## Workstream order
1. Complete US historical membership+missing-outcome audit purely as an **integrity test**,
   never mistake the S&P 500 membership file for global scope.
2. Integrate a country-agnostic instrument/listing entity model and discovery interface.
3. Pilot 3-5 exchange regions with verified historical listing dates, delisted securities,
   as-of fundamentals and intraday quote quality. Then expand to global exchange coverage.
4. Build realistic event-driven simulation with original timestamps, GBP ledger, order books,
   local hours, slippage, corporate actions, FX and contributions.
5. Only then run a genuinely independent all-markets alpha test; retain broker OFF and
   evaluate viability/cost before switching on controlled paper trading.

## References
- https://www.interactivebrokers.com/docs/tws-api/doc/md/introduction
- https://eodhd.com/pricing/
- https://www.openfigi.com/api/documentation
