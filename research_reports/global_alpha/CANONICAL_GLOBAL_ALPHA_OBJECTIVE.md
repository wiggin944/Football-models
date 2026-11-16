# Global Alpha — canonical objective (confirmed 15 November 2026)

**One continuously updated global portfolio, not separate rule-based strategies.**

The system must consider all eligible markets, company fundamentals, valuations, technical conditions, liquidity, competitor information, dividends, currencies, rates, macro events and uncertainty *together*. It must evaluate short (minutes/days), medium (weeks/months) and long (years) opportunities **simultaneously across stocks, gold, gilts, cash and other suitable investable assets**.

## One shared capital allocation decision

An opportunity is not an order. It is a time-stamped, calibrated estimate of potential net profit, risks, expiry and time horizon on an instrument, conditional on current information. An ensemble may have several correlated views on the *same* asset at different horizons. Those views must be reconciled, **not simply added as if independent profits**, nor allocated to phantom sub-portfolios that each spend the same money.

A central, receding-horizon portfolio optimiser evaluates **all current opportunity forecasts and existing holdings jointly**, including correlations, exposure to the same issuer/asset/country/currency, downside risk, execution and liquidation friction, cash carry and alternative opportunities. It chooses portfolio targets now, and planned future possibilities for modelling only. Future instructions are never sent until revalidated using new information.

Cash is an investable option with an explicit achievable yield; gilts/gold/equities compete with it. Both standing long-term positions and short-term trades can coexist in one net account. The investor adds GBP at any time; the algorithm determines when to allocate it. There is no mandate to trade or to choose one universal holding period.

## Outcome and evidence

The desired outcome is **substantial, sustainable, risk-adjusted alpha net of all realistic costs** relative to appropriately matched multi-asset and global equity passive benchmarks, with attention to adverse regimes. This cannot be guaranteed. Predictions may be wrong; abstaining is an acceptable action. No live orders until repeated independent, out-of-time and prospective paper results justify it.

Research stack: (1) correct global point-in-time asset/security and company identities, data/returns/corporate actions; (2) independent horizon forecasts of *distributions*, as-of timestamps and calibrated confidence; (3) joint risk-constrained multi-period allocation; (4) event-time execution simulator with bid/ask, liquidity, FX, broker/tax restrictions and cash ledger; (5) independently evaluated walk-forward results. Good source availability, synthetic profitable fills or an attractive academic-factor graph are **not evidence of genuine alpha**.

The existing v0.27 quality/value/momentum blend and v0.28 binary-regime experiments are **baselines**, not the desired final model. The 11-country intraday probe is plumbing only, not the strategy's universe or validation. The project remains research-only with live trading disabled.

## Continuous replacement: sell a good opportunity for a *better* one

At every meaningful information update the whole current portfolio competes with all newly discovered opportunities. **Past entry price and an earlier buy signal are not reasons to keep a position**. The optimiser compares the expected future net utility of (a) continuing a holding, (b) selling/reducing it, (c) buying another opportunity, (d) cash or other defensive assets, and (e) holding both when diversification makes that preferable.

A switch is rational only when the **incremental expected benefit of changing allocations** exceeds all incremental buy and sell spreads, commissions, FX, market impact, capital-gains or other tax cost where relevant, slippage and forecast uncertainty. Small marginal gains should not trigger endless turnover; the model must account for tax lots and avoid false loss/gain harvesting from invalid tax assumptions. Previously received dividends are sunk; *future* expected distributions may matter. Ex-dividend price adjustments mean that an expected dividend must **never be counted twice** in total-return forecasts.

Diversification isn't just a maximum number of stocks: combine issuer and dual-listing links, correlated business/sector exposure, currency, country, factor and interest-rate duration risk, liquidity, stress losses, concentration and the opportunity cost of holding cash. The decision isn't a fixed formula deciding 'stocks versus gold': every candidate and horizon competes within the same capital and risk constraints.

A receding-horizon plan can see better opportunities expected soon, including dividend events, earnings and liquidity windows, but orders are only created from valid **current information**. Re-run the whole allocation when the information and costs change, not just when money enters the account.

### First engineering acceptance tests (synthetic only, no alpha implication)
- An old asset with a good forecast remains held if a new opportunity is only marginally better after realistic turnover costs.
- A sufficiently better opportunity causes the allocator to *sell or reduce* the old asset to fund the replacement when cash/risk is constrained.
- Multiple horizons on one asset are reconciled, not double counted.
- Gold, gilts and achievable GBP cash yield are real alternatives in that same optimisation.
- Limits across issuer, industry, region, currency and tail-risk scenarios prevent an individually high-scoring trade from making the whole portfolio too risky.
- Dividends are entered exactly once according to whether the forecast basis is price-only or already total-return; withholding/tax and corporate actions are explicitly modelled.
- Research outputs remain **paper-only** and no source-independent synthetic outcome is called alpha.
