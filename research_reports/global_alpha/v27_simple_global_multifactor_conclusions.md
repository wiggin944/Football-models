# Global Alpha v27 — simplified worldwide quality/value/momentum baseline (2026-11-15)

**Decision:** Research ONLY; no verified reliable alpha; real-money trading DISABLED. This is the new core hypothesis, not a reason to resurrect high-frequency overengineering.

**Frozen design:** 55% US, 30% developed ex-US, 15% emerging; each region 1/3 value (high book-to-market), 1/3 quality (high operating profitability), 1/3 momentum (prior 12-to-2 returns); long-only 80% big / 20% small high-factor academic groups. Rebalance monthly. Fama-French USD dividend-inclusive reconstructed historical portfolios, 1995–2025, 2026 YTD observations. Reference = same geographic mix of regional value-weighted research markets; neutral group controls use same size mix. Research one-way sleeve cost 15 bps; 0.75% annual assumed hidden implementation drag on factor sleeves; broker costs unknown.

**Main observed results (annualized, nominal USD):**
- 1995–2025: factor **10.49%** versus fixed-region gross market **9.32%**; max drawdowns **-54.18% vs -53.61%**. Factor versus size-matched neutral 8.88%. Volatility 15.27% vs 15.30%. Positive 36-month windows vs regional market ~62%.
- 2016–2025: factor **12.25%** vs same-region market **12.14%**: nearly identical; factor DD -25.73%.
- 2009–2025: factor **12.04%** vs actual dividend-adjusted **Vanguard Total World Stock ETF (VT) 11.31%**, but **same-region market returns 12.27%**, *exceeding the factor strategy*. This is the critical geographic-exposure control: factor versus VT performance cannot be called demonstrated stock-selection alpha.
- Factor minus VT 2009–2025 annualized arithmetic return difference +0.66pp; descriptive 12-month circular block 95% interval **[-0.57,+1.84]pp**, crosses zero.
- 2009–2025 factor with underlying annual implementation drag 0%, 0.5%, 1.0%, 1.5%, 2% produced CAGR 12.89%, 12.33%, 11.76%, 11.20%, 10.63% respectively, versus VT 11.31%. **At 1.5% drag factor loses**.
- 2016–2025 factor 12.25% vs VT 11.77%, **but not same-region alpha**.

**Do not misinterpret:** academic sleeves are *separately* screened factor baskets (not one company scored on all three attributes), with backwards-reconstructed and revised monthly return series; underlying holdings/turnover cannot be priced directly. Not equivalent to investable worldwide stocks, UK/Irish tax and ISA eligibility, GBP FX exposure, broker spread/fees, dividends by residence, salary allocation or live autonomous orders. Risk and >50% historical drawdowns fall short of the user's conservative mandate.

**Next strict gate:** test an actually purchasable low-cost global quality/value/momentum basket with a **matched-country/size/currency passive comparator** in GBP, after explicit purchase spreads and TER. Fail closed if improvement < costs or risk worse. Keep ML and intraday activity out of core until independently additive. Begin prospective paper track, avoid hyperparameter choices based on inspected past returns. Goal remains all-world equity discovery with autonomous buy/sell/hold/cash, not monthly-only execution or US index strategy.

**Reproducibility:** research code `research/global_alpha/v27_simple_global_factor_backtest.py` and `v27b_broad_world_vt_comparator.py` on branch `research/global-alpha-v27-simple-global-factors`. Archived reports and monthly matched series:
https://github.com/wiggin944/Football-models/actions/runs/43776368409
https://github.com/wiggin944/Football-models/actions/runs/43776576043

Sources: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html ; https://advisors.vanguard.com/investments/products/vt/vanguard-total-world-stock-etf
