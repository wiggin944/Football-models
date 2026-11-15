# Global Alpha v0.18 — confirmed research result (2026-11-15)

**Decision: DO NOT DEPLOY. No independently verified global or US alpha.**

Source code: `research/global_alpha/v18_sec_fundamental_ablation.py` on branch `research/global-alpha-v18-fundamentals-ablation`. Reproducible execution: https://github.com/wiggin944/Football-models/actions/runs/43763373517.

### Study design

- 70 predeclared sampled US securities tradable at August 2025 decision cutoff (100 folders sampled initially).
- Original SEC as-filed records, availability conservatively lagged five days from filing and gated for age/quality; 81-company accounting panel, 4,617 company-months.
- Expanding chronological ridge retraining, **identical 2022+ training observations** for price-only ML and SEC+price models, 2,788–3,698 matured training records per monthly fit.
- Comparator: identical features with SEC data mismatched to another company's ticker; plus 11-month momentum, equal weighting, SPY research ETF proxy.
- Fourteen subsequent GBP return months from Aug 2025–Sep 2026 decisions. Assumed stock one-way dealing 20 bps and FX 30 bps times traded notional.
- **Period previously inspected: exploratory, not untouched holdout.**

### Observed performance

| Method | GBP CAGR | Max drawdown | Mean monthly one-way turnover |
|---|---:|---:|---:|
| Price-only Ridge | 7.00% | -6.86% | 39.8% |
| Price + correct SEC fundamentals | 7.13% | -7.68% | 40.5% |
| Price + mismatched SEC data | 15.78% | -7.25% | 41.2% |
| Momentum | 7.17% | -10.92% | 28.6% |
| Equal weight of sample | 10.94% | -5.34% | 7.1% |
| SPY research ETF proxy | 17.94% | -4.85% | not comparable |

Correct SEC features beat price-only in 7/14 months. Observed annualised arithmetic mean return increment +0.06 percentage points (descriptive three-month block-bootstrap 95% interval -7.46 to +7.63). Correct-company SEC features trailed the mismatched-company placebo by -7.93 points annualised arithmetic (-14.83 to -1.41 descriptive bootstrap interval). No clean firm-specific predictive gain.

**Version caution:** Price-only returns here are different from v0.14 because the current test matches both models to the same 2022-onward training horizon; do not compare model vintages as identical runs.

### Next gate

No model tuning on this period. Resolve historical security-to-CIK, delistings and corporate actions, original-filing-safe Q4/TTM and sector-specific statement semantics, sector peer identity, macro/publication timing and multiple countries. Freeze risk, portfolio selection, salary-contribution and UK account execution assumptions before a distinct untouched period or long prospective paper run. No live trading.
