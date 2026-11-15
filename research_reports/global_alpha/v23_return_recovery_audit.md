# Global Alpha v0.23 — fixed-prediction return recovery audit

**15 November 2026 — research only — no verified alpha — no live trades**

## Experiment

This report documents an exploratory January 2024–June 2025 US equity test (18 monthly decisions, 460–482 model-eligible stocks per month). It **does not** represent a global, independently tested investment system.

The original v0.21 models had seven missing return months for price-only Ridge and between four and ten for other portfolios because the historical S&P constituent archive stopped recording some companies after they left the index. The companies were not necessarily delisted.

v0.22 matched 12 of 13 affected tickers against independent adjusted Yahoo daily series, with 25-day original-versus-independent price-return overlap checks; 24 stock-month return gaps were repaired. QRVO's two additional missing returns use $69.32, $70.72 and $81.78 first-monthly opens for December 2024, January 2025 and February 2025 from [StatMuse](https://www.statmuse.com/money/ask/when-did-qorvo-go-public), with the original archive independently confirming December's $69.32 first-day open. These **two QRVO months have lower source assurance**, not a daily-overlap check.

v0.23 confirms **no model fitting or portfolio selection changed**, comparing every monthly turnover to the original v0.21 run. Only missing realised returns and corresponding observed rank-correlation calculations were repaired.

## Results

Returns are in USD with a modelled 20 bps one-way turnover trading cost. No GBP FX or UK tax modelling.

| Strategy | Annualised return (%) | Maximum drawdown (%) | Average monthly one-way turnover (%) | Mean rank IC |
|---|---:|---:|---:|---:|
| SPY dividend-adjusted benchmark | 19.98 | -6.96 | 5.6 | n/a |
| 11-month momentum | 18.49 | -8.08 | 23.3 | +0.0420 |
| Price-only Ridge | 15.65 | -8.87 | 33.7 | +0.0203 |
| Price + quarterly/TTM SEC fundamentals | 12.66 | -8.02 | 31.0 | -0.0055 |
| Price + real SEC fundamentals + SIC2 peers | 11.74 | -9.84 | 30.1 | -0.0039 |
| Price + real SEC fundamentals + deliberately mismatched SIC2 peers | 15.12 | -6.82 | 28.0 | +0.0012 |
| Equal-weight researched security sample | 12.12 | -9.01 | 6.1 | n/a |

**No ML strategy beat momentum or SPY on annualised return in this investigated period.**

The returns were measured across an already-viewed historical sample. This cannot establish a statistically independent estimate of alpha. Headline risk-adjusted metrics are also not decisive given sample size and data limitations.

### Conditional, post-hoc paired comparisons

Annualised mean arithmetic monthly return differences (not CAGR differences), with descriptive circular 3-month block bootstrap intervals:

- Fundamentals vs price-only Ridge: **-2.72 percentage points**; 95% interval **[-5.07, -0.41]**; 6/18 positive months.
- Fundamentals + SIC2 peers vs price-only Ridge: **-3.51 pp**; interval **[-7.55, +0.01]**; 7/18 positive months.
- Fundamentals + mismatched peers vs price-only Ridge: **-0.50 pp**; interval **[-3.98, +3.32]**; 10/18 positive months.
- Momentum vs price-only Ridge: **+2.55 pp**; interval **[-3.23, +6.80]**; 9/18 positive months.

Intervals are **descriptive only**, not multiplicity-corrected or valid out-of-sample significance estimates.

### Source robustness: QRVO

The only stock whose two missing outcomes lack daily-overlap verification was selected by price-only Ridge. Replacing its individual monthly returns with values ten percentage points higher or lower **in both affected months** changes price-only Ridge's annualised CAGR from 15.65% to only **15.54–15.76%**; momentum and SPY remain ahead. This does not erase lower source assurance, but the overall negative result is insensitive to those two QRVO return estimates under that stress test.

## Methodology and remaining blockers

1. The original historical S&P index archive is a retrospective collection. Its stocks' index membership, full monthly price continuity and delisting recovery proceeds are **not** verified throughout the period. Some model inputs may be stale when the archive stops mid-month.
2. Original model training labels still exclude some unobservable historical exits during 2022–2023. The v0.23 repair explicitly **does not refit or repair training labels**.
3. SEC original filing timestamps and reconstructed trailing-year fundamentals are much more complete than earlier versions, but historical security-to-CIK mapping still uses a modern directory, and SIC2 peers are approximate industry groups rather than actual competitor maps.
4. No complete point-in-time cross-border accounting coverage, valuation denominators (split-adjusted vs raw price), earnings revisions/consensus surprise, country risk, dividend tax treatment, GBP FX execution or regulatory UK ISA suitability check.
5. Results cover 18 researched historical decisions with repeated researcher inspection, not prospective performance. The performance of a monthly salary contribution scheme, cash allocation, stop conditions, downside limits and real brokerage has **not** been evaluated.

**Decision:** Do not trade the system. The current price-only, financial and industry-peer Ridge models fail the basic benchmark comparison. Freeze v0.23 as an exploratory negative result. Prioritise historical security identity, continuous daily corporate-action-adjusted pricing (including exits), and genuinely differentiated point-in-time valuation/earnings-surprise/competitor signals, then reserve untouched data for a pre-registered global walk-forward evaluation.

## Reproducibility

- Price, original model and as-filed accounting: versioned GitHub research branch `research/global-alpha-v21-us-economic-context-pilot`.
- Original exit audit: `research/global-alpha-v22-exit-price-audit`.
- v0.23 exact-prediction comparison: [GitHub Actions run](https://github.com/wiggin944/Football-models/actions/runs/43771294911).
- Output artifact: `global-alpha-v23-frozen-prediction-exit-bridge`, with all 18 monthly return rows and sourcing metadata.
