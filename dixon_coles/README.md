# Dixon-Coles World Cup Simulator

Probabilistic 2026 World Cup simulation built around a Dixon-Coles goal model fitted by maximum likelihood.

## What it does

- fits team attack and defence parameters from historical international results
- applies exponential time decay so recent matches carry more weight
- includes home advantage and the Dixon-Coles low-score dependence correction
- simulates full group stages with points, goal difference and goals scored
- selects the best third-place qualifiers
- simulates the knockout bracket through to the final
- compares model probabilities with bookmaker-implied prices

The emphasis is on repeated full-tournament simulation rather than only forecasting isolated matches.

## Methods

Python · pandas · NumPy · SciPy · maximum-likelihood estimation · Poisson models · Monte Carlo simulation
