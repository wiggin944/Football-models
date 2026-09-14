# Football Models

A small collection of football modelling projects built in Python, covering probabilistic match simulation and unsupervised player-rating methods.

## Projects

### 1. Dixon-Coles World Cup simulator

[`dixon_coles/`](./dixon_coles)

A 2026 World Cup simulation built around a Dixon-Coles goal model fitted by maximum likelihood.

The implementation includes:

- time-decayed historical match weighting
- team-specific attacking and defensive strengths
- home-advantage and low-score dependence adjustments
- full group-stage simulation with points, goal difference and goals scored
- best-third-place qualification logic
- knockout-bracket simulation
- comparison of model probabilities with bookmaker-implied prices

The project was designed to automate repeated full-tournament simulations rather than only predict individual matches.

### 2. GMM player-rating system

[`player_ratings/`](./player_ratings)

An unsupervised player-rating approach designed to let statistical role archetypes emerge from the data rather than imposing fixed positional categories.

The pipeline:

- normalises volume statistics per 90
- standardises player features
- selects the number of Gaussian Mixture Model components using silhouette score
- estimates role-specific feature importance from cluster separation
- combines weighted z-scores with sample-size reliability
- applies diminishing returns and low-minute safeguards
- outputs relative squad ratings on a 0-10 scale

The model was tested on real squad data, including Mjällby AIF.

## Tech

Python · pandas · NumPy · SciPy · scikit-learn · maximum-likelihood estimation · Monte Carlo simulation · unsupervised learning

## Related work

For a more recent example of production-style ML engineering, see my [Tactical Style ML Service](https://github.com/wiggin944/tactical-style-ml-service), which packages part of my MSc football-modelling research behind FastAPI with tests, Docker and CI.

## Author

William Higgin
