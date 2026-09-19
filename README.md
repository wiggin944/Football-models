# Football Models

A collection of football modelling projects built in Python, covering probabilistic match simulation, unsupervised player-rating methods and spatial football analytics.

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

### 3. Second-Ball 360

[`second_ball_360/`](./second_ball_360)

A coaching-oriented spatial modelling project using StatsBomb 360 data to study how player positioning around contested aerial contacts relates to second-ball readiness.

The project:

- reconstructs physical aerial contests from StatsBomb event semantics
- defines and validates the first secure controlled action after the duel
- audits and corrects the target to remove directly completed headed passes
- engineers local support, spacing and incoming-ball-relative geometry
- decomposes predictive information into pre-contact and post-contact stages
- benchmarks XGBoost against several graph-neural-network formulations
- uses grouped cross-validation and match-clustered bootstrap evaluation
- finds that compact football-engineered geometry is more data-efficient than the tested GNNs on event-centred 360 snapshots

The final pre-contact model is intended as a coaching/R&D diagnostic rather than a causal prescription.

## Tech

Python · pandas · NumPy · SciPy · scikit-learn · XGBoost · PyTorch · graph neural networks · maximum-likelihood estimation · Monte Carlo simulation · unsupervised learning · spatial football analytics

## Related work

For a more recent example of production-style ML engineering, see my [Tactical Style ML Service](https://github.com/wiggin944/tactical-style-ml-service), which packages part of my MSc football-modelling research behind FastAPI with tests, Docker and CI.

## Author

William Higgin
