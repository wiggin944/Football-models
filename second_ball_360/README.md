# Second-Ball 360

A coaching-oriented football analytics project using StatsBomb 360 freeze frames to study **how player positioning around contested aerial contacts relates to second-ball readiness**.

The core question is deliberately pre-contact:

> **At the instant of an aerial contest, before observing the outgoing touch, what does the surrounding player structure tell us about which side is better prepared to secure the loose second ball?**

The project combines event semantics, 360 spatial context, football-engineered geometry, gradient-boosted trees and graph neural networks (GNNs). A major part of the work was not simply fitting models, but auditing what a "second ball" actually meant in the underlying event data and rejecting model complexity when it did not earn its place.

## Why this problem matters

Second balls are often discussed as effort or aggression, but they are also structural. Before the ball drops, teams can differ in:

- support density around the contest
- nearest-support distances
- spacing and spread
- numerical balance ahead of and behind the duel
- alignment with the incoming ball
- how well the likely drop zone is surrounded

The final coaching model treats these as **predictive indicators of second-ball readiness**, not causal instructions. It can rank situations and identify spatial patterns associated with stronger or weaker recovery states, but it does not claim that moving one player to a particular coordinate would cause a recovery.

## Data

The project uses **StatsBomb Open Data** and StatsBomb 360 freeze frames.

- 300 matches with available 360 data were downloaded.
- 7,856 physical aerial contests were paired from the event representation.
- 6,912 had a canonical winner-side 360 frame.
- After the final target audit, the strict modelling sample contains **2,683 genuine loose second-ball episodes across 291 matches**.
- The aerial winner's team secured the first controlled action in **21.3%** of those strict episodes.

Raw StatsBomb event and 360 JSON files are not committed to this repository. The download scripts rebuild them locally.

See [`ATTRIBUTION.md`](./ATTRIBUTION.md) for data-source attribution and publication notes.

## The target-definition audit

The most important modelling decision in the project came after an apparently excellent result.

An early landing-zone model on aerial passes achieved roughly:

- log loss: **0.538**
- ROC-AUC: **0.797**

That looked impressive, but a football-semantic audit showed why.

Among the aerial passes in the original sample:

- **52.9%** were marked complete
- **98.3%** of those completed passes were followed by an immediate same-team receipt

In other words, much of the model's apparent success came from predicting **directly completed headed passes**, not genuinely loose second balls.

The target was therefore rebuilt.

### Final operational definition

A **true second-ball episode** is a contested aerial contact that:

1. does **not** directly complete to a teammate, and
2. is followed by a first secure controlled action.

The strict primary sample contains:

- resolved **Clearances**
- **Incomplete** aerial Passes

Directly completed headed passes, `Out`, and `Unknown` pass outcomes are excluded from the primary target.

This reduced the sample to **2,683** episodes and forced every subsequent model comparison to be rerun.

## Sequential information experiment

The project explicitly separates information available **before contact** from information that only becomes available **after the first touch**.

| Information stage | Log loss | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|
| Prior only | 0.5182 | 0.4972 | 0.2120 | 0.1677 |
| Contest location | 0.5108 | 0.6022 | 0.2736 | 0.1657 |
| Setup geometry | 0.5054 | 0.6355 | 0.2992 | 0.1634 |
| Setup + incoming-ball context | **0.5032** | **0.6416** | **0.3074** | **0.1627** |
| + first-contact type | 0.4734 | 0.7127 | 0.3736 | 0.1525 |
| + outgoing-touch information | 0.4543 | 0.7473 | 0.4079 | 0.1471 |
| + landing-zone geometry | 0.4533 | 0.7500 | 0.3971 | 0.1473 |

### Coaching interpretation

The pre-contact model is deliberately the main coaching model.

It is not a high-certainty event predictor, but it contains **real structural signal before the first contact happens**:

- location alone improves on the prior
- local support geometry improves further
- incoming-ball-relative structure adds additional information
- the largest predictive gains occur only after the first contact is observed

That distinction matters. The pre-contact model asks about **readiness**; the stronger contact-aware model asks what becomes predictable once the duel has already shaped the ball's next state.

## Branch behaviour

The two main first-contact branches behave differently.

### Incomplete aerial passes

For incomplete aerial passes, much of the useful information arrives through the **outgoing touch and landing geometry**. Pre-contact geometry adds relatively little after location.

### Clearances

For clearances, surrounding **setup and incoming-ball geometry** matter more. There is no equivalent clean pass endpoint, so the problem remains more dependent on the structure around the contest.

This is one reason a single headline metric is not enough: aggregate discrimination partly reflects different base rates and mechanisms across the two branches.

## GNN experiments

The problem is naturally relational, so several GNN formulations were tested. The aim was not to force a deep-learning result, but to test whether raw player-to-player structure added information beyond compact football-engineered geometry.

### Standalone pre-contact GNN

Fair same-information comparison:

| Model | Log loss | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|
| Setup + incoming XGB | **0.5032** | **0.6416** | **0.3074** | **0.1627** |
| Pre-contact positioning GNN | 0.5120 | 0.6075 | 0.2856 | 0.1649 |

The GNN did not outperform the tabular model.

### First-contact multitask GNN

A second GNN was given a more explicitly relational task: predict the first-contact behaviour itself.

| Task | GNN | Tabular |
|---|---:|---:|
| Pass vs clearance log loss | 0.3365 | **0.3349** |
| Pass-length RMSE | **7.96** | 7.99 |
| Pass-length R² | **-0.017** | -0.023 |
| Pass-height log loss | **0.6682** | 0.6703 |

The models were effectively tied. Neither model extracted meaningful predictive power for outgoing pass length.

### Residual graph correction

The final GNN experiment asked the cleanest question:

> **Does the raw player graph contain useful information after the engineered XGB has already made its prediction?**

The GNN was trained only to learn a correction to the XGB logit.

| Model | Log loss | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|
| Nested pre-contact XGB | **0.5038** | **0.6409** | **0.3102** | **0.1626** |
| XGB + residual GNN | 0.5056 | 0.6378 | 0.3056 | 0.1634 |

The average absolute graph correction was only **0.040 logits**.

The conclusion is therefore evidence-based rather than architectural:

> **With this sample size and event-centred 360 representation, carefully engineered football geometry is more data-efficient than graph learning.**

The relational framing still matters, but the GNN did not justify its additional complexity.

## What the model is useful for

The pre-contact model is best viewed as a **coaching and R&D diagnostic**.

It can support:

- ranking aerial situations by structural second-ball readiness
- identifying recurring support structures associated with stronger recovery probability
- comparing how local spacing changes across different contest contexts
- producing spatial hypotheses for coaches or analysts to investigate with richer tracking data
- separating what is knowable before contact from what only becomes knowable after the first touch

It should **not** be treated as:

- a causal prescription for exact player movement
- an individual-player rating system
- a team-ranking model
- a replacement for tracking data
- a guarantee of who will recover a specific loose ball

## Data limitations

StatsBomb 360 is event-centred freeze-frame data rather than continuous tracking.

The model does not observe:

- player velocity
- acceleration
- body orientation
- continuous trajectories
- reliable persistent player identities across freeze frames
- the full 22-player configuration in every frame

Those limitations matter particularly for second balls, where movement immediately before and after contact is important.

A natural future extension would use continuous tracking data, potentially with state estimation for player movement and a dynamic graph model.

## Repository structure

```text
second_ball_360/
├── README.md
├── ATTRIBUTION.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── download_open_data.py
│   ├── download_360_data.py
│   ├── build_contest_catalogue.py
│   ├── audit_second_ball_sequences.py
│   ├── validate_second_ball_labels.py
│   ├── build_core_dataset.py
│   ├── identify_contestants.py
│   ├── validate_contestant_identification.py
│   ├── build_graph_dataset.py
│   ├── build_features.py
│   ├── audit_target_definition.py
│   ├── rebuild_strict_second_ball_models.py
│   ├── train_information_stage_models.py
│   ├── train_positioning_gnn.py
│   ├── evaluate_residual_gnn.py
│   ├── evaluate_subset_sensitivity.py
│   └── make_figures.py
├── outputs/
│   └── final/
├── figures/
├── docs/
└── tests/
```

## Reproducing the pipeline

Create and activate a virtual environment, then install:

```bash
pip install -r requirements.txt
```

The source files are intentionally explicit and research-oriented. The main pipeline is:

```bash
python src/download_open_data.py
python src/download_360_data.py
python src/build_contest_catalogue.py
python src/audit_second_ball_sequences.py
python src/validate_second_ball_labels.py
python src/build_core_dataset.py
python src/identify_contestants.py
python src/validate_contestant_identification.py
python src/build_graph_dataset.py
python src/build_features.py
python src/audit_target_definition.py
python src/rebuild_strict_second_ball_models.py
python src/train_information_stage_models.py
python src/train_positioning_gnn.py
python src/evaluate_residual_gnn.py
```

`evaluate_subset_sensitivity.py` and `make_figures.py` are supporting analysis/visualisation scripts rather than required build stages.

## Evaluation design

Key evaluation choices:

- whole-match grouped cross-validation
- no row-level leakage across train/validation matches
- nested out-of-fold prediction where stacked/residual models require a baseline prediction
- match-clustered bootstrap comparisons for model deltas
- PR-AUC reported alongside ROC-AUC because the strict positive class is only 21.3%
- branch-level analysis for Pass and Clearance mechanisms
- no tuning on an already-inspected held-out test set

## Methodological takeaway

The strongest result of the project is not that a particular neural network won.

It is that:

1. a suspiciously strong model exposed a target-definition problem,
2. the target was audited and rebuilt,
3. information was decomposed by the moment it becomes observable,
4. several relational deep-learning formulations were tested fairly,
5. the simpler engineered model survived those comparisons.

That is the final model-selection result.

## Author

William Higgin
