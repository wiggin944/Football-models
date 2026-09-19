import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVENT_DIR = PROJECT_ROOT / "data" / "events"
THREE_SIXTY_DIR = PROJECT_ROOT / "data" / "three_sixty"

FEATURE_PATH = OUTPUT_DIR / "handcrafted_features.csv"

RESULTS_PATH = OUTPUT_DIR / "true_second_ball_model_results.csv"
OOF_PATH = OUTPUT_DIR / "true_second_ball_oof_predictions.csv"
BOOTSTRAP_PATH = OUTPUT_DIR / "true_second_ball_bootstrap.csv"
AUDIT_PATH = OUTPUT_DIR / "true_second_ball_strict_sample.csv"

SEED = 42
N_SPLITS = 5
N_BOOTSTRAP = 5000


EVENT_FEATURES = [
    "is_clearance",
    "contest_x",
    "contest_y_from_centre",
    "pass_length",
    "pass_is_high",
    "pass_is_low",
]


GEOMETRY_FEATURES = [
    "is_clearance",
    "contest_x",
    "contest_y_from_centre",
    "pass_length",
    "pass_is_high",
    "pass_is_low",

    "winner_support_5m",
    "loser_support_5m",
    "winner_support_10m",
    "loser_support_10m",
    "winner_support_15m",
    "loser_support_15m",

    "contestant_distance",
    "nearest_winner_support",
    "nearest_loser_support",
    "second_winner_support",
    "second_loser_support",
    "winner_support_spread",
    "loser_support_spread",
    "winner_angular_concentration",
    "loser_angular_concentration",
    "winner_centroid_dx",
    "winner_centroid_dy",
    "loser_centroid_dx",
    "loser_centroid_dy",
    "support_centroid_gap",
]


LANDING_FEATURES = [
    "landing_dx",
    "landing_dy_abs",
    "landing_visible",
    "winner_nearest_landing",
    "loser_nearest_landing",
    "winner_second_landing",
    "loser_second_landing",
    "landing_nearest_advantage",
    "landing_top2_advantage",
    "winner_count_5m_landing",
    "loser_count_5m_landing",
    "winner_count_10m_landing",
    "loser_count_10m_landing",
    "landing_count_adv_5m",
    "landing_count_adv_10m",
    "winner_density_landing",
    "loser_density_landing",
    "landing_density_advantage",
    "winner_corridor_count",
    "loser_corridor_count",
    "corridor_advantage",
    "winner_second_wave",
    "loser_second_wave",
    "second_wave_advantage",
]


# ============================================================
# HELPERS
# ============================================================

def to_float(value):
    if value in {
        None,
        "",
        "None",
        "nan",
        "NaN",
    }:
        return np.nan
    return float(value)


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1 - 1e-6,
    )

    return {
        "log_loss": log_loss(
            y,
            p,
            labels=[0, 1],
        ),
        "brier": brier_score_loss(
            y,
            p,
        ),
        "roc_auc": roc_auc_score(
            y,
            p,
        ),
        "pr_auc": average_precision_score(
            y,
            p,
        ),
    }


def new_model():
    return XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.07,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )


def distance_xy(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def point_in_polygon(x, y, polygon):
    if not polygon or len(polygon) < 6:
        return np.nan

    pts = [
        (
            float(polygon[i]),
            float(polygon[i + 1]),
        )
        for i in range(
            0,
            len(polygon) - 1,
            2,
        )
    ]

    inside = False
    j = len(pts) - 1

    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]

        if (yi > y) != (yj > y):
            x_cross = (
                (xj - xi)
                * (y - yi)
                / ((yj - yi) + 1e-12)
                + xi
            )

            if x < x_cross:
                inside = not inside

        j = i

    return 1.0 if inside else 0.0


def point_segment_distance_and_projection(
    point,
    start,
    end,
):
    px, py = map(float, point)
    x1, y1 = map(float, start)
    x2, y2 = map(float, end)

    vx = x2 - x1
    vy = y2 - y1

    denom = vx * vx + vy * vy

    if denom <= 1e-12:
        return (
            distance_xy(point, start),
            0.0,
        )

    t = (
        (px - x1) * vx
        + (py - y1) * vy
    ) / denom

    nearest_x = x1 + t * vx
    nearest_y = y1 + t * vy

    return (
        math.hypot(
            px - nearest_x,
            py - nearest_y,
        ),
        t,
    )


# ============================================================
# LOAD FEATURE TABLE
# ============================================================

with open(
    FEATURE_PATH,
    "r",
    encoding="utf-8",
) as f:
    rows = list(csv.DictReader(f))


for row in rows:
    row["contest_y_from_centre"] = abs(
        float(row["contest_y"]) - 40.0
    )


# ============================================================
# LOAD PASS OUTCOMES
# ============================================================

match_ids = sorted(
    set(
        str(row["match_id"])
        for row in rows
    )
)

events_by_match = {}
event_lookup = {}


print("=" * 104)
print("TRUE SECOND-BALL MODEL REBUILD")
print("=" * 104)

print()
print(
    f"Loading event data for "
    f"{len(match_ids):,} matches..."
)


for i, match_id in enumerate(
    match_ids,
    start=1,
):
    path = (
        EVENT_DIR
        / f"{match_id}.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        events = json.load(f)

    events_by_match[
        match_id
    ] = events

    event_lookup[
        match_id
    ] = {
        event["id"]: event
        for event in events
    }

    if (
        i % 50 == 0
        or i == len(match_ids)
    ):
        print(
            f"  events loaded "
            f"{i}/{len(match_ids)}"
        )


# ============================================================
# STRICT FOOTBALL SAMPLE
# ============================================================

strict_rows = []
excluded_completed = 0
excluded_other_pass = 0


for row in rows:

    action = row[
        "winner_event_type"
    ]

    if action == "Clearance":
        strict_rows.append(
            dict(row)
        )
        continue

    if action != "Pass":
        continue

    match_id = str(
        row["match_id"]
    )

    event_id = row[
        "winner_event_id"
    ]

    event = event_lookup[
        match_id
    ][event_id]

    pass_data = event.get(
        "pass",
        {},
    )

    outcome = pass_data.get(
        "outcome"
    )

    if isinstance(
        outcome,
        dict,
    ):
        outcome_name = outcome.get(
            "name",
            "Unknown",
        )
    else:
        outcome_name = "Complete"

    if outcome_name == "Incomplete":
        new_row = dict(row)
        new_row[
            "strict_pass_outcome"
        ] = outcome_name

        strict_rows.append(
            new_row
        )

    elif outcome_name == "Complete":
        excluded_completed += 1

    else:
        excluded_other_pass += 1


print()
print("STRICT SAMPLE DEFINITION")
print()

print(
    "  Included: resolved Clearances + "
    "Incomplete aerial Passes"
)

print(
    "  Excluded: completed headed passes, "
    "Out/Unknown pass outcomes"
)

print()
print(
    f"Strict contests:      "
    f"{len(strict_rows):,}"
)

print(
    f"Unique matches:       "
    f"{len(set(r['match_id'] for r in strict_rows)):,}"
)

print(
    f"Completed passes removed: "
    f"{excluded_completed:,}"
)

print(
    f"Other pass outcomes removed: "
    f"{excluded_other_pass:,}"
)


clearance_n = sum(
    row[
        "winner_event_type"
    ] == "Clearance"
    for row in strict_rows
)

incomplete_pass_n = sum(
    row[
        "winner_event_type"
    ] == "Pass"
    for row in strict_rows
)

y = np.asarray(
    [
        int(row["target"])
        for row in strict_rows
    ],
    dtype=int,
)


print(
    f"  Clearances:         "
    f"{clearance_n:,}"
)

print(
    f"  Incomplete passes:  "
    f"{incomplete_pass_n:,}"
)

print(
    f"Winner-team recovery: "
    f"{int(y.sum()):,}/{len(y):,} "
    f"({100*y.mean():.1f}%)"
)


# ============================================================
# LOAD 360 ONLY FOR STRICT MATCHES
# ============================================================

strict_match_ids = sorted(
    set(
        str(
            row["match_id"]
        )
        for row in strict_rows
    )
)

frames_by_match = {}


print()
print(
    "Loading 360 frames for strict sample..."
)


for i, match_id in enumerate(
    strict_match_ids,
    start=1,
):
    path = (
        THREE_SIXTY_DIR
        / f"{match_id}.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        frames = json.load(f)

    frames_by_match[
        match_id
    ] = {
        frame[
            "event_uuid"
        ]: frame
        for frame in frames
    }

    if (
        i % 50 == 0
        or i == len(
            strict_match_ids
        )
    ):
        print(
            f"  360 loaded "
            f"{i}/{len(strict_match_ids)}"
        )


# ============================================================
# LANDING FEATURES FOR INCOMPLETE PASSES
# ============================================================

landing_by_event = {}


print()
print(
    "Building contact-aware landing features..."
)


for i, row in enumerate(
    strict_rows,
    start=1,
):

    event_id = row[
        "winner_event_id"
    ]

    features = {
        name: np.nan
        for name in LANDING_FEATURES
    }

    # Landing endpoint only exists for the aerial Pass cases.
    if row[
        "winner_event_type"
    ] == "Pass":

        match_id = str(
            row["match_id"]
        )

        event = event_lookup[
            match_id
        ][event_id]

        frame = frames_by_match[
            match_id
        ].get(
            event_id
        )

        pass_data = event.get(
            "pass",
            {},
        )

        landing = pass_data.get(
            "end_location"
        )

        if (
            frame is not None
            and landing
        ):
            contest = [
                float(
                    row["contest_x"]
                ),
                float(
                    row["contest_y"]
                ),
            ]

            landing = [
                float(
                    landing[0]
                ),
                float(
                    landing[1]
                ),
            ]

            features[
                "landing_dx"
            ] = (
                landing[0]
                - contest[0]
            )

            features[
                "landing_dy_abs"
            ] = abs(
                landing[1]
                - contest[1]
            )

            features[
                "landing_visible"
            ] = point_in_polygon(
                landing[0],
                landing[1],
                frame.get(
                    "visible_area"
                ),
            )

            winner_points = []
            loser_points = []

            for player in frame.get(
                "freeze_frame",
                [],
            ):
                loc = player.get(
                    "location"
                )

                if not loc:
                    continue

                point = [
                    float(loc[0]),
                    float(loc[1]),
                ]

                if player.get(
                    "teammate",
                    False,
                ):
                    winner_points.append(
                        point
                    )
                else:
                    loser_points.append(
                        point
                    )

            winner_d = sorted(
                distance_xy(
                    point,
                    landing,
                )
                for point in winner_points
            )

            loser_d = sorted(
                distance_xy(
                    point,
                    landing,
                )
                for point in loser_points
            )

            def kth(values, k):
                if len(values) < k:
                    return np.nan
                return float(
                    values[
                        k - 1
                    ]
                )

            features[
                "winner_nearest_landing"
            ] = kth(
                winner_d,
                1,
            )

            features[
                "loser_nearest_landing"
            ] = kth(
                loser_d,
                1,
            )

            features[
                "winner_second_landing"
            ] = kth(
                winner_d,
                2,
            )

            features[
                "loser_second_landing"
            ] = kth(
                loser_d,
                2,
            )

            if (
                len(winner_d) >= 1
                and len(loser_d) >= 1
            ):
                features[
                    "landing_nearest_advantage"
                ] = (
                    loser_d[0]
                    - winner_d[0]
                )

            if (
                len(winner_d) >= 2
                and len(loser_d) >= 2
            ):
                features[
                    "landing_top2_advantage"
                ] = (
                    np.mean(
                        loser_d[:2]
                    )
                    -
                    np.mean(
                        winner_d[:2]
                    )
                )

            for radius, suffix in [
                (5.0, "5m"),
                (10.0, "10m"),
            ]:
                winner_count = sum(
                    d <= radius
                    for d in winner_d
                )

                loser_count = sum(
                    d <= radius
                    for d in loser_d
                )

                features[
                    f"winner_count_{suffix}_landing"
                ] = float(
                    winner_count
                )

                features[
                    f"loser_count_{suffix}_landing"
                ] = float(
                    loser_count
                )

                features[
                    f"landing_count_adv_{suffix}"
                ] = float(
                    winner_count
                    - loser_count
                )

            tau = 5.0

            winner_density = sum(
                math.exp(
                    -d / tau
                )
                for d in winner_d
            )

            loser_density = sum(
                math.exp(
                    -d / tau
                )
                for d in loser_d
            )

            features[
                "winner_density_landing"
            ] = float(
                winner_density
            )

            features[
                "loser_density_landing"
            ] = float(
                loser_density
            )

            features[
                "landing_density_advantage"
            ] = float(
                winner_density
                -
                loser_density
            )

            winner_corridor = 0
            loser_corridor = 0

            for point in winner_points:
                perp, t = (
                    point_segment_distance_and_projection(
                        point,
                        contest,
                        landing,
                    )
                )

                if (
                    0.0 <= t <= 1.15
                    and perp <= 5.0
                ):
                    winner_corridor += 1

            for point in loser_points:
                perp, t = (
                    point_segment_distance_and_projection(
                        point,
                        contest,
                        landing,
                    )
                )

                if (
                    0.0 <= t <= 1.15
                    and perp <= 5.0
                ):
                    loser_corridor += 1

            features[
                "winner_corridor_count"
            ] = float(
                winner_corridor
            )

            features[
                "loser_corridor_count"
            ] = float(
                loser_corridor
            )

            features[
                "corridor_advantage"
            ] = float(
                winner_corridor
                -
                loser_corridor
            )

            winner_second_wave = sum(
                15.0
                < distance_xy(
                    point,
                    contest,
                )
                <= 25.0
                for point in winner_points
            )

            loser_second_wave = sum(
                15.0
                < distance_xy(
                    point,
                    contest,
                )
                <= 25.0
                for point in loser_points
            )

            features[
                "winner_second_wave"
            ] = float(
                winner_second_wave
            )

            features[
                "loser_second_wave"
            ] = float(
                loser_second_wave
            )

            features[
                "second_wave_advantage"
            ] = float(
                winner_second_wave
                -
                loser_second_wave
            )

    landing_by_event[
        event_id
    ] = features

    if (
        i % 500 == 0
        or i == len(strict_rows)
    ):
        print(
            f"  processed "
            f"{i}/{len(strict_rows)} contests"
        )


# ============================================================
# MATRIX BUILDERS
# ============================================================

def matrix(
    rows,
    feature_names,
):
    data = []

    for row in rows:
        event_id = row[
            "winner_event_id"
        ]

        landing = landing_by_event[
            event_id
        ]

        values = []

        for feature in feature_names:

            if feature in row:
                values.append(
                    to_float(
                        row[feature]
                    )
                )
            else:
                values.append(
                    float(
                        landing[
                            feature
                        ]
                    )
                )

        data.append(
            values
        )

    return np.asarray(
        data,
        dtype=float,
    )


X_event = matrix(
    strict_rows,
    EVENT_FEATURES,
)

X_geometry = matrix(
    strict_rows,
    GEOMETRY_FEATURES,
)

X_contact = matrix(
    strict_rows,
    GEOMETRY_FEATURES
    + LANDING_FEATURES,
)

groups = np.asarray(
    [
        row["match_id"]
        for row in strict_rows
    ]
)


# ============================================================
# OOF CV
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)


pred_event = np.full(
    len(strict_rows),
    np.nan,
)

pred_geometry = np.full(
    len(strict_rows),
    np.nan,
)

pred_contact = np.full(
    len(strict_rows),
    np.nan,
)

pred_prior = np.full(
    len(strict_rows),
    np.nan,
)


print()
print("=" * 104)
print("WHOLE-MATCH OOF MODEL COMPARISON")
print("=" * 104)

print()


for fold, (
    train_idx,
    val_idx,
) in enumerate(
    cv.split(
        X_geometry,
        y,
        groups,
    ),
    start=1,
):

    train_prevalence = float(
        y[
            train_idx
        ].mean()
    )

    pred_prior[
        val_idx
    ] = train_prevalence

    event_model = new_model()

    event_model.fit(
        X_event[
            train_idx
        ],
        y[
            train_idx
        ],
    )

    pred_event[
        val_idx
    ] = (
        event_model
        .predict_proba(
            X_event[
                val_idx
            ]
        )[:, 1]
    )

    geometry_model = new_model()

    geometry_model.fit(
        X_geometry[
            train_idx
        ],
        y[
            train_idx
        ],
    )

    pred_geometry[
        val_idx
    ] = (
        geometry_model
        .predict_proba(
            X_geometry[
                val_idx
            ]
        )[:, 1]
    )

    contact_model = new_model()

    contact_model.fit(
        X_contact[
            train_idx
        ],
        y[
            train_idx
        ],
    )

    pred_contact[
        val_idx
    ] = (
        contact_model
        .predict_proba(
            X_contact[
                val_idx
            ]
        )[:, 1]
    )

    fold_prior = metrics(
        y[val_idx],
        pred_prior[val_idx],
    )

    fold_event = metrics(
        y[val_idx],
        pred_event[val_idx],
    )

    fold_geometry = metrics(
        y[val_idx],
        pred_geometry[val_idx],
    )

    fold_contact = metrics(
        y[val_idx],
        pred_contact[val_idx],
    )

    print(
        f"Fold {fold}: "
        f"prior={fold_prior['log_loss']:.4f} | "
        f"event={fold_event['log_loss']:.4f} | "
        f"pre-contact={fold_geometry['log_loss']:.4f} | "
        f"contact-aware={fold_contact['log_loss']:.4f}"
    )


# ============================================================
# OVERALL RESULTS
# ============================================================

results = {
    "Prior": metrics(
        y,
        pred_prior,
    ),
    "Event context": metrics(
        y,
        pred_event,
    ),
    "Pre-contact geometry": metrics(
        y,
        pred_geometry,
    ),
    "Contact-aware landing": metrics(
        y,
        pred_contact,
    ),
}


print()
print("=" * 104)
print("TRUE SECOND-BALL OOF RESULTS")
print("=" * 104)

print()
print(
    f"Positive-class prevalence: "
    f"{100*y.mean():.1f}%"
)

print(
    "(PR-AUC baseline is approximately the prevalence.)"
)

print()


for name, result in results.items():
    print(
        f"{name:<24}"
        f"LL={result['log_loss']:.4f} | "
        f"ROC-AUC={result['roc_auc']:.4f} | "
        f"PR-AUC={result['pr_auc']:.4f} | "
        f"Brier={result['brier']:.4f}"
    )


# ============================================================
# ACTION BREAKDOWN
# ============================================================

print()
print("BY FIRST-CONTACT TYPE")

for action in [
    "Pass",
    "Clearance",
]:

    idx = np.asarray(
        [
            i
            for i, row in enumerate(
                strict_rows
            )
            if row[
                "winner_event_type"
            ] == action
        ],
        dtype=int,
    )

    print()
    print(
        f"{action} "
        f"(n={len(idx):,}, "
        f"positive={100*y[idx].mean():.1f}%)"
    )

    for name, pred in [
        (
            "Event context",
            pred_event,
        ),
        (
            "Pre-contact geometry",
            pred_geometry,
        ),
        (
            "Contact-aware landing",
            pred_contact,
        ),
    ]:
        result = metrics(
            y[idx],
            pred[idx],
        )

        print(
            f"  {name:<22}"
            f"LL={result['log_loss']:.4f} | "
            f"ROC={result['roc_auc']:.4f} | "
            f"PR={result['pr_auc']:.4f}"
        )


# ============================================================
# PAIRED MATCH BOOTSTRAP
# ============================================================

indices_by_match = defaultdict(
    list
)

for i, match_id in enumerate(
    groups
):
    indices_by_match[
        match_id
    ].append(i)

unique_matches = list(
    indices_by_match.keys()
)

rng = random.Random(
    20260918
)


comparisons = {
    "event_minus_prior": (
        pred_event,
        pred_prior,
    ),
    "precontact_minus_event": (
        pred_geometry,
        pred_event,
    ),
    "contact_minus_precontact": (
        pred_contact,
        pred_geometry,
    ),
}


bootstrap_rows = []


print()
print("=" * 104)
print("PAIRED MATCH-BOOTSTRAP")
print("=" * 104)

print()
print(
    "Negative delta means the first named model is better."
)


for comparison, (
    first_pred,
    second_pred,
) in comparisons.items():

    deltas = []

    for _ in range(
        N_BOOTSTRAP
    ):

        sampled_matches = [
            rng.choice(
                unique_matches
            )
            for _ in unique_matches
        ]

        idx = []

        for match_id in sampled_matches:
            idx.extend(
                indices_by_match[
                    match_id
                ]
            )

        idx = np.asarray(
            idx,
            dtype=int,
        )

        first_ll = log_loss(
            y[idx],
            first_pred[idx],
            labels=[0, 1],
        )

        second_ll = log_loss(
            y[idx],
            second_pred[idx],
            labels=[0, 1],
        )

        deltas.append(
            first_ll
            -
            second_ll
        )

    deltas = np.asarray(
        deltas,
        dtype=float,
    )

    low = float(
        np.percentile(
            deltas,
            2.5,
        )
    )

    high = float(
        np.percentile(
            deltas,
            97.5,
        )
    )

    fraction_better = float(
        np.mean(
            deltas < 0
        )
    )

    print(
        f"  {comparison:<30}"
        f"mean={deltas.mean():+.4f} | "
        f"95% CI "
        f"[{low:+.4f}, {high:+.4f}] | "
        f"better="
        f"{100*fraction_better:.1f}%"
    )

    bootstrap_rows.append(
        {
            "comparison": comparison,
            "mean_difference": float(
                deltas.mean()
            ),
            "ci_2_5": low,
            "ci_97_5": high,
            "bootstrap_fraction_first_better": (
                fraction_better
            ),
            "n_bootstrap": N_BOOTSTRAP,
        }
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

result_rows = []

for name, result in results.items():
    result_rows.append(
        {
            "model": name,
            "n": len(y),
            "positive_prevalence": float(
                y.mean()
            ),
            **result,
        }
    )


with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=result_rows[
            0
        ].keys(),
    )

    writer.writeheader()
    writer.writerows(
        result_rows
    )


with open(
    OOF_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "match_id",
        "winner_event_id",
        "winner_event_type",
        "target",
        "p_prior",
        "p_event",
        "p_precontact_geometry",
        "p_contact_aware",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for i, row in enumerate(
        strict_rows
    ):
        writer.writerow(
            {
                "match_id": row[
                    "match_id"
                ],
                "winner_event_id": row[
                    "winner_event_id"
                ],
                "winner_event_type": row[
                    "winner_event_type"
                ],
                "target": int(
                    row["target"]
                ),
                "p_prior": float(
                    pred_prior[i]
                ),
                "p_event": float(
                    pred_event[i]
                ),
                "p_precontact_geometry": float(
                    pred_geometry[i]
                ),
                "p_contact_aware": float(
                    pred_contact[i]
                ),
            }
        )


with open(
    BOOTSTRAP_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=bootstrap_rows[
            0
        ].keys(),
    )

    writer.writeheader()
    writer.writerows(
        bootstrap_rows
    )


with open(
    AUDIT_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "match_id",
        "winner_event_id",
        "winner_event_type",
        "target",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row in strict_rows:
        writer.writerow(
            {
                key: row[key]
                for key in fieldnames
            }
        )


print()
print("Saved:")
print(f"  {RESULTS_PATH}")
print(f"  {OOF_PATH}")
print(f"  {BOOTSTRAP_PATH}")
print(f"  {AUDIT_PATH}")

print()
print("=" * 104)
print("DONE")
print("=" * 104)
