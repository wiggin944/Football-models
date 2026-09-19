import csv
import json
import math
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
STRICT_PATH = OUTPUT_DIR / "true_second_ball_strict_sample.csv"

RESULTS_PATH = OUTPUT_DIR / "information_stage_results.csv"
OOF_PATH = OUTPUT_DIR / "information_stage_oof_predictions.csv"

SEED = 42
N_SPLITS = 5


# ============================================================
# FEATURE STAGES
# ============================================================

LOCATION_FEATURES = [
    "contest_x",
    "contest_y_from_centre",
]


SETUP_GEOMETRY = [
    "contest_x",
    "contest_y_from_centre",

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


INCOMING_FEATURES = [
    "incoming_found",
    "incoming_length",
    "incoming_start_x",
    "incoming_start_y_from_centre",
    "incoming_source_same_winner",
    "incoming_seconds_before",
    "incoming_high",
    "incoming_low",
    "incoming_ground",
    "incoming_cross",
    "incoming_switch",
    "incoming_goal_kick",
    "incoming_free_kick",
    "incoming_corner",
    "incoming_under_pressure",

    "ahead_adv",
    "behind_adv",
    "central_adv",
    "ahead_corridor_adv",
    "behind_corridor_adv",
    "mean_along_adv",
    "flight_line_proximity_adv",
]


OUTGOING_FEATURES = [
    "is_clearance",
    "pass_length",
    "pass_is_high",
    "pass_is_low",
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


def event_time_seconds(event):
    return (
        float(event.get("minute", 0) or 0) * 60.0
        + float(event.get("second", 0) or 0)
    )


def mirror(point):
    return [
        120.0 - float(point[0]),
        80.0 - float(point[1]),
    ]


def distance_xy(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


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
        "roc_auc": roc_auc_score(
            y,
            p,
        ),
        "pr_auc": average_precision_score(
            y,
            p,
        ),
        "brier": brier_score_loss(
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

    nx = x1 + t * vx
    ny = y1 + t * vy

    return (
        math.hypot(
            px - nx,
            py - ny,
        ),
        t,
    )


# ============================================================
# LOAD STRICT SAMPLE + FEATURE TABLE
# ============================================================

with open(
    STRICT_PATH,
    "r",
    encoding="utf-8",
) as f:
    strict_ids = {
        row["winner_event_id"]
        for row in csv.DictReader(f)
    }


with open(
    FEATURE_PATH,
    "r",
    encoding="utf-8",
) as f:
    all_rows = list(
        csv.DictReader(f)
    )


rows = [
    row
    for row in all_rows
    if row[
        "winner_event_id"
    ] in strict_ids
]


for row in rows:
    row[
        "contest_y_from_centre"
    ] = abs(
        float(
            row["contest_y"]
        ) - 40.0
    )


# ============================================================
# LOAD EVENTS + 360
# ============================================================

match_ids = sorted(
    set(
        str(row["match_id"])
        for row in rows
    )
)


events_by_match = {}
event_lookup = {}
event_index = {}
frames_by_match = {}


print("=" * 104)
print("TRUE SECOND-BALL INFORMATION-STAGE DECOMPOSITION")
print("=" * 104)

print()
print(
    f"Strict contests: {len(rows):,}"
)

print(
    f"Matches:         {len(match_ids):,}"
)

print()
print("Loading event + 360 data...")


for i, match_id in enumerate(
    match_ids,
    start=1,
):

    with open(
        EVENT_DIR / f"{match_id}.json",
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

    event_index[
        match_id
    ] = {
        event["id"]: idx
        for idx, event in enumerate(events)
    }

    with open(
        THREE_SIXTY_DIR / f"{match_id}.json",
        "r",
        encoding="utf-8",
    ) as f:
        frames = json.load(f)

    frames_by_match[
        match_id
    ] = {
        frame["event_uuid"]: frame
        for frame in frames
    }

    if (
        i % 50 == 0
        or i == len(match_ids)
    ):
        print(
            f"  loaded "
            f"{i}/{len(match_ids)}"
        )


# ============================================================
# INCOMING + LANDING CONTEXT
# ============================================================

def canonicalise_pass(event, contest_xy):
    pass_data = event.get(
        "pass"
    )

    if not isinstance(
        pass_data,
        dict,
    ):
        return None

    start = event.get(
        "location"
    )

    end = pass_data.get(
        "end_location"
    )

    if not start or not end:
        return None

    direct_start = [
        float(start[0]),
        float(start[1]),
    ]

    direct_end = [
        float(end[0]),
        float(end[1]),
    ]

    mirror_start = mirror(
        direct_start
    )

    mirror_end = mirror(
        direct_end
    )

    d_direct = distance_xy(
        direct_end,
        contest_xy,
    )

    d_mirror = distance_xy(
        mirror_end,
        contest_xy,
    )

    if d_direct <= d_mirror:
        return {
            "start": direct_start,
            "end": direct_end,
            "endpoint_error": d_direct,
        }

    return {
        "start": mirror_start,
        "end": mirror_end,
        "endpoint_error": d_mirror,
    }


def find_incoming(
    row,
):
    match_id = str(
        row["match_id"]
    )

    event_id = row[
        "winner_event_id"
    ]

    events = events_by_match[
        match_id
    ]

    idx = event_index[
        match_id
    ][event_id]

    winner_event = events[
        idx
    ]

    winner_team = (
        winner_event.get(
            "team",
            {},
        ).get("id")
    )

    winner_time = event_time_seconds(
        winner_event
    )

    contest = [
        float(row["contest_x"]),
        float(row["contest_y"]),
    ]

    best = None

    for j in range(
        idx - 1,
        max(
            -1,
            idx - 31,
        ),
        -1,
    ):
        event = events[j]

        if event.get(
            "type",
            {},
        ).get("name") != "Pass":
            continue

        dt = (
            winner_time
            - event_time_seconds(
                event
            )
        )

        if (
            dt < -0.01
            or dt > 12.0
        ):
            continue

        canonical = canonicalise_pass(
            event,
            contest,
        )

        if canonical is None:
            continue

        if (
            canonical[
                "endpoint_error"
            ] > 12.0
        ):
            continue

        score = (
            canonical[
                "endpoint_error"
            ]
            + 0.08 * dt
        )

        candidate = {
            "event": event,
            "winner_team": winner_team,
            "dt": dt,
            "score": score,
            **canonical,
        }

        if (
            best is None
            or score < best[
                "score"
            ]
        ):
            best = candidate

    return best


def support_points(
    frame,
):
    winner = []
    loser = []

    for player in frame.get(
        "freeze_frame",
        [],
    ):
        location = player.get(
            "location"
        )

        if not location:
            continue

        point = [
            float(
                location[0]
            ),
            float(
                location[1]
            ),
        ]

        if player.get(
            "teammate",
            False,
        ):
            winner.append(
                point
            )
        else:
            loser.append(
                point
            )

    return winner, loser


def incoming_features(
    row,
    frame,
):
    out = {
        name: np.nan
        for name in INCOMING_FEATURES
    }

    out[
        "incoming_found"
    ] = 0.0

    incoming = find_incoming(
        row
    )

    if incoming is None:
        return out

    event = incoming[
        "event"
    ]

    pass_data = event.get(
        "pass",
        {},
    )

    start = incoming[
        "start"
    ]

    end = incoming[
        "end"
    ]

    vx = (
        end[0]
        - start[0]
    )

    vy = (
        end[1]
        - start[1]
    )

    norm = math.hypot(
        vx,
        vy,
    )

    if norm <= 1e-6:
        return out

    ux = vx / norm
    uy = vy / norm

    out[
        "incoming_found"
    ] = 1.0

    out[
        "incoming_length"
    ] = float(
        pass_data.get(
            "length",
            norm,
        )
        or norm
    )

    out[
        "incoming_start_x"
    ] = float(
        start[0]
    )

    out[
        "incoming_start_y_from_centre"
    ] = abs(
        float(start[1])
        - 40.0
    )

    out[
        "incoming_seconds_before"
    ] = float(
        incoming["dt"]
    )

    source_team = (
        event.get(
            "team",
            {},
        ).get("id")
    )

    out[
        "incoming_source_same_winner"
    ] = (
        1.0
        if (
            source_team
            == incoming[
                "winner_team"
            ]
        )
        else 0.0
    )

    height = pass_data.get(
        "height"
    )

    height_name = (
        height.get("name")
        if isinstance(
            height,
            dict,
        )
        else None
    )

    out[
        "incoming_high"
    ] = (
        1.0
        if height_name
        == "High Pass"
        else 0.0
    )

    out[
        "incoming_low"
    ] = (
        1.0
        if height_name
        == "Low Pass"
        else 0.0
    )

    out[
        "incoming_ground"
    ] = (
        1.0
        if height_name
        == "Ground Pass"
        else 0.0
    )

    out[
        "incoming_cross"
    ] = (
        1.0
        if pass_data.get(
            "cross",
            False,
        )
        else 0.0
    )

    out[
        "incoming_switch"
    ] = (
        1.0
        if pass_data.get(
            "switch",
            False,
        )
        else 0.0
    )

    pass_type = pass_data.get(
        "type"
    )

    pass_type_name = (
        pass_type.get("name")
        if isinstance(
            pass_type,
            dict,
        )
        else None
    )

    out[
        "incoming_goal_kick"
    ] = (
        1.0
        if pass_type_name
        == "Goal Kick"
        else 0.0
    )

    out[
        "incoming_free_kick"
    ] = (
        1.0
        if pass_type_name
        == "Free Kick"
        else 0.0
    )

    out[
        "incoming_corner"
    ] = (
        1.0
        if pass_type_name
        == "Corner"
        else 0.0
    )

    out[
        "incoming_under_pressure"
    ] = (
        1.0
        if event.get(
            "under_pressure",
            False,
        )
        else 0.0
    )

    winner_points, loser_points = (
        support_points(
            frame
        )
    )

    contest = np.asarray(
        [
            float(
                row[
                    "contest_x"
                ]
            ),
            float(
                row[
                    "contest_y"
                ]
            ),
        ],
        dtype=float,
    )

    def rotate(points):
        result = []

        for point in points:
            dx = (
                point[0]
                - contest[0]
            )

            dy = (
                point[1]
                - contest[1]
            )

            along = (
                dx * ux
                + dy * uy
            )

            cross = (
                -dx * uy
                + dy * ux
            )

            result.append(
                (
                    along,
                    cross,
                )
            )

        return result

    winner_rot = rotate(
        winner_points
    )

    loser_rot = rotate(
        loser_points
    )

    def compact(rotated):
        if not rotated:
            return {
                "ahead": 0.0,
                "behind": 0.0,
                "central": 0.0,
                "ahead_corridor": 0.0,
                "behind_corridor": 0.0,
                "mean_along": np.nan,
                "nearest_line": np.nan,
            }

        along = np.asarray(
            [
                p[0]
                for p in rotated
            ],
            dtype=float,
        )

        cross = np.asarray(
            [
                p[1]
                for p in rotated
            ],
            dtype=float,
        )

        abs_cross = np.abs(
            cross
        )

        return {
            "ahead": float(
                np.sum(
                    along > 2.0
                )
            ),
            "behind": float(
                np.sum(
                    along < -2.0
                )
            ),
            "central": float(
                np.sum(
                    abs_cross <= 4.0
                )
            ),
            "ahead_corridor": float(
                np.sum(
                    (along > 0)
                    &
                    (abs_cross <= 5.0)
                )
            ),
            "behind_corridor": float(
                np.sum(
                    (along < 0)
                    &
                    (abs_cross <= 5.0)
                )
            ),
            "mean_along": float(
                np.mean(
                    along
                )
            ),
            "nearest_line": float(
                np.min(
                    abs_cross
                )
            ),
        }

    w = compact(
        winner_rot
    )

    l = compact(
        loser_rot
    )

    out[
        "ahead_adv"
    ] = (
        w["ahead"]
        - l["ahead"]
    )

    out[
        "behind_adv"
    ] = (
        w["behind"]
        - l["behind"]
    )

    out[
        "central_adv"
    ] = (
        w["central"]
        - l["central"]
    )

    out[
        "ahead_corridor_adv"
    ] = (
        w[
            "ahead_corridor"
        ]
        - l[
            "ahead_corridor"
        ]
    )

    out[
        "behind_corridor_adv"
    ] = (
        w[
            "behind_corridor"
        ]
        - l[
            "behind_corridor"
        ]
    )

    out[
        "mean_along_adv"
    ] = (
        w["mean_along"]
        - l["mean_along"]
    )

    out[
        "flight_line_proximity_adv"
    ] = (
        l["nearest_line"]
        - w["nearest_line"]
    )

    return out


def landing_features(
    row,
    frame,
):
    out = {
        name: np.nan
        for name in LANDING_FEATURES
    }

    if row[
        "winner_event_type"
    ] != "Pass":
        return out

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

    landing = pass_data.get(
        "end_location"
    )

    if not landing:
        return out

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

    out[
        "landing_dx"
    ] = (
        landing[0]
        - contest[0]
    )

    out[
        "landing_dy_abs"
    ] = abs(
        landing[1]
        - contest[1]
    )

    out[
        "landing_visible"
    ] = point_in_polygon(
        landing[0],
        landing[1],
        frame.get(
            "visible_area"
        ),
    )

    winner_points, loser_points = (
        support_points(
            frame
        )
    )

    winner_d = sorted(
        distance_xy(
            p,
            landing,
        )
        for p in winner_points
    )

    loser_d = sorted(
        distance_xy(
            p,
            landing,
        )
        for p in loser_points
    )

    def kth(values, k):
        return (
            float(
                values[
                    k - 1
                ]
            )
            if len(values) >= k
            else np.nan
        )

    out[
        "winner_nearest_landing"
    ] = kth(
        winner_d,
        1,
    )

    out[
        "loser_nearest_landing"
    ] = kth(
        loser_d,
        1,
    )

    out[
        "winner_second_landing"
    ] = kth(
        winner_d,
        2,
    )

    out[
        "loser_second_landing"
    ] = kth(
        loser_d,
        2,
    )

    if (
        len(winner_d) >= 1
        and len(loser_d) >= 1
    ):
        out[
            "landing_nearest_advantage"
        ] = (
            loser_d[0]
            - winner_d[0]
        )

    if (
        len(winner_d) >= 2
        and len(loser_d) >= 2
    ):
        out[
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
        wc = sum(
            d <= radius
            for d in winner_d
        )

        lc = sum(
            d <= radius
            for d in loser_d
        )

        out[
            f"winner_count_{suffix}_landing"
        ] = float(
            wc
        )

        out[
            f"loser_count_{suffix}_landing"
        ] = float(
            lc
        )

        out[
            f"landing_count_adv_{suffix}"
        ] = float(
            wc - lc
        )

    tau = 5.0

    wd = sum(
        math.exp(
            -d / tau
        )
        for d in winner_d
    )

    ld = sum(
        math.exp(
            -d / tau
        )
        for d in loser_d
    )

    out[
        "winner_density_landing"
    ] = float(
        wd
    )

    out[
        "loser_density_landing"
    ] = float(
        ld
    )

    out[
        "landing_density_advantage"
    ] = float(
        wd - ld
    )

    winner_corridor = 0
    loser_corridor = 0

    for p in winner_points:
        perp, t = (
            point_segment_distance_and_projection(
                p,
                contest,
                landing,
            )
        )

        if (
            0 <= t <= 1.15
            and perp <= 5.0
        ):
            winner_corridor += 1

    for p in loser_points:
        perp, t = (
            point_segment_distance_and_projection(
                p,
                contest,
                landing,
            )
        )

        if (
            0 <= t <= 1.15
            and perp <= 5.0
        ):
            loser_corridor += 1

    out[
        "winner_corridor_count"
    ] = float(
        winner_corridor
    )

    out[
        "loser_corridor_count"
    ] = float(
        loser_corridor
    )

    out[
        "corridor_advantage"
    ] = float(
        winner_corridor
        - loser_corridor
    )

    winner_second_wave = sum(
        15.0
        < distance_xy(
            p,
            contest,
        )
        <= 25.0
        for p in winner_points
    )

    loser_second_wave = sum(
        15.0
        < distance_xy(
            p,
            contest,
        )
        <= 25.0
        for p in loser_points
    )

    out[
        "winner_second_wave"
    ] = float(
        winner_second_wave
    )

    out[
        "loser_second_wave"
    ] = float(
        loser_second_wave
    )

    out[
        "second_wave_advantage"
    ] = float(
        winner_second_wave
        - loser_second_wave
    )

    return out


# ============================================================
# BUILD CONTEXT TABLE
# ============================================================

context_by_event = {}


print()
print(
    "Building incoming + landing context..."
)


for i, row in enumerate(
    rows,
    start=1,
):
    match_id = str(
        row["match_id"]
    )

    event_id = row[
        "winner_event_id"
    ]

    frame = frames_by_match[
        match_id
    ][event_id]

    context = {}

    context.update(
        incoming_features(
            row,
            frame,
        )
    )

    context.update(
        landing_features(
            row,
            frame,
        )
    )

    context_by_event[
        event_id
    ] = context

    if (
        i % 500 == 0
        or i == len(rows)
    ):
        print(
            f"  processed "
            f"{i}/{len(rows)}"
        )


# ============================================================
# MATRICES
# ============================================================

def matrix(
    feature_names,
):
    data = []

    for row in rows:
        context = context_by_event[
            row[
                "winner_event_id"
            ]
        ]

        values = []

        for feature in feature_names:
            if feature in row:
                values.append(
                    to_float(
                        row[
                            feature
                        ]
                    )
                )
            else:
                values.append(
                    float(
                        context[
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


X_location = matrix(
    LOCATION_FEATURES
)

X_setup = matrix(
    SETUP_GEOMETRY
)

X_setup_incoming = matrix(
    SETUP_GEOMETRY
    + INCOMING_FEATURES
)

X_contact_type = matrix(
    SETUP_GEOMETRY
    + INCOMING_FEATURES
    + ["is_clearance"]
)

X_outgoing = matrix(
    SETUP_GEOMETRY
    + INCOMING_FEATURES
    + OUTGOING_FEATURES
)

X_landing = matrix(
    SETUP_GEOMETRY
    + INCOMING_FEATURES
    + OUTGOING_FEATURES
    + LANDING_FEATURES
)


y = np.asarray(
    [
        int(
            row["target"]
        )
        for row in rows
    ],
    dtype=int,
)

groups = np.asarray(
    [
        row["match_id"]
        for row in rows
    ]
)


STAGES = [
    (
        "location_only",
        X_location,
    ),
    (
        "setup_geometry",
        X_setup,
    ),
    (
        "setup_plus_incoming",
        X_setup_incoming,
    ),
    (
        "plus_contact_type",
        X_contact_type,
    ),
    (
        "plus_outgoing_touch",
        X_outgoing,
    ),
    (
        "plus_landing_zone",
        X_landing,
    ),
]


# ============================================================
# WHOLE-MATCH OOF
# ============================================================

predictions = {
    name: np.full(
        len(rows),
        np.nan,
    )
    for name, _ in STAGES
}

predictions[
    "prior"
] = np.full(
    len(rows),
    np.nan,
)


cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)


print()
print("=" * 104)
print("WHOLE-MATCH OOF INFORMATION STAGES")
print("=" * 104)

print()


for fold, (
    train_idx,
    val_idx,
) in enumerate(
    cv.split(
        X_setup,
        y,
        groups,
    ),
    start=1,
):

    predictions[
        "prior"
    ][
        val_idx
    ] = y[
        train_idx
    ].mean()

    fold_text = [
        f"Fold {fold}"
    ]

    for name, X in STAGES:

        model = new_model()

        model.fit(
            X[
                train_idx
            ],
            y[
                train_idx
            ],
        )

        p = model.predict_proba(
            X[
                val_idx
            ]
        )[:, 1]

        predictions[
            name
        ][
            val_idx
        ] = p

        fold_text.append(
            (
                f"{name}="
                f"{log_loss(y[val_idx], p, labels=[0,1]):.4f}"
            )
        )

    print(
        " | ".join(
            fold_text
        )
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 104)
print("INFORMATION-STAGE OOF RESULTS")
print("=" * 104)

print()
print(
    f"Positive prevalence: "
    f"{100*y.mean():.1f}%"
)

print()


ordered_names = [
    "prior",
] + [
    name
    for name, _ in STAGES
]


result_rows = []


for name in ordered_names:

    result = metrics(
        y,
        predictions[
            name
        ],
    )

    result_rows.append(
        {
            "stage": name,
            **result,
        }
    )

    print(
        f"{name:<24}"
        f"LL={result['log_loss']:.4f} | "
        f"ROC={result['roc_auc']:.4f} | "
        f"PR={result['pr_auc']:.4f} | "
        f"Brier={result['brier']:.4f}"
    )


print()
print("INCREMENTAL LOG-LOSS CHANGES")
print(
    "(negative = added information improved prediction)"
)

print()


for previous, current in zip(
    ordered_names[:-1],
    ordered_names[1:],
):

    prev_result = next(
        row
        for row in result_rows
        if row["stage"] == previous
    )

    curr_result = next(
        row
        for row in result_rows
        if row["stage"] == current
    )

    delta = (
        curr_result[
            "log_loss"
        ]
        -
        prev_result[
            "log_loss"
        ]
    )

    print(
        f"  {previous:<22} -> "
        f"{current:<22} "
        f"{delta:+.4f}"
    )


# ============================================================
# BY ACTION TYPE
# ============================================================

print()
print("=" * 104)
print("BY FIRST-CONTACT TYPE")
print("=" * 104)


for action in [
    "Pass",
    "Clearance",
]:

    idx = np.asarray(
        [
            i
            for i, row in enumerate(
                rows
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

    for name in ordered_names:

        result = metrics(
            y[
                idx
            ],
            predictions[
                name
            ][
                idx
            ],
        )

        print(
            f"  {name:<22}"
            f"LL={result['log_loss']:.4f} | "
            f"ROC={result['roc_auc']:.4f} | "
            f"PR={result['pr_auc']:.4f}"
        )


# ============================================================
# SAVE
# ============================================================

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
    ] + [
        f"p_{name}"
        for name in ordered_names
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for i, row in enumerate(
        rows
    ):
        out = {
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
        }

        for name in ordered_names:
            out[
                f"p_{name}"
            ] = float(
                predictions[
                    name
                ][i]
            )

        writer.writerow(
            out
        )


print()
print("Saved:")
print(f"  {RESULTS_PATH}")
print(f"  {OOF_PATH}")

print()
print("=" * 104)
print("DONE")
print("=" * 104)
