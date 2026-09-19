import csv
import json
import math
from pathlib import Path
from statistics import mean, median


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CONTESTS_PATH = OUTPUT_DIR / "all_aerial_contests.csv"
FRAMES_PATH = OUTPUT_DIR / "all_aerial_contest_frames.jsonl"
LABELS_PATH = OUTPUT_DIR / "second_ball_label_validation.csv"

MODEL_OUTPUT = OUTPUT_DIR / "second_ball_model_dataset.csv"
VISIBILITY_OUTPUT = OUTPUT_DIR / "visibility_radius_audit.csv"

RADII = [5, 10, 15, 20, 25]

CORE_ACTIONS = {
    "Pass",
    "Clearance",
}

VALID_LABELS = {
    "winner_team",
    "loser_team",
}


# ============================================================
# SHAPELY CHECK
# ============================================================

try:
    from shapely.geometry import Point, Polygon
except ImportError as exc:
    raise ImportError(
        "\nThis script needs shapely.\n"
        "Install it with:\n\n"
        "    pip install shapely\n"
    ) from exc


# ============================================================
# HELPERS
# ============================================================

def euclidean(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def parse_visible_area(raw):
    """
    StatsBomb 360 visible_area is usually a flat coordinate list:
        [x1, y1, x2, y2, ...]

    This also tolerates [[x1,y1], [x2,y2], ...].
    """
    if not raw:
        return None

    if (
        isinstance(raw, list)
        and raw
        and isinstance(raw[0], list)
    ):
        coords = [
            (float(p[0]), float(p[1]))
            for p in raw
            if len(p) >= 2
        ]

    else:
        if len(raw) < 6 or len(raw) % 2 != 0:
            return None

        coords = [
            (
                float(raw[i]),
                float(raw[i + 1]),
            )
            for i in range(
                0,
                len(raw),
                2,
            )
        ]

    if len(coords) < 3:
        return None

    polygon = Polygon(coords)

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    if polygon.is_empty:
        return None

    return polygon


def describe(values):
    if not values:
        return None

    values = sorted(values)
    n = len(values)

    if n % 2:
        med = values[n // 2]
    else:
        med = (
            values[n // 2 - 1]
            +
            values[n // 2]
        ) / 2

    return {
        "min": values[0],
        "mean": mean(values),
        "median": med,
        "max": values[-1],
    }


def percentile(values, q):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    pos = (
        (len(values) - 1)
        * q
    )

    lower = int(
        math.floor(pos)
    )

    upper = int(
        math.ceil(pos)
    )

    if lower == upper:
        return values[lower]

    weight = pos - lower

    return (
        values[lower]
        * (1 - weight)
        +
        values[upper]
        * weight
    )


# ============================================================
# LOAD CONTEST TABLE
# ============================================================

if not CONTESTS_PATH.exists():
    raise FileNotFoundError(
        f"Missing:\n{CONTESTS_PATH}"
    )

if not LABELS_PATH.exists():
    raise FileNotFoundError(
        f"Missing:\n{LABELS_PATH}"
    )

if not FRAMES_PATH.exists():
    raise FileNotFoundError(
        f"Missing:\n{FRAMES_PATH}"
    )


with open(
    CONTESTS_PATH,
    "r",
    encoding="utf-8",
) as f:

    contests = list(
        csv.DictReader(f)
    )


with open(
    LABELS_PATH,
    "r",
    encoding="utf-8",
) as f:

    labels = list(
        csv.DictReader(f)
    )


contest_by_event = {
    row["winner_event_id"]: row
    for row in contests
}


label_by_event = {
    row["winner_event_id"]: row
    for row in labels
}


# ============================================================
# LOAD CANONICAL FRAMES
# ============================================================

frame_by_event = {}

with open(
    FRAMES_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        if not line.strip():
            continue

        record = json.loads(
            line
        )

        frame_by_event[
            record["winner_event_id"]
        ] = record


# ============================================================
# BUILD CLEAN LABELLED CORE SAMPLE
# ============================================================

candidate_ids = []

for event_id, contest in contest_by_event.items():

    if (
        contest.get(
            "has_winner_360",
            "",
        ).lower()
        != "true"
    ):
        continue

    if (
        contest.get(
            "winner_event_type"
        )
        not in CORE_ACTIONS
    ):
        continue

    label_row = label_by_event.get(
        event_id
    )

    if label_row is None:
        continue

    if (
        label_row.get("label_5s")
        not in VALID_LABELS
    ):
        continue

    if event_id not in frame_by_event:
        continue

    candidate_ids.append(
        event_id
    )


print("=" * 84)
print("FINAL CORE SAMPLE + VISIBILITY AUDIT")
print("=" * 84)

print()
print(
    f"Resolved Pass/Clearance contests "
    f"with canonical 360: "
    f"{len(candidate_ids):,}"
)


# ============================================================
# BUILD FEATURES / VISIBILITY METRICS
# ============================================================

rows = []

coverage_by_radius = {
    radius: []
    for radius in RADII
}

full_coverage_counts = {
    radius: 0
    for radius in RADII
}

coverage_90_counts = {
    radius: 0
    for radius in RADII
}

coverage_75_counts = {
    radius: 0
    for radius in RADII
}

winner_counts_by_radius = {
    radius: []
    for radius in RADII
}

opponent_counts_by_radius = {
    radius: []
    for radius in RADII
}

advantage_by_radius = {
    radius: []
    for radius in RADII
}

nearest_winner_distances = []
nearest_opponent_distances = []

invalid_visible_area = 0
missing_contest_location = 0


for event_id in candidate_ids:

    contest = contest_by_event[
        event_id
    ]

    label_row = label_by_event[
        event_id
    ]

    frame_record = frame_by_event[
        event_id
    ]

    contest_location = (
        frame_record.get(
            "contest_location"
        )
    )

    if (
        not contest_location
        or len(contest_location) < 2
    ):
        missing_contest_location += 1
        continue

    contest_xy = (
        float(contest_location[0]),
        float(contest_location[1]),
    )

    freeze_frame = (
        frame_record.get(
            "freeze_frame",
            [],
        )
    )

    visible_polygon = parse_visible_area(
        frame_record.get(
            "visible_area"
        )
    )

    if visible_polygon is None:
        invalid_visible_area += 1

    winner_players = [
        player
        for player in freeze_frame
        if player.get("teammate") is True
    ]

    opponent_players = [
        player
        for player in freeze_frame
        if player.get("teammate") is False
    ]

    winner_distances = [
        euclidean(
            contest_xy,
            (
                float(
                    player["location"][0]
                ),
                float(
                    player["location"][1]
                ),
            ),
        )
        for player in winner_players
        if player.get("location")
    ]

    opponent_distances = [
        euclidean(
            contest_xy,
            (
                float(
                    player["location"][0]
                ),
                float(
                    player["location"][1]
                ),
            ),
        )
        for player in opponent_players
        if player.get("location")
    ]

    # Remove the aerial winner/actor itself from
    # "support" nearest-distance calculations.
    support_winner_distances = []

    for player in winner_players:

        if player.get("actor") is True:
            continue

        location = player.get(
            "location"
        )

        if not location:
            continue

        support_winner_distances.append(
            euclidean(
                contest_xy,
                (
                    float(location[0]),
                    float(location[1]),
                ),
            )
        )

    nearest_winner = (
        min(support_winner_distances)
        if support_winner_distances
        else None
    )

    nearest_opponent = (
        min(opponent_distances)
        if opponent_distances
        else None
    )

    if nearest_winner is not None:
        nearest_winner_distances.append(
            nearest_winner
        )

    if nearest_opponent is not None:
        nearest_opponent_distances.append(
            nearest_opponent
        )

    row = {
        "match_id": contest["match_id"],
        "competition": contest["competition"],
        "season": contest["season"],
        "match_date": contest["match_date"],
        "home_team": contest["home_team"],
        "away_team": contest["away_team"],

        "winner_event_id": event_id,
        "winner_event_type": contest["winner_event_type"],

        "winner_team": contest["winner_team"],
        "loser_team": contest["loser_team"],

        "period": contest["period"],
        "minute": contest["minute"],
        "second": contest["second"],

        "contest_x": contest_xy[0],
        "contest_y": contest_xy[1],

        "target_winner_controls_second_ball": (
            1
            if label_row["label_5s"]
            == "winner_team"
            else 0
        ),

        "control_reason_5s": label_row.get(
            "reason_5s"
        ),

        "control_dt_5s": label_row.get(
            "control_dt_5s"
        ),

        "n_visible": len(
            freeze_frame
        ),

        "n_winner_visible": len(
            winner_players
        ),

        "n_opponent_visible": len(
            opponent_players
        ),

        "nearest_winner_support": nearest_winner,
        "nearest_opponent": nearest_opponent,
    }

    for radius in RADII:

        winner_count = sum(
            d <= radius
            for d in support_winner_distances
        )

        opponent_count = sum(
            d <= radius
            for d in opponent_distances
        )

        local_advantage = (
            winner_count
            -
            opponent_count
        )

        winner_counts_by_radius[
            radius
        ].append(
            winner_count
        )

        opponent_counts_by_radius[
            radius
        ].append(
            opponent_count
        )

        advantage_by_radius[
            radius
        ].append(
            local_advantage
        )

        row[
            f"winner_support_{radius}m"
        ] = winner_count

        row[
            f"opponents_{radius}m"
        ] = opponent_count

        row[
            f"local_advantage_{radius}m"
        ] = local_advantage

        if visible_polygon is not None:

            circle = Point(
                contest_xy
            ).buffer(
                radius,
                resolution=64,
            )

            circle_area = (
                circle.area
            )

            intersection_area = (
                circle
                .intersection(
                    visible_polygon
                )
                .area
            )

            coverage = (
                intersection_area
                /
                circle_area
                if circle_area > 0
                else 0
            )

            coverage = max(
                0.0,
                min(
                    1.0,
                    coverage,
                ),
            )

            coverage_by_radius[
                radius
            ].append(
                coverage
            )

            if coverage >= 0.999:
                full_coverage_counts[
                    radius
                ] += 1

            if coverage >= 0.90:
                coverage_90_counts[
                    radius
                ] += 1

            if coverage >= 0.75:
                coverage_75_counts[
                    radius
                ] += 1

            row[
                f"coverage_{radius}m"
            ] = coverage

        else:

            row[
                f"coverage_{radius}m"
            ] = None

    rows.append(
        row
    )


# ============================================================
# SAVE MODEL TABLE
# ============================================================

if rows:

    with open(
        MODEL_OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# SAVE RADIUS AUDIT TABLE
# ============================================================

radius_rows = []

for radius in RADII:

    values = coverage_by_radius[
        radius
    ]

    summary = describe(
        values
    )

    n = len(values)

    radius_rows.append(
        {
            "radius_m": radius,

            "n_with_valid_polygon": n,

            "coverage_mean": (
                summary["mean"]
                if summary
                else None
            ),

            "coverage_median": (
                summary["median"]
                if summary
                else None
            ),

            "coverage_p10": (
                percentile(
                    values,
                    0.10,
                )
                if values
                else None
            ),

            "coverage_p25": (
                percentile(
                    values,
                    0.25,
                )
                if values
                else None
            ),

            "coverage_p75": (
                percentile(
                    values,
                    0.75,
                )
                if values
                else None
            ),

            "coverage_p90": (
                percentile(
                    values,
                    0.90,
                )
                if values
                else None
            ),

            "pct_full_coverage": (
                100
                * full_coverage_counts[
                    radius
                ]
                / n
                if n
                else 0
            ),

            "pct_ge_90_coverage": (
                100
                * coverage_90_counts[
                    radius
                ]
                / n
                if n
                else 0
            ),

            "pct_ge_75_coverage": (
                100
                * coverage_75_counts[
                    radius
                ]
                / n
                if n
                else 0
            ),

            "mean_winner_support_count": (
                mean(
                    winner_counts_by_radius[
                        radius
                    ]
                )
                if winner_counts_by_radius[
                    radius
                ]
                else None
            ),

            "mean_opponent_count": (
                mean(
                    opponent_counts_by_radius[
                        radius
                    ]
                )
                if opponent_counts_by_radius[
                    radius
                ]
                else None
            ),

            "mean_local_advantage": (
                mean(
                    advantage_by_radius[
                        radius
                    ]
                )
                if advantage_by_radius[
                    radius
                ]
                else None
            ),
        }
    )


with open(
    VISIBILITY_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=radius_rows[0].keys(),
    )

    writer.writeheader()

    writer.writerows(
        radius_rows
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

positive = sum(
    row[
        "target_winner_controls_second_ball"
    ]
    for row in rows
)

negative = (
    len(rows)
    -
    positive
)


print()
print("=" * 84)
print("FINAL CORE SAMPLE SUMMARY")
print("=" * 84)

print()
print(
    f"Rows written:                    "
    f"{len(rows):,}"
)

print(
    f"Winner controls second ball:     "
    f"{positive:,} "
    f"({100 * positive / len(rows):.2f}%)"
)

print(
    f"Loser controls second ball:      "
    f"{negative:,} "
    f"({100 * negative / len(rows):.2f}%)"
)

print()
print(
    f"Invalid visible-area polygons:   "
    f"{invalid_visible_area:,}"
)

print(
    f"Missing contest locations:       "
    f"{missing_contest_location:,}"
)


print()
print("VISIBILITY BY RADIUS")
print()

header = (
    f"{'Radius':>7} | "
    f"{'Mean':>7} | "
    f"{'Median':>7} | "
    f"{'P10':>7} | "
    f"{'>=90%':>7} | "
    f"{'>=75%':>7} | "
    f"{'Win n':>7} | "
    f"{'Opp n':>7}"
)

print(header)
print("-" * len(header))


for row in radius_rows:

    print(
        f"{row['radius_m']:>6}m | "
        f"{100 * row['coverage_mean']:>6.1f}% | "
        f"{100 * row['coverage_median']:>6.1f}% | "
        f"{100 * row['coverage_p10']:>6.1f}% | "
        f"{row['pct_ge_90_coverage']:>6.1f}% | "
        f"{row['pct_ge_75_coverage']:>6.1f}% | "
        f"{row['mean_winner_support_count']:>7.2f} | "
        f"{row['mean_opponent_count']:>7.2f}"
    )


winner_nearest_summary = describe(
    nearest_winner_distances
)

opponent_nearest_summary = describe(
    nearest_opponent_distances
)


if winner_nearest_summary:

    print()
    print(
        "Nearest non-actor winner-team support:"
    )

    print(
        f"  Mean:   "
        f"{winner_nearest_summary['mean']:.2f}"
    )

    print(
        f"  Median: "
        f"{winner_nearest_summary['median']:.2f}"
    )


if opponent_nearest_summary:

    print()
    print(
        "Nearest opponent:"
    )

    print(
        f"  Mean:   "
        f"{opponent_nearest_summary['mean']:.2f}"
    )

    print(
        f"  Median: "
        f"{opponent_nearest_summary['median']:.2f}"
    )


print()
print("Saved:")
print(f"  {MODEL_OUTPUT}")
print(f"  {VISIBILITY_OUTPUT}")

print()
print("=" * 84)
print("DONE")
print("=" * 84)
