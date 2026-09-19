import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

THREE_SIXTY_DIR = PROJECT_ROOT / "data" / "three_sixty"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DATASET = OUTPUT_DIR / "second_ball_model_dataset.csv"
CONTEST_FRAMES = OUTPUT_DIR / "all_aerial_contest_frames.jsonl"

AUDIT_OUTPUT = OUTPUT_DIR / "contestant_role_audit.csv"
GRAPH_OUTPUT = OUTPUT_DIR / "core_contest_graphs.jsonl"

RADII = [5, 10, 15, 20]


# ============================================================
# HELPERS
# ============================================================

def distance(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def nearest_index(players, location):
    if not players:
        return None, None

    best_idx = None
    best_dist = None

    for i, player in enumerate(players):
        loc = player.get("location")

        if not loc:
            continue

        d = distance(loc, location)

        if best_dist is None or d < best_dist:
            best_idx = i
            best_dist = d

    return best_idx, best_dist


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


# ============================================================
# LOAD CORE MODEL TABLE
# ============================================================

if not MODEL_DATASET.exists():
    raise FileNotFoundError(
        f"Missing:\n{MODEL_DATASET}"
    )

with open(
    MODEL_DATASET,
    "r",
    encoding="utf-8",
) as f:
    model_rows = list(csv.DictReader(f))


# ============================================================
# LOAD CANONICAL WINNER FRAMES
# ============================================================

frame_by_winner_event = {}

with open(
    CONTEST_FRAMES,
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)

        frame_by_winner_event[
            record["winner_event_id"]
        ] = record


# ============================================================
# GROUP BY MATCH SO ORIGINAL 360 FILES ARE READ ONCE
# ============================================================

rows_by_match = {}

for row in model_rows:
    rows_by_match.setdefault(
        row["match_id"],
        [],
    ).append(row)


print("=" * 84)
print("AERIAL CONTESTANT ROLE AUDIT")
print("=" * 84)

print()
print(f"Core labelled contests: {len(model_rows):,}")
print()


# ============================================================
# ACCUMULATORS
# ============================================================

audit_rows = []
graph_records = []

missing_loser_frame = 0
bad_winner_actor_count = 0
bad_loser_actor_count = 0
bad_loser_match = 0

winner_actor_errors = []
loser_actor_match_errors = []
contestant_separations = []

nearest_winner_support = []
nearest_loser_support = []

support_counts = {
    radius: {
        "winner": [],
        "loser": [],
    }
    for radius in RADII
}


# ============================================================
# PROCESS
# ============================================================

for match_num, (match_id, match_rows) in enumerate(
    sorted(
        rows_by_match.items(),
        key=lambda x: int(x[0]),
    ),
    start=1,
):

    path = (
        THREE_SIXTY_DIR /
        f"{match_id}.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        raw_frames = json.load(f)

    frame_by_id = {
        frame["event_uuid"]: frame
        for frame in raw_frames
    }

    print(
        f"[{match_num:03d}/"
        f"{len(rows_by_match):03d}] "
        f"{match_id}: "
        f"{len(match_rows)} contests"
    )

    for row in match_rows:

        winner_event_id = (
            row["winner_event_id"]
        )

        canonical = (
            frame_by_winner_event[
                winner_event_id
            ]
        )

        loser_event_id = (
            canonical["loser_event_id"]
        )

        winner_frame = (
            frame_by_id.get(
                winner_event_id
            )
        )

        loser_frame = (
            frame_by_id.get(
                loser_event_id
            )
        )

        if loser_frame is None:
            missing_loser_frame += 1
            continue

        winner_players = (
            winner_frame.get(
                "freeze_frame",
                [],
            )
        )

        loser_players = (
            loser_frame.get(
                "freeze_frame",
                [],
            )
        )

        winner_actors = [
            p
            for p in winner_players
            if p.get("actor") is True
        ]

        loser_actors = [
            p
            for p in loser_players
            if p.get("actor") is True
        ]

        if len(winner_actors) != 1:
            bad_winner_actor_count += 1
            continue

        if len(loser_actors) != 1:
            bad_loser_actor_count += 1
            continue

        winner_actor = winner_actors[0]
        loser_actor = loser_actors[0]

        # Winner actor should already be one exact node
        # in the canonical winner frame.
        winner_idx, winner_err = nearest_index(
            winner_players,
            winner_actor["location"],
        )

        winner_actor_errors.append(
            winner_err
        )

        # Find the loser actor's physical player inside
        # the canonical winner-side point cloud.
        loser_idx, loser_err = nearest_index(
            winner_players,
            loser_actor["location"],
        )

        loser_actor_match_errors.append(
            loser_err
        )

        if loser_idx is None or loser_err > 0.5:
            bad_loser_match += 1
            continue

        if winner_idx == loser_idx:
            bad_loser_match += 1
            continue

        winner_location = (
            winner_players[winner_idx][
                "location"
            ]
        )

        loser_location = (
            winner_players[loser_idx][
                "location"
            ]
        )

        contestant_separation = distance(
            winner_location,
            loser_location,
        )

        contestant_separations.append(
            contestant_separation
        )

        contest_location = (
            canonical[
                "contest_location"
            ]
        )

        # ----------------------------------------------------
        # TAG NODES
        # ----------------------------------------------------

        nodes = []

        winner_support_distances = []
        loser_support_distances = []

        for i, player in enumerate(
            winner_players
        ):

            loc = player.get(
                "location"
            )

            if not loc:
                continue

            if i == winner_idx:
                role = "aerial_winner"

            elif i == loser_idx:
                role = "aerial_loser"

            elif player.get(
                "teammate"
            ) is True:
                role = "winner_support"

            else:
                role = "loser_support"

            d_contest = distance(
                loc,
                contest_location,
            )

            if role == "winner_support":
                winner_support_distances.append(
                    d_contest
                )

            elif role == "loser_support":
                loser_support_distances.append(
                    d_contest
                )

            nodes.append(
                {
                    "x": float(loc[0]),
                    "y": float(loc[1]),
                    "dx": (
                        float(loc[0])
                        -
                        float(contest_location[0])
                    ),
                    "dy": (
                        float(loc[1])
                        -
                        float(contest_location[1])
                    ),
                    "distance_to_contest": d_contest,
                    "role": role,
                    "teammate_to_winner": (
                        player.get(
                            "teammate"
                        )
                        is True
                    ),
                    "keeper": (
                        player.get(
                            "keeper",
                            False,
                        )
                        is True
                    ),
                }
            )

        nearest_win = (
            min(winner_support_distances)
            if winner_support_distances
            else None
        )

        nearest_lose = (
            min(loser_support_distances)
            if loser_support_distances
            else None
        )

        if nearest_win is not None:
            nearest_winner_support.append(
                nearest_win
            )

        if nearest_lose is not None:
            nearest_loser_support.append(
                nearest_lose
            )

        audit = {
            "match_id": match_id,
            "winner_event_id": winner_event_id,
            "loser_event_id": loser_event_id,
            "winner_event_type": row[
                "winner_event_type"
            ],
            "target": int(
                row[
                    "target_winner_controls_second_ball"
                ]
            ),
            "n_visible": len(
                winner_players
            ),
            "winner_actor_match_error": (
                winner_err
            ),
            "loser_actor_match_error": (
                loser_err
            ),
            "contestant_separation": (
                contestant_separation
            ),
            "nearest_winner_support": (
                nearest_win
            ),
            "nearest_loser_support": (
                nearest_lose
            ),
        }

        for radius in RADII:

            winner_count = sum(
                d <= radius
                for d in winner_support_distances
            )

            loser_count = sum(
                d <= radius
                for d in loser_support_distances
            )

            support_counts[
                radius
            ]["winner"].append(
                winner_count
            )

            support_counts[
                radius
            ]["loser"].append(
                loser_count
            )

            audit[
                f"winner_support_{radius}m"
            ] = winner_count

            audit[
                f"loser_support_{radius}m"
            ] = loser_count

            audit[
                f"support_advantage_{radius}m"
            ] = (
                winner_count
                -
                loser_count
            )

        audit_rows.append(
            audit
        )

        graph_records.append(
            {
                "match_id": int(
                    match_id
                ),
                "winner_event_id": (
                    winner_event_id
                ),
                "loser_event_id": (
                    loser_event_id
                ),
                "competition": row[
                    "competition"
                ],
                "season": row[
                    "season"
                ],
                "winner_event_type": row[
                    "winner_event_type"
                ],
                "target_winner_controls_second_ball": int(
                    row[
                        "target_winner_controls_second_ball"
                    ]
                ),
                "contest_location": [
                    float(
                        contest_location[0]
                    ),
                    float(
                        contest_location[1]
                    ),
                ],
                "coverage_5m": float(
                    row["coverage_5m"]
                ),
                "coverage_10m": float(
                    row["coverage_10m"]
                ),
                "coverage_15m": float(
                    row["coverage_15m"]
                ),
                "coverage_20m": float(
                    row["coverage_20m"]
                ),
                "nodes": nodes,
            }
        )


# ============================================================
# SAVE
# ============================================================

with open(
    AUDIT_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=audit_rows[0].keys(),
    )

    writer.writeheader()

    writer.writerows(
        audit_rows
    )


with open(
    GRAPH_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    for record in graph_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 84)
print("CONTESTANT ROLE SUMMARY")
print("=" * 84)

print()

print(
    f"Core contests attempted:           "
    f"{len(model_rows):,}"
)

print(
    f"Graphs successfully role-tagged:   "
    f"{len(graph_records):,}"
)

print(
    f"Missing loser-side 360 frame:      "
    f"{missing_loser_frame:,}"
)

print(
    f"Bad winner actor count:            "
    f"{bad_winner_actor_count:,}"
)

print(
    f"Bad loser actor count:             "
    f"{bad_loser_actor_count:,}"
)

print(
    f"Could not map loser actor:         "
    f"{bad_loser_match:,}"
)


winner_err_summary = describe(
    winner_actor_errors
)

loser_err_summary = describe(
    loser_actor_match_errors
)

separation_summary = describe(
    contestant_separations
)

winner_support_summary = describe(
    nearest_winner_support
)

loser_support_summary = describe(
    nearest_loser_support
)


if winner_err_summary:

    print()
    print(
        "Winner actor -> canonical node error:"
    )

    print(
        f"  Mean:   "
        f"{winner_err_summary['mean']:.4f}"
    )

    print(
        f"  Median: "
        f"{winner_err_summary['median']:.4f}"
    )

    print(
        f"  Max:    "
        f"{winner_err_summary['max']:.4f}"
    )


if loser_err_summary:

    print()
    print(
        "Loser actor -> canonical node error:"
    )

    print(
        f"  Mean:   "
        f"{loser_err_summary['mean']:.4f}"
    )

    print(
        f"  Median: "
        f"{loser_err_summary['median']:.4f}"
    )

    print(
        f"  Max:    "
        f"{loser_err_summary['max']:.4f}"
    )


if separation_summary:

    print()
    print(
        "Distance between aerial contestants:"
    )

    print(
        f"  Mean:   "
        f"{separation_summary['mean']:.2f}"
    )

    print(
        f"  Median: "
        f"{separation_summary['median']:.2f}"
    )

    print(
        f"  Max:    "
        f"{separation_summary['max']:.2f}"
    )


if winner_support_summary:

    print()
    print(
        "Nearest WINNER-team support "
        "(both contestants excluded):"
    )

    print(
        f"  Mean:   "
        f"{winner_support_summary['mean']:.2f}"
    )

    print(
        f"  Median: "
        f"{winner_support_summary['median']:.2f}"
    )


if loser_support_summary:

    print()
    print(
        "Nearest LOSER-team support "
        "(both contestants excluded):"
    )

    print(
        f"  Mean:   "
        f"{loser_support_summary['mean']:.2f}"
    )

    print(
        f"  Median: "
        f"{loser_support_summary['median']:.2f}"
    )


print()
print("Support counts after removing BOTH aerial contestants:")

print()

header = (
    f"{'Radius':>7} | "
    f"{'Winner support':>14} | "
    f"{'Loser support':>13} | "
    f"{'Difference':>10}"
)

print(header)
print("-" * len(header))

for radius in RADII:

    w = support_counts[
        radius
    ]["winner"]

    l = support_counts[
        radius
    ]["loser"]

    w_mean = mean(w)
    l_mean = mean(l)

    print(
        f"{radius:>6}m | "
        f"{w_mean:>14.2f} | "
        f"{l_mean:>13.2f} | "
        f"{w_mean - l_mean:>10.2f}"
    )


print()
print("Saved:")
print(f"  {AUDIT_OUTPUT}")
print(f"  {GRAPH_OUTPUT}")

print()
print("=" * 84)
print("DONE")
print("=" * 84)
