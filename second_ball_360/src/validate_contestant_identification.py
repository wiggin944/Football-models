import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THREE_SIXTY_DIR = PROJECT_ROOT / "data" / "three_sixty"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DATASET = OUTPUT_DIR / "second_ball_model_dataset.csv"
CONTEST_FRAMES = OUTPUT_DIR / "all_aerial_contest_frames.jsonl"

OUTPUT_PATH = OUTPUT_DIR / "trusted_loser_heuristic_validation.csv"

MATCH_TOLERANCE = 0.5


# ============================================================
# HELPERS
# ============================================================

def dist(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def mirror(point):
    return [
        120.0 - float(point[0]),
        80.0 - float(point[1]),
    ]


def nearest_index(players, location):
    best_i = None
    best_d = None

    for i, player in enumerate(players):
        loc = player.get("location")
        if not loc:
            continue

        d = dist(loc, location)

        if best_d is None or d < best_d:
            best_i = i
            best_d = d

    return best_i, best_d


# ============================================================
# LOAD
# ============================================================

with open(
    MODEL_DATASET,
    "r",
    encoding="utf-8",
) as f:
    model_rows = list(csv.DictReader(f))


frame_by_event = {}

with open(
    CONTEST_FRAMES,
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        if not line.strip():
            continue

        rec = json.loads(line)
        frame_by_event[
            rec["winner_event_id"]
        ] = rec


rows_by_match = defaultdict(list)

for row in model_rows:
    rows_by_match[
        row["match_id"]
    ].append(row)


# ============================================================
# ACCUMULATORS
# ============================================================

classification = Counter()

trusted_total = 0
trusted_nearest_correct = 0
trusted_true_rank_counts = Counter()

trusted_by_comp = defaultdict(
    lambda: Counter()
)

distance_bins = [
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 3.0),
    (3.0, 5.0),
    (5.0, 999.0),
]

bin_counts = {
    b: Counter()
    for b in distance_bins
}

rows_out = []


# ============================================================
# PROCESS
# ============================================================

print("=" * 88)
print("TRUSTED LOSER + NEAREST-OPPONENT VALIDATION")
print("=" * 88)
print()
print(f"Core contests: {len(model_rows):,}")
print()


for match_num, (match_id, match_rows) in enumerate(
    sorted(rows_by_match.items(), key=lambda x: int(x[0])),
    start=1,
):

    frame_path = (
        THREE_SIXTY_DIR /
        f"{match_id}.json"
    )

    with open(
        frame_path,
        "r",
        encoding="utf-8",
    ) as f:
        frames = json.load(f)

    frame_by_id = {
        frame["event_uuid"]: frame
        for frame in frames
    }

    print(
        f"[{match_num:03d}/{len(rows_by_match):03d}] "
        f"{match_id}: {len(match_rows)} contests"
    )

    for row in match_rows:

        winner_event_id = row[
            "winner_event_id"
        ]

        canonical = frame_by_event[
            winner_event_id
        ]

        loser_event_id = canonical[
            "loser_event_id"
        ]

        winner_frame = frame_by_id.get(
            winner_event_id
        )

        loser_frame = frame_by_id.get(
            loser_event_id
        )

        if loser_frame is None:
            classification[
                "missing_loser_frame"
            ] += 1
            continue

        winner_players = winner_frame.get(
            "freeze_frame",
            [],
        )

        loser_players = loser_frame.get(
            "freeze_frame",
            [],
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
            classification[
                "bad_winner_actor_count"
            ] += 1
            continue

        if len(loser_actors) != 1:
            classification[
                "bad_loser_actor_count"
            ] += 1
            continue

        winner_idx, winner_err = nearest_index(
            winner_players,
            winner_actors[0][
                "location"
            ],
        )

        loser_loc = loser_actors[0][
            "location"
        ]

        loser_idx_as_is, loser_err_as_is = (
            nearest_index(
                winner_players,
                loser_loc,
            )
        )

        loser_idx_mirror, loser_err_mirror = (
            nearest_index(
                winner_players,
                mirror(loser_loc),
            )
        )

        if (
            loser_err_as_is is not None
            and loser_err_as_is
            <= MATCH_TOLERANCE
        ):
            loser_idx = loser_idx_as_is
            loser_err = loser_err_as_is
            orientation = "as_is"

        elif (
            loser_err_mirror is not None
            and loser_err_mirror
            <= MATCH_TOLERANCE
        ):
            loser_idx = loser_idx_mirror
            loser_err = loser_err_mirror
            orientation = "mirrored"

        else:
            loser_idx = None
            loser_err = None
            orientation = "unmapped"

        if loser_idx is None:
            case_type = "unmapped"

        elif loser_idx == winner_idx:
            case_type = "same_node_as_winner"

        elif (
            winner_players[
                loser_idx
            ].get("teammate")
            is True
        ):
            case_type = (
                "distinct_but_winner_team_node"
            )

        else:
            case_type = (
                "trusted_distinct_opponent"
            )

        classification[
            case_type
        ] += 1

        # ------------------------------------------
        # Opponents ranked by distance to winner
        # ------------------------------------------

        winner_location = (
            winner_players[
                winner_idx
            ]["location"]
        )

        opponents = []

        for i, player in enumerate(
            winner_players
        ):
            if (
                player.get("teammate")
                is not False
            ):
                continue

            loc = player.get("location")

            if not loc:
                continue

            opponents.append(
                (
                    i,
                    dist(
                        winner_location,
                        loc,
                    ),
                )
            )

        opponents.sort(
            key=lambda x: x[1]
        )

        nearest_opp_idx = (
            opponents[0][0]
            if opponents
            else None
        )

        nearest_opp_distance = (
            opponents[0][1]
            if opponents
            else None
        )

        true_rank = None
        nearest_correct = None

        if (
            case_type
            ==
            "trusted_distinct_opponent"
        ):

            trusted_total += 1

            comp = row[
                "competition"
            ]

            trusted_by_comp[
                comp
            ]["total"] += 1

            for rank, (
                opp_idx,
                opp_distance,
            ) in enumerate(
                opponents,
                start=1,
            ):
                if opp_idx == loser_idx:
                    true_rank = rank
                    true_distance = (
                        opp_distance
                    )
                    break

            trusted_true_rank_counts[
                true_rank
            ] += 1

            nearest_correct = (
                nearest_opp_idx
                ==
                loser_idx
            )

            if nearest_correct:
                trusted_nearest_correct += 1
                trusted_by_comp[
                    comp
                ]["correct"] += 1

            if (
                nearest_opp_distance
                is not None
            ):
                for lower, upper in distance_bins:
                    if (
                        lower
                        <= nearest_opp_distance
                        < upper
                    ):
                        bin_counts[
                            (lower, upper)
                        ]["total"] += 1

                        if nearest_correct:
                            bin_counts[
                                (lower, upper)
                            ]["correct"] += 1

                        break

        rows_out.append(
            {
                "match_id": match_id,
                "competition": row[
                    "competition"
                ],
                "season": row[
                    "season"
                ],
                "winner_event_id": winner_event_id,
                "loser_event_id": loser_event_id,
                "case_type": case_type,
                "mapping_orientation": orientation,
                "loser_match_error": loser_err,
                "nearest_opponent_distance": nearest_opp_distance,
                "true_loser_opponent_rank": true_rank,
                "nearest_opponent_correct": nearest_correct,
            }
        )


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows_out[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows_out)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 88)
print("TRUSTED-CASE SUMMARY")
print("=" * 88)

print()
print("Loser actor mapping classification:")

order = [
    "trusted_distinct_opponent",
    "same_node_as_winner",
    "distinct_but_winner_team_node",
    "unmapped",
    "missing_loser_frame",
    "bad_winner_actor_count",
    "bad_loser_actor_count",
]

for key in order:
    n = classification[key]

    if n == 0:
        continue

    print(
        f"  {key:<34}"
        f"{n:>7,} "
        f"({100*n/len(model_rows):6.2f}%)"
    )


print()
print(
    f"Trusted distinct-opponent cases:   "
    f"{trusted_total:,}"
)

if trusted_total:

    acc = (
        100
        * trusted_nearest_correct
        / trusted_total
    )

    print(
        f"Nearest-opponent correct:          "
        f"{trusted_nearest_correct:,}"
    )

    print(
        f"Nearest-opponent accuracy:         "
        f"{acc:.2f}%"
    )


print()
print("True loser rank among winner-frame opponents:")

for rank in sorted(
    trusted_true_rank_counts,
    key=lambda x: (
        999 if x is None else x
    ),
):

    n = trusted_true_rank_counts[
        rank
    ]

    print(
        f"  Rank {str(rank):<4}"
        f"{n:>7,} "
        f"({100*n/trusted_total:6.2f}%)"
    )


print()
print("Nearest-opponent accuracy by nearest-opponent distance:")

for lower, upper in distance_bins:

    counts = bin_counts[
        (lower, upper)
    ]

    total = counts[
        "total"
    ]

    correct = counts[
        "correct"
    ]

    if total == 0:
        continue

    label = (
        f"{lower:.1f}-{upper:.1f}m"
        if upper < 999
        else f">={lower:.1f}m"
    )

    print(
        f"  {label:<12}"
        f"n={total:>5,}  "
        f"accuracy="
        f"{100*correct/total:6.2f}%"
    )


print()
print("Nearest-opponent accuracy by competition:")

for comp in sorted(
    trusted_by_comp
):

    total = trusted_by_comp[
        comp
    ]["total"]

    correct = trusted_by_comp[
        comp
    ]["correct"]

    print(
        f"  {comp:<30}"
        f"{correct:>5,}/{total:<5,} "
        f"({100*correct/total:6.2f}%)"
    )


print()
print("Saved:")
print(f"  {OUTPUT_PATH}")

print()
print("=" * 88)
print("DONE")
print("=" * 88)
