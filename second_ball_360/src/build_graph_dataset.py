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

MANIFEST_OUTPUT = OUTPUT_DIR / "final_graph_manifest.csv"
ALL_GRAPHS_OUTPUT = OUTPUT_DIR / "final_graph_dataset_all.jsonl"
PRIMARY_GRAPHS_OUTPUT = OUTPUT_DIR / "final_graph_dataset_primary.jsonl"

LOSER_MATCH_TOLERANCE = 0.5
PROXY_MAX_DISTANCE = 5.0

GRAPH_RADIUS = 15.0
PRIMARY_MIN_COVERAGE = 0.75


# ============================================================
# HELPERS
# ============================================================

def distance(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def mirror(point):
    return [
        120.0 - float(point[0]),
        80.0 - float(point[1]),
    ]


def nearest_index(players, location, predicate=None):
    best_idx = None
    best_dist = None

    for i, player in enumerate(players):

        if predicate is not None and not predicate(player):
            continue

        loc = player.get("location")

        if not loc:
            continue

        d = distance(loc, location)

        if best_dist is None or d < best_dist:
            best_idx = i
            best_dist = d

    return best_idx, best_dist


# ============================================================
# LOAD CORE DATASET
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

        record = json.loads(line)

        frame_by_event[
            record["winner_event_id"]
        ] = record


rows_by_match = defaultdict(list)

for row in model_rows:
    rows_by_match[
        row["match_id"]
    ].append(row)


# ============================================================
# ACCUMULATORS
# ============================================================

manifest_rows = []
all_graphs = []
primary_graphs = []

source_counts = Counter()
primary_source_counts = Counter()
exclusion_counts = Counter()
action_counts = Counter()
primary_action_counts = Counter()

proxy_distance_bins = Counter()


# ============================================================
# PROCESS
# ============================================================

print("=" * 88)
print("FINAL GRAPH DATASET BUILD")
print("=" * 88)
print()
print(f"Core labelled contests entering build: {len(model_rows):,}")
print()


for match_num, (match_id, match_rows) in enumerate(
    sorted(
        rows_by_match.items(),
        key=lambda x: int(x[0]),
    ),
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

        winner_event_id = (
            row["winner_event_id"]
        )

        canonical = frame_by_event[
            winner_event_id
        ]

        loser_event_id = (
            canonical["loser_event_id"]
        )

        winner_frame = frame_by_id.get(
            winner_event_id
        )

        if winner_frame is None:
            exclusion_counts[
                "missing_winner_frame"
            ] += 1
            continue

        loser_frame = frame_by_id.get(
            loser_event_id
        )

        winner_players = winner_frame.get(
            "freeze_frame",
            [],
        )

        winner_actors = [
            p
            for p in winner_players
            if p.get("actor") is True
        ]

        if len(winner_actors) != 1:
            exclusion_counts[
                "bad_winner_actor_count"
            ] += 1
            continue

        winner_idx, winner_err = nearest_index(
            winner_players,
            winner_actors[0]["location"],
        )

        if (
            winner_idx is None
            or winner_err is None
            or winner_err > LOSER_MATCH_TOLERANCE
        ):
            exclusion_counts[
                "winner_actor_not_mapped"
            ] += 1
            continue

        # ----------------------------------------------------
        # TRY TO IDENTIFY LOSER EXACTLY
        # ----------------------------------------------------

        loser_idx = None
        loser_source = None
        loser_match_error = None
        proxy_distance = None

        if loser_frame is not None:

            loser_actors = [
                p
                for p in loser_frame.get(
                    "freeze_frame",
                    [],
                )
                if p.get("actor") is True
            ]

            if len(loser_actors) == 1:

                loser_loc = (
                    loser_actors[0]["location"]
                )

                idx_as_is, err_as_is = nearest_index(
                    winner_players,
                    loser_loc,
                )

                idx_mirror, err_mirror = nearest_index(
                    winner_players,
                    mirror(loser_loc),
                )

                candidates = []

                if (
                    idx_as_is is not None
                    and err_as_is is not None
                    and err_as_is <= LOSER_MATCH_TOLERANCE
                ):
                    candidates.append(
                        (
                            err_as_is,
                            idx_as_is,
                            "exact_as_is",
                        )
                    )

                if (
                    idx_mirror is not None
                    and err_mirror is not None
                    and err_mirror <= LOSER_MATCH_TOLERANCE
                ):
                    candidates.append(
                        (
                            err_mirror,
                            idx_mirror,
                            "exact_mirrored",
                        )
                    )

                candidates.sort(
                    key=lambda x: x[0]
                )

                for err, idx, source in candidates:

                    # An exact loser must be a distinct opponent
                    # in the canonical winner-side frame.
                    if idx == winner_idx:
                        continue

                    if (
                        winner_players[idx].get(
                            "teammate"
                        )
                        is not False
                    ):
                        continue

                    loser_idx = idx
                    loser_source = source
                    loser_match_error = err
                    break

        # ----------------------------------------------------
        # FALLBACK: VALIDATED NEAREST-OPPONENT PROXY
        # ----------------------------------------------------

        if loser_idx is None:

            proxy_idx, proxy_distance = nearest_index(
                winner_players,
                winner_players[
                    winner_idx
                ]["location"],
                predicate=lambda p: (
                    p.get("teammate")
                    is False
                ),
            )

            if proxy_idx is None:
                exclusion_counts[
                    "no_visible_opponent"
                ] += 1
                continue

            if (
                proxy_distance
                is None
                or proxy_distance
                > PROXY_MAX_DISTANCE
            ):
                exclusion_counts[
                    "proxy_over_5m"
                ] += 1
                continue

            loser_idx = proxy_idx
            loser_source = "proxy_nearest_opponent"

            if proxy_distance < 0.5:
                proxy_distance_bins[
                    "0-0.5m"
                ] += 1
            elif proxy_distance < 1.0:
                proxy_distance_bins[
                    "0.5-1.0m"
                ] += 1
            elif proxy_distance < 1.5:
                proxy_distance_bins[
                    "1.0-1.5m"
                ] += 1
            elif proxy_distance < 2.0:
                proxy_distance_bins[
                    "1.5-2.0m"
                ] += 1
            elif proxy_distance < 3.0:
                proxy_distance_bins[
                    "2.0-3.0m"
                ] += 1
            else:
                proxy_distance_bins[
                    "3.0-5.0m"
                ] += 1

        # ----------------------------------------------------
        # BUILD 15m LOCAL GRAPH
        # ----------------------------------------------------

        contest_location = (
            canonical["contest_location"]
        )

        nodes = []

        for i, player in enumerate(
            winner_players
        ):

            loc = player.get(
                "location"
            )

            if not loc:
                continue

            d_contest = distance(
                loc,
                contest_location,
            )

            # Local support graph.
            # Keep either contestant even in a pathological case
            # where one lies fractionally outside the radius.
            if (
                d_contest > GRAPH_RADIUS
                and i not in {
                    winner_idx,
                    loser_idx,
                }
            ):
                continue

            if i == winner_idx:
                role = "aerial_winner"

            elif i == loser_idx:
                role = "aerial_loser"

            elif (
                player.get("teammate")
                is True
            ):
                role = "winner_support"

            else:
                role = "loser_support"

            dx = (
                float(loc[0])
                -
                float(contest_location[0])
            )

            dy = (
                float(loc[1])
                -
                float(contest_location[1])
            )

            nodes.append(
                {
                    "x": float(loc[0]),
                    "y": float(loc[1]),
                    "dx": dx,
                    "dy": dy,
                    "distance_to_contest": d_contest,
                    "role": role,
                    "team_relative": (
                        "winner_team"
                        if player.get("teammate") is True
                        else "loser_team"
                    ),
                    "keeper": bool(
                        player.get(
                            "keeper",
                            False,
                        )
                    ),
                    "is_contestant": (
                        i in {
                            winner_idx,
                            loser_idx,
                        }
                    ),
                }
            )

        coverage_15 = float(
            row["coverage_15m"]
        )

        graph = {
            "match_id": int(
                row["match_id"]
            ),
            "competition": row[
                "competition"
            ],
            "season": row[
                "season"
            ],
            "match_date": row[
                "match_date"
            ],
            "home_team": row[
                "home_team"
            ],
            "away_team": row[
                "away_team"
            ],
            "winner_event_id": (
                winner_event_id
            ),
            "loser_event_id": (
                loser_event_id
            ),
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
            "coverage_15m": coverage_15,
            "loser_source": loser_source,
            "loser_proxy_distance": proxy_distance,
            "graph_radius_m": GRAPH_RADIUS,
            "nodes": nodes,
        }

        all_graphs.append(
            graph
        )

        source_counts[
            loser_source
        ] += 1

        action_counts[
            row["winner_event_type"]
        ] += 1

        is_exact = (
            loser_source
            in {
                "exact_as_is",
                "exact_mirrored",
            }
        )

        proxy_le_2m = (
            loser_source
            == "proxy_nearest_opponent"
            and proxy_distance is not None
            and proxy_distance <= 2.0
        )

        high_visibility = (
            coverage_15
            >= PRIMARY_MIN_COVERAGE
        )

        primary_eligible = (
            high_visibility
        )

        if primary_eligible:
            primary_graphs.append(
                graph
            )

            primary_source_counts[
                loser_source
            ] += 1

            primary_action_counts[
                row["winner_event_type"]
            ] += 1

        manifest_rows.append(
            {
                "match_id": row[
                    "match_id"
                ],
                "competition": row[
                    "competition"
                ],
                "season": row[
                    "season"
                ],
                "winner_event_id": (
                    winner_event_id
                ),
                "winner_event_type": row[
                    "winner_event_type"
                ],
                "target": int(
                    row[
                        "target_winner_controls_second_ball"
                    ]
                ),
                "coverage_15m": coverage_15,
                "n_nodes_15m": len(
                    nodes
                ),
                "loser_source": (
                    loser_source
                ),
                "loser_match_error": (
                    loser_match_error
                ),
                "loser_proxy_distance": (
                    proxy_distance
                ),
                "is_exact_loser": (
                    is_exact
                ),
                "proxy_le_2m": (
                    proxy_le_2m
                ),
                "coverage_ge_75": (
                    coverage_15 >= 0.75
                ),
                "coverage_ge_90": (
                    coverage_15 >= 0.90
                ),
                "primary_eligible": (
                    primary_eligible
                ),
            }
        )


# ============================================================
# SAVE OUTPUTS
# ============================================================

with open(
    MANIFEST_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=manifest_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(
        manifest_rows
    )


with open(
    ALL_GRAPHS_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    for graph in all_graphs:
        f.write(
            json.dumps(
                graph,
                ensure_ascii=False,
            )
            + "\n"
        )


with open(
    PRIMARY_GRAPHS_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    for graph in primary_graphs:
        f.write(
            json.dumps(
                graph,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# SUMMARY
# ============================================================

def target_summary(graphs):
    winner = sum(
        g[
            "target_winner_controls_second_ball"
        ]
        for g in graphs
    )

    loser = (
        len(graphs)
        -
        winner
    )

    return winner, loser


all_winner, all_loser = target_summary(
    all_graphs
)

primary_winner, primary_loser = (
    target_summary(
        primary_graphs
    )
)


print()
print("=" * 88)
print("FINAL GRAPH BUILD SUMMARY")
print("=" * 88)

print()
print(
    f"Core labelled contests:                 "
    f"{len(model_rows):,}"
)

print(
    f"Graphs with loser identified:            "
    f"{len(all_graphs):,}"
)

print(
    f"Primary graphs (15m coverage >=75%):     "
    f"{len(primary_graphs):,}"
)

print()

print("Loser identification source — all accepted graphs:")

for key, value in (
    source_counts.most_common()
):
    print(
        f"  {key:<28}"
        f"{value:>7,} "
        f"({100*value/len(all_graphs):6.2f}%)"
    )


print()
print("Loser identification source — primary sample:")

for key, value in (
    primary_source_counts.most_common()
):
    print(
        f"  {key:<28}"
        f"{value:>7,} "
        f"({100*value/len(primary_graphs):6.2f}%)"
    )


print()
print("Excluded before graph construction:")

if exclusion_counts:
    for key, value in (
        exclusion_counts.most_common()
    ):
        print(
            f"  {key:<28}"
            f"{value:>7,}"
        )
else:
    print("  None")


print()
print("Proxy distances:")

for key in [
    "0-0.5m",
    "0.5-1.0m",
    "1.0-1.5m",
    "1.5-2.0m",
    "2.0-3.0m",
    "3.0-5.0m",
]:
    value = proxy_distance_bins[
        key
    ]

    if value:
        print(
            f"  {key:<12}"
            f"{value:>7,}"
        )


print()
print("ALL ACCEPTED GRAPH TARGET BALANCE:")

print(
    f"  Winner team controls: "
    f"{all_winner:,} "
    f"({100*all_winner/len(all_graphs):.2f}%)"
)

print(
    f"  Loser team controls:  "
    f"{all_loser:,} "
    f"({100*all_loser/len(all_graphs):.2f}%)"
)


print()
print("PRIMARY SAMPLE TARGET BALANCE:")

print(
    f"  Winner team controls: "
    f"{primary_winner:,} "
    f"({100*primary_winner/len(primary_graphs):.2f}%)"
)

print(
    f"  Loser team controls:  "
    f"{primary_loser:,} "
    f"({100*primary_loser/len(primary_graphs):.2f}%)"
)


print()
print("Primary sample winner-action mix:")

for key, value in (
    primary_action_counts.most_common()
):
    print(
        f"  {key:<20}"
        f"{value:>7,} "
        f"({100*value/len(primary_graphs):6.2f}%)"
    )


exact_primary = sum(
    value
    for key, value
    in primary_source_counts.items()
    if key.startswith("exact_")
)

proxy_primary = (
    primary_source_counts[
        "proxy_nearest_opponent"
    ]
)

print()
print(
    f"Primary exact-loser subset:              "
    f"{exact_primary:,}"
)

print(
    f"Primary proxy-loser subset:              "
    f"{proxy_primary:,}"
)

print()
print("Saved:")
print(f"  {MANIFEST_OUTPUT}")
print(f"  {ALL_GRAPHS_OUTPUT}")
print(f"  {PRIMARY_GRAPHS_OUTPUT}")

print()
print("=" * 88)
print("DONE")
print("=" * 88)
