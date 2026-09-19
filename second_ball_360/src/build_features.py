import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

GRAPH_PATH = OUTPUT_DIR / "final_graph_dataset_primary.jsonl"
CONTESTS_PATH = OUTPUT_DIR / "all_aerial_contests.csv"

FEATURE_OUTPUT = OUTPUT_DIR / "handcrafted_features.csv"
SPLIT_OUTPUT = OUTPUT_DIR / "match_split_manifest.csv"
EDA_OUTPUT = OUTPUT_DIR / "handcrafted_eda_summary.csv"

RANDOM_SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

RADII = [5.0, 10.0, 15.0]


# ============================================================
# HELPERS
# ============================================================

def euclidean(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def safe_mean(values):
    return (
        mean(values)
        if values
        else None
    )


def centroid(nodes):
    if not nodes:
        return None

    return (
        mean(n["dx"] for n in nodes),
        mean(n["dy"] for n in nodes),
    )


def spread(nodes):
    """
    Mean distance of support players from their team's local centroid.
    """
    c = centroid(nodes)

    if c is None:
        return None

    return mean(
        math.hypot(
            n["dx"] - c[0],
            n["dy"] - c[1],
        )
        for n in nodes
    )


def nearest_distance(nodes):
    if not nodes:
        return None

    return min(
        n["distance_to_contest"]
        for n in nodes
    )


def kth_nearest(nodes, k):
    if len(nodes) < k:
        return None

    distances = sorted(
        n["distance_to_contest"]
        for n in nodes
    )

    return distances[k - 1]


def angular_balance(nodes):
    """
    Resultant length of support-player directions around the contest.
    0 = angularly balanced around the contest.
    1 = all support concentrated in one direction.
    """
    if not nodes:
        return None

    cos_vals = []
    sin_vals = []

    for n in nodes:
        angle = math.atan2(
            n["dy"],
            n["dx"],
        )

        cos_vals.append(
            math.cos(angle)
        )
        sin_vals.append(
            math.sin(angle)
        )

    x = mean(cos_vals)
    y = mean(sin_vals)

    return math.hypot(x, y)


def opposing_centroid_gap(
    winner_nodes,
    loser_nodes,
):
    cw = centroid(
        winner_nodes
    )

    cl = centroid(
        loser_nodes
    )

    if cw is None or cl is None:
        return None

    return math.hypot(
        cw[0] - cl[0],
        cw[1] - cl[1],
    )


def radial_count(
    nodes,
    radius,
):
    return sum(
        n["distance_to_contest"]
        <= radius
        for n in nodes
    )


def parse_float(value):
    if value in {
        None,
        "",
        "None",
    }:
        return None

    try:
        return float(value)
    except Exception:
        return None


# ============================================================
# LOAD GRAPHS
# ============================================================

graphs = []

with open(
    GRAPH_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        if not line.strip():
            continue

        graphs.append(
            json.loads(line)
        )


# ============================================================
# LOAD EVENT CONTEXT
# ============================================================

contest_by_event = {}

with open(
    CONTESTS_PATH,
    "r",
    encoding="utf-8",
) as f:

    for row in csv.DictReader(f):

        contest_by_event[
            row["winner_event_id"]
        ] = row


# ============================================================
# BUILD FIXED WHOLE-MATCH SPLIT
# ============================================================

match_to_graphs = defaultdict(list)

for graph in graphs:
    match_to_graphs[
        str(graph["match_id"])
    ].append(graph)


match_ids = list(
    match_to_graphs.keys()
)

rng = random.Random(
    RANDOM_SEED
)

rng.shuffle(
    match_ids
)

n_matches = len(
    match_ids
)

n_train = round(
    n_matches * TRAIN_FRAC
)

n_val = round(
    n_matches * VAL_FRAC
)

train_matches = set(
    match_ids[:n_train]
)

val_matches = set(
    match_ids[
        n_train:
        n_train + n_val
    ]
)

test_matches = set(
    match_ids[
        n_train + n_val:
    ]
)


def get_split(match_id):

    match_id = str(
        match_id
    )

    if match_id in train_matches:
        return "train"

    if match_id in val_matches:
        return "validation"

    return "test"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

feature_rows = []


for graph in graphs:

    nodes = graph[
        "nodes"
    ]

    winner_support = [
        n
        for n in nodes
        if n["role"]
        == "winner_support"
    ]

    loser_support = [
        n
        for n in nodes
        if n["role"]
        == "loser_support"
    ]

    aerial_winner = [
        n
        for n in nodes
        if n["role"]
        == "aerial_winner"
    ]

    aerial_loser = [
        n
        for n in nodes
        if n["role"]
        == "aerial_loser"
    ]

    if (
        len(aerial_winner) != 1
        or len(aerial_loser) != 1
    ):
        continue

    winner_node = (
        aerial_winner[0]
    )

    loser_node = (
        aerial_loser[0]
    )

    contest_event = (
        contest_by_event.get(
            graph[
                "winner_event_id"
            ],
            {},
        )
    )

    row = {
        "match_id": graph[
            "match_id"
        ],

        "competition": graph[
            "competition"
        ],

        "season": graph[
            "season"
        ],

        "winner_event_id": graph[
            "winner_event_id"
        ],

        "split": get_split(
            graph[
                "match_id"
            ]
        ),

        "target": graph[
            "target_winner_controls_second_ball"
        ],

        "winner_event_type": graph[
            "winner_event_type"
        ],

        "is_clearance": int(
            graph[
                "winner_event_type"
            ]
            == "Clearance"
        ),

        "contest_x": float(
            graph[
                "contest_location"
            ][0]
        ),

        "contest_y": float(
            graph[
                "contest_location"
            ][1]
        ),

        "coverage_15m": float(
            graph[
                "coverage_15m"
            ]
        ),

        "loser_source_proxy": int(
            graph[
                "loser_source"
            ]
            == "proxy_nearest_opponent"
        ),

        "contestant_distance": euclidean(
            (
                winner_node["dx"],
                winner_node["dy"],
            ),
            (
                loser_node["dx"],
                loser_node["dy"],
            ),
        ),

        "nearest_winner_support": nearest_distance(
            winner_support
        ),

        "nearest_loser_support": nearest_distance(
            loser_support
        ),

        "second_winner_support": kth_nearest(
            winner_support,
            2,
        ),

        "second_loser_support": kth_nearest(
            loser_support,
            2,
        ),

        "winner_support_spread": spread(
            winner_support
        ),

        "loser_support_spread": spread(
            loser_support
        ),

        "winner_angular_concentration": angular_balance(
            winner_support
        ),

        "loser_angular_concentration": angular_balance(
            loser_support
        ),

        "support_centroid_gap": opposing_centroid_gap(
            winner_support,
            loser_support,
        ),

        "pass_length": parse_float(
            contest_event.get(
                "pass_length"
            )
        ),

        "pass_is_high": int(
            contest_event.get(
                "pass_height"
            )
            == "High Pass"
        ),

        "pass_is_low": int(
            contest_event.get(
                "pass_height"
            )
            == "Low Pass"
        ),
    }

    winner_cent = centroid(
        winner_support
    )

    loser_cent = centroid(
        loser_support
    )

    row[
        "winner_centroid_dx"
    ] = (
        winner_cent[0]
        if winner_cent
        else None
    )

    row[
        "winner_centroid_dy"
    ] = (
        winner_cent[1]
        if winner_cent
        else None
    )

    row[
        "loser_centroid_dx"
    ] = (
        loser_cent[0]
        if loser_cent
        else None
    )

    row[
        "loser_centroid_dy"
    ] = (
        loser_cent[1]
        if loser_cent
        else None
    )

    for radius in RADII:

        r_label = int(
            radius
        )

        winner_count = (
            radial_count(
                winner_support,
                radius,
            )
        )

        loser_count = (
            radial_count(
                loser_support,
                radius,
            )
        )

        row[
            f"winner_support_{r_label}m"
        ] = winner_count

        row[
            f"loser_support_{r_label}m"
        ] = loser_count

        row[
            f"support_advantage_{r_label}m"
        ] = (
            winner_count
            -
            loser_count
        )

    feature_rows.append(
        row
    )


# ============================================================
# SAVE FEATURES
# ============================================================

with open(
    FEATURE_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=feature_rows[
            0
        ].keys(),
    )

    writer.writeheader()
    writer.writerows(
        feature_rows
    )


# ============================================================
# SAVE MATCH SPLIT MANIFEST
# ============================================================

split_rows = []

for match_id in sorted(
    match_ids,
    key=int,
):

    split = get_split(
        match_id
    )

    match_graphs = (
        match_to_graphs[
            match_id
        ]
    )

    n_positive = sum(
        g[
            "target_winner_controls_second_ball"
        ]
        for g in match_graphs
    )

    split_rows.append(
        {
            "match_id": match_id,
            "split": split,
            "n_graphs": len(
                match_graphs
            ),
            "n_winner_controls": n_positive,
            "n_loser_controls": (
                len(match_graphs)
                -
                n_positive
            ),
        }
    )


with open(
    SPLIT_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=split_rows[
            0
        ].keys(),
    )

    writer.writeheader()
    writer.writerows(
        split_rows
    )


# ============================================================
# SIMPLE EDA: SECOND-BALL WIN RATE BY 15m NUMERICAL ADVANTAGE
# ============================================================

advantage_groups = defaultdict(
    list
)

for row in feature_rows:

    advantage = int(
        row[
            "support_advantage_15m"
        ]
    )

    advantage_groups[
        advantage
    ].append(
        int(
            row["target"]
        )
    )


eda_rows = []

for advantage in sorted(
    advantage_groups
):

    values = (
        advantage_groups[
            advantage
        ]
    )

    eda_rows.append(
        {
            "support_advantage_15m": advantage,
            "n": len(values),
            "winner_control_rate": (
                mean(values)
            ),
        }
    )


with open(
    EDA_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=eda_rows[
            0
        ].keys(),
    )

    writer.writeheader()
    writer.writerows(
        eda_rows
    )


# ============================================================
# SUMMARY
# ============================================================

split_counts = Counter(
    row["split"]
    for row in feature_rows
)

split_targets = defaultdict(
    list
)

for row in feature_rows:
    split_targets[
        row["split"]
    ].append(
        int(
            row["target"]
        )
    )


print()
print("=" * 88)
print("HANDCRAFTED FEATURES + FIXED MATCH SPLIT")
print("=" * 88)

print()
print(
    f"Primary graphs loaded:              "
    f"{len(graphs):,}"
)

print(
    f"Feature rows written:               "
    f"{len(feature_rows):,}"
)

print(
    f"Unique matches:                     "
    f"{len(match_ids):,}"
)

print()
print("Whole-match split:")

for split in [
    "train",
    "validation",
    "test",
]:

    values = (
        split_targets[
            split
        ]
    )

    positive_rate = (
        mean(values)
        if values
        else 0
    )

    n_split_matches = sum(
        1
        for row in split_rows
        if row["split"]
        == split
    )

    print(
        f"  {split:<11}"
        f"{split_counts[split]:>6,} contests | "
        f"{n_split_matches:>3} matches | "
        f"winner-control rate "
        f"{100*positive_rate:5.2f}%"
    )


print()
print(
    "Observed winner-control rate by "
    "15m support numerical advantage:"
)

for row in eda_rows:

    if row["n"] < 25:
        continue

    print(
        f"  advantage "
        f"{row['support_advantage_15m']:>+3}: "
        f"n={row['n']:>4,} | "
        f"winner control "
        f"{100*row['winner_control_rate']:5.1f}%"
    )


missingness = Counter()

feature_names = [
    "nearest_winner_support",
    "nearest_loser_support",
    "second_winner_support",
    "second_loser_support",
    "winner_support_spread",
    "loser_support_spread",
    "winner_angular_concentration",
    "loser_angular_concentration",
    "support_centroid_gap",
    "winner_centroid_dx",
    "winner_centroid_dy",
    "loser_centroid_dx",
    "loser_centroid_dy",
    "pass_length",
]

for name in feature_names:
    missingness[name] = sum(
        row[name] is None
        for row in feature_rows
    )


print()
print("Feature missingness:")

for name, count in missingness.items():

    print(
        f"  {name:<32}"
        f"{count:>6,} "
        f"({100*count/len(feature_rows):5.2f}%)"
    )


print()
print("Saved:")
print(f"  {FEATURE_OUTPUT}")
print(f"  {SPLIT_OUTPUT}")
print(f"  {EDA_OUTPUT}")

print()
print("=" * 88)
print("DONE")
print("=" * 88)
