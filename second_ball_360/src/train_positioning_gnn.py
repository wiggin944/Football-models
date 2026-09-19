import ast
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import orjson

    def fast_load_json(path):
        with open(path, "rb") as f:
            return orjson.loads(f.read())

    JSON_BACKEND = "orjson"

except Exception:
    def fast_load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    JSON_BACKEND = "stdlib json"
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "outputs"

GRAPH_PATH = OUT / "final_graph_dataset_primary.jsonl"
STRICT_PATH = OUT / "true_second_ball_strict_sample.csv"
FEATURE_PATH = OUT / "handcrafted_features.csv"

OOF_OUT = OUT / "positioning_gnn_exact_precontact_oof_predictions.csv"
RESULTS_OUT = OUT / "positioning_gnn_exact_precontact_results.csv"
BOOT_OUT = OUT / "positioning_gnn_exact_precontact_bootstrap.csv"
FEATURE_AUDIT_OUT = OUT / "positioning_gnn_exact_precontact_feature_audit.csv"
CONTEXT_CACHE = OUT / "exact_precontact_context_cache.csv"

SEED = 42
N_SPLITS = 5
EPOCHS = 18
HIDDEN = 48
LAYERS = 2
DROPOUT = 0.30
LR = 0.0007
WEIGHT_DECAY = 0.001
BATCH_SIZE = 96
N_BOOT = 5000

SAME_TEAM_K = 3
OPPONENT_K = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type == "cpu":
    torch.set_num_threads(
        max(1, min(8, torch.get_num_threads()))
    )

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# HELPERS
# ============================================================

def safe_auc(y, p):
    y = np.asarray(y, dtype=int)

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(y, p)


def safe_pr(y, p):
    y = np.asarray(y, dtype=int)

    if len(np.unique(y)) < 2:
        return np.nan

    return average_precision_score(y, p)


def metrics(y, p):
    y = np.asarray(y, dtype=int)

    p = np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )

    return {
        "n": len(y),
        "positive_rate": float(y.mean()),
        "log_loss": log_loss(
            y,
            p,
            labels=[0, 1],
        ),
        "roc_auc": safe_auc(y, p),
        "pr_auc": safe_pr(y, p),
        "brier": brier_score_loss(y, p),
    }


def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c

    return None


# ============================================================
# EXACT PRE-CONTACT FEATURE PARITY
# ============================================================
#
# IMPORTANT:
# Rather than manually reimplement script 31's incoming-ball
# feature engineering, this script parses the user's actual local
# src/train_information_stage_models.py and executes ONLY its
# top-level function definitions.
#
# We then call its exact incoming_features(row, frame) function.
#
# This gives us:
#   raw relational graph representation of setup geometry
#   + exact script-31 location/incoming context
#
# Comparator:
#   script-31 p_setup_plus_incoming XGB
#
# No OUTGOING_FEATURES or LANDING_FEATURES are supplied.
# ============================================================

SCRIPT31_CANDIDATES = list(
    SRC.glob("31*.py")
)

if not SCRIPT31_CANDIDATES:
    raise FileNotFoundError(
        "Could not find src/31*.py."
    )

SCRIPT31_CANDIDATES.sort(
    key=lambda p: (
        0 if "information" in p.name.lower() else 1,
        0 if "decomposition" in p.name.lower() else 1,
        p.name,
    )
)

SCRIPT31 = SCRIPT31_CANDIDATES[0]

LOCATION_FEATURES = [
    "contest_x",
    "contest_y_from_centre",
]

INCOMING_FEATURES_EXACT = [
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

FORBIDDEN_POST_CONTACT = [
    "is_clearance",
    "pass_length",
    "pass_is_high",
    "pass_is_low",
    "landing_dx",
    "landing_dy_abs",
    "landing_visible",
]

CONTEXT_COLS = (
    LOCATION_FEATURES
    + INCOMING_FEATURES_EXACT
)


# ============================================================
# LOAD STRICT SAMPLE + BASE FEATURE TABLE
# ============================================================

strict = pd.read_csv(
    STRICT_PATH
)

strict["winner_event_id"] = (
    strict["winner_event_id"]
    .astype(str)
)

strict_ids = set(
    strict["winner_event_id"]
)

features = pd.read_csv(
    FEATURE_PATH
)

features["winner_event_id"] = (
    features["winner_event_id"]
    .astype(str)
)

rows = (
    features[
        features["winner_event_id"].isin(
            strict_ids
        )
    ]
    .copy()
    .reset_index(drop=True)
)

if len(rows) != len(strict_ids):
    raise RuntimeError(
        f"Expected {len(strict_ids)} strict rows; "
        f"reconstructed {len(rows)}."
    )


target_col = first_existing(
    rows.columns,
    [
        "target",
        "label",
        "y",
    ],
)

match_col = first_existing(
    rows.columns,
    [
        "match_id",
        "match",
    ],
)

if target_col is None:
    raise KeyError(
        "Could not find target column."
    )

if match_col is None:
    raise KeyError(
        "Could not find match_id column."
    )


event_ids = (
    rows["winner_event_id"]
    .astype(str)
    .to_numpy()
)

y = (
    rows[target_col]
    .astype(int)
    .to_numpy()
)

groups = (
    rows[match_col]
    .astype(str)
    .to_numpy()
)


# Script 31 explicitly creates this from contest_y.
if "contest_y" not in rows.columns:
    raise KeyError(
        "handcrafted_features.csv is missing contest_y."
    )

rows[
    "contest_y_from_centre"
] = np.abs(
    pd.to_numeric(
        rows["contest_y"],
        errors="raise",
    ).to_numpy(
        dtype=float
    )
    - 40.0
)


# ============================================================
# EXTRACT EXACT FUNCTION DEFINITIONS FROM SCRIPT 31
# ============================================================

script31_source = SCRIPT31.read_text(
    encoding="utf-8"
)

script31_tree = ast.parse(
    script31_source
)

function_nodes = [
    node
    for node in script31_tree.body
    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    )
]

function_module = ast.Module(
    body=function_nodes,
    type_ignores=[],
)

ast.fix_missing_locations(
    function_module
)

# Globals needed by the extracted functions.
exact_ns = {
    "np": np,
    "math": math,
    "INCOMING_FEATURES": list(
        INCOMING_FEATURES_EXACT
    ),
}

exec(
    compile(
        function_module,
        filename=str(
            SCRIPT31
        ),
        mode="exec",
    ),
    exact_ns,
)

if "incoming_features" not in exact_ns:
    raise RuntimeError(
        "Could not extract incoming_features() "
        "from script 31."
    )

if "find_incoming" not in exact_ns:
    raise RuntimeError(
        "Could not extract find_incoming() "
        "from script 31."
    )


# ============================================================
# EXACT CONTEXT CACHE
# ============================================================

EVENT_DIR = ROOT / "data" / "events"
THREE_SIXTY_DIR = ROOT / "data" / "three_sixty"

match_ids = sorted(
    set(
        rows[
            match_col
        ]
        .astype(str)
        .tolist()
    )
)


def resolve_match_file(
    folder,
    match_id,
):
    candidates = [
        folder / f"{match_id}.json",
    ]

    try:
        candidates.append(
            folder / f"{int(float(match_id))}.json"
        )
    except Exception:
        pass

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No JSON found for match {match_id} "
        f"in {folder}"
    )


print("=" * 108)
print(
    "POSITIONING GNN â€” TRUE PRE-CONTACT PARITY "
    "WITH EXACT SCRIPT-31 FEATURE BUILDER"
)
print("=" * 108)

print()
print(
    f"Script 31 source: {SCRIPT31.name}"
)
print(
    f"Strict episodes:  {len(rows):,}"
)
print(
    f"Matches:          {len(match_ids):,}"
)
print(
    f"Positive rate:    {100*y.mean():.1f}%"
)
print(
    f"Device:           {DEVICE}"
)
print(
    f"JSON backend:     {JSON_BACKEND}"
)


cache_valid = False
incoming_df = None

if CONTEXT_CACHE.exists():
    try:
        cached = pd.read_csv(
            CONTEXT_CACHE
        )

        cached[
            "winner_event_id"
        ] = (
            cached[
                "winner_event_id"
            ]
            .astype(str)
        )

        required_cache = (
            [
                "winner_event_id",
            ]
            + INCOMING_FEATURES_EXACT
        )

        if (
            len(cached)
            == len(rows)
            and all(
                c in cached.columns
                for c in required_cache
            )
            and set(
                cached[
                    "winner_event_id"
                ]
            )
            == set(
                event_ids
            )
        ):
            cache_map = (
                cached
                .set_index(
                    "winner_event_id"
                )
                .loc[
                    list(
                        event_ids
                    )
                ]
                .reset_index()
            )

            incoming_df = (
                cache_map[
                    INCOMING_FEATURES_EXACT
                ]
                .reset_index(
                    drop=True
                )
            )

            cache_valid = True

            print()
            print(
                f"Using cached exact incoming context:"
            )
            print(
                f"  {CONTEXT_CACHE}"
            )
            print(
                "  Raw event/360 loading skipped."
            )

    except Exception as exc:
        print()
        print(
            f"Existing context cache could not be used: "
            f"{exc}"
        )


if not cache_valid:
    print()
    print(
        "No valid exact-context cache found."
    )
    print(
        "Building it once from raw events + 360..."
    )

    incoming_records = []

    grouped_rows = list(
        rows.groupby(
            match_col,
            sort=False,
        )
    )

    for group_number, (
        match_value,
        match_rows,
    ) in enumerate(
        grouped_rows,
        start=1,
    ):
        match_id = str(
            match_value
        )

        events_path = resolve_match_file(
            EVENT_DIR,
            match_id,
        )

        frames_path = resolve_match_file(
            THREE_SIXTY_DIR,
            match_id,
        )

        events = fast_load_json(
            events_path
        )

        frames = fast_load_json(
            frames_path
        )

        # Bind only the current match into the exact script-31
        # function globals. This avoids retaining ~300 matches
        # of large JSON objects in memory.
        events_by_match = {
            match_id: events,
        }

        event_lookup = {
            match_id: {
                str(
                    event["id"]
                ): event
                for event in events
            },
        }

        event_index = {
            match_id: {
                str(
                    event["id"]
                ): idx
                for idx, event in enumerate(
                    events
                )
            },
        }

        frames_by_match = {
            match_id: {
                str(
                    frame[
                        "event_uuid"
                    ]
                ): frame
                for frame in frames
            },
        }

        exact_ns[
            "events_by_match"
        ] = events_by_match

        exact_ns[
            "event_lookup"
        ] = event_lookup

        exact_ns[
            "event_index"
        ] = event_index

        exact_ns[
            "frames_by_match"
        ] = frames_by_match

        incoming_features_exact = (
            exact_ns[
                "incoming_features"
            ]
        )

        for row_idx, row in match_rows.iterrows():
            row_dict = row.to_dict()

            event_id = str(
                row_dict[
                    "winner_event_id"
                ]
            )

            row_dict[
                "match_id"
            ] = match_id

            row_dict[
                "winner_event_id"
            ] = event_id

            frame = frames_by_match[
                match_id
            ].get(
                event_id
            )

            if frame is None:
                raise RuntimeError(
                    f"Missing 360 frame for "
                    f"{event_id}"
                )

            incoming = incoming_features_exact(
                row_dict,
                frame,
            )

            incoming[
                "winner_event_id"
            ] = event_id

            incoming_records.append(
                incoming
            )

        if (
            group_number % 25 == 0
            or group_number
            == len(
                grouped_rows
            )
        ):
            done_rows = len(
                incoming_records
            )

            print(
                f"  processed matches "
                f"{group_number}/"
                f"{len(grouped_rows)} | "
                f"episodes="
                f"{done_rows:,}/"
                f"{len(rows):,}"
            )

        # Explicitly release large raw objects before next match.
        del events
        del frames
        del events_by_match
        del event_lookup
        del event_index
        del frames_by_match

    cache_df = pd.DataFrame(
        incoming_records
    )

    missing_exact = [
        c
        for c in INCOMING_FEATURES_EXACT
        if c not in cache_df.columns
    ]

    if missing_exact:
        raise RuntimeError(
            "Exact incoming builder failed to return: "
            + ", ".join(
                missing_exact
            )
        )

    cache_df.to_csv(
        CONTEXT_CACHE,
        index=False,
    )

    cache_df[
        "winner_event_id"
    ] = (
        cache_df[
            "winner_event_id"
        ]
        .astype(str)
    )

    cache_df = (
        cache_df
        .set_index(
            "winner_event_id"
        )
        .loc[
            list(
                event_ids
            )
        ]
        .reset_index()
    )

    incoming_df = (
        cache_df[
            INCOMING_FEATURES_EXACT
        ]
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "Exact incoming context cached:"
    )
    print(
        f"  {CONTEXT_CACHE}"
    )


for c in INCOMING_FEATURES_EXACT:
    rows[
        c
    ] = pd.to_numeric(
        incoming_df[
            c
        ],
        errors="coerce",
    )


# ============================================================
# LEAKAGE GUARD + FEATURE AUDIT
# ============================================================

used_forbidden = [
    c
    for c in CONTEXT_COLS
    if c in FORBIDDEN_POST_CONTACT
]

if used_forbidden:
    raise RuntimeError(
        "Post-contact leakage detected: "
        + ", ".join(
            used_forbidden
        )
    )


print()
print(
    "Exact GNN context:"
)

for c in LOCATION_FEATURES:
    print(
        f"  [location] {c}"
    )

for c in INCOMING_FEATURES_EXACT:
    print(
        f"  [incoming] {c}"
    )

print()
print(
    "Explicitly excluded:"
)

for c in FORBIDDEN_POST_CONTACT:
    print(
        f"  [excluded] {c}"
    )


# Useful diagnostics.
incoming_found = pd.to_numeric(
    rows[
        "incoming_found"
    ],
    errors="coerce",
)

print()
print(
    f"Incoming pass found: "
    f"{int((incoming_found == 1).sum()):,}/"
    f"{len(rows):,} "
    f"({100*(incoming_found == 1).mean():.1f}%)"
)


audit_rows = []

for c in LOCATION_FEATURES:
    audit_rows.append(
        {
            "feature": c,
            "stage": "location",
            "included": True,
            "source":
            "exact script-31 definition",
        }
    )

for c in INCOMING_FEATURES_EXACT:
    audit_rows.append(
        {
            "feature": c,
            "stage": "incoming",
            "included": True,
            "source":
            "script-31 incoming_features()",
        }
    )

for c in FORBIDDEN_POST_CONTACT:
    audit_rows.append(
        {
            "feature": c,
            "stage": "post_contact",
            "included": False,
            "source":
            "explicit leakage guard",
        }
    )

pd.DataFrame(
    audit_rows
).to_csv(
    FEATURE_AUDIT_OUT,
    index=False,
)


# ============================================================
# CONTEXT MATRIX
# ============================================================

raw_context = np.column_stack(
    [
        pd.to_numeric(
            rows[c],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )
        for c in CONTEXT_COLS
    ]
)

# ============================================================
# FIND EXACT SETUP+INCOMING XGB OOF COMPARATOR
# ============================================================

def discover_setup_incoming_oof():
    candidates_local = []

    for path in OUT.glob(
        "*.csv"
    ):
        try:
            head = pd.read_csv(
                path,
                nrows=5,
            )
        except Exception:
            continue

        if (
            "winner_event_id"
            not in head.columns
        ):
            continue

        pred_cols = []

        for c in head.columns:
            lc = c.lower()

            if (
                "setup" in lc
                and "incoming" in lc
                and (
                    lc.startswith("p_")
                    or "prob" in lc
                    or "pred" in lc
                )
            ):
                pred_cols.append(
                    c
                )

        if not pred_cols:
            continue

        score = 0
        name = path.name.lower()

        if "information" in name:
            score += 4

        if "stage" in name:
            score += 3

        if "oof" in name:
            score += 3

        if "31" in name:
            score += 2

        candidates_local.append(
            (
                score,
                path,
                pred_cols,
            )
        )

    if not candidates_local:
        return None

    candidates_local.sort(
        key=lambda x: (
            -x[0],
            x[1].name,
        )
    )

    _, path, pred_cols = (
        candidates_local[0]
    )

    col = sorted(
        pred_cols,
        key=lambda c: (
            -(
                "setup_plus_incoming"
                in c.lower()
            ),
            c,
        ),
    )[0]

    comp = pd.read_csv(
        path,
        usecols=[
            "winner_event_id",
            col,
        ],
    )

    comp["winner_event_id"] = (
        comp["winner_event_id"]
        .astype(str)
    )

    mapping = dict(
        zip(
            comp[
                "winner_event_id"
            ],
            comp[col],
        )
    )

    if not all(
        eid in mapping
        for eid in event_ids
    ):
        return None

    p = np.asarray(
        [
            float(
                mapping[eid]
            )
            for eid in event_ids
        ],
        dtype=float,
    )

    return (
        path,
        col,
        p,
    )


comparator = discover_setup_incoming_oof()

if comparator is None:
    print()
    print(
        "WARNING: row-level setup+incoming OOF comparator "
        "not found; aggregate stage-31 reference will be shown."
    )
else:
    comp_path, comp_col, _ = comparator

    print()
    print(
        f"Exact comparator: "
        f"{comp_path.name} -> {comp_col}"
    )


# ============================================================
# LOAD GRAPHS
# ============================================================

graphs_by_id = {}

with open(
    GRAPH_PATH,
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        if not line.strip():
            continue

        graph = json.loads(
            line
        )

        eid = str(
            graph[
                "winner_event_id"
            ]
        )

        if eid in strict_ids:
            graphs_by_id[
                eid
            ] = graph


missing = (
    set(event_ids)
    - set(graphs_by_id)
)

if missing:
    raise RuntimeError(
        f"Missing {len(missing)} strict graphs."
    )


# ============================================================
# FOOTBALL GRAPH
# ============================================================

WINNER_ROLES = {
    "aerial_winner",
    "winner_support",
}

LOSER_ROLES = {
    "aerial_loser",
    "loser_support",
}

ROLE_ORDER = [
    "aerial_winner",
    "aerial_loser",
    "winner_support",
    "loser_support",
]

ROLE_TO_INDEX = {
    role: i
    for i, role in enumerate(
        ROLE_ORDER
    )
}


def role_of(node):
    return str(
        node.get(
            "role",
            "",
        )
    )


def side_of(role):
    if role in WINNER_ROLES:
        return 1

    if role in LOSER_ROLES:
        return -1

    return 0


def keeper_flag(node):
    for key in [
        "keeper",
        "is_keeper",
        "goalkeeper",
    ]:
        if key in node:
            return (
                1.0
                if bool(
                    node[key]
                )
                else 0.0
            )

    return 0.0


NODE_DIM = 12


def node_feature(node):
    dx = float(
        node.get(
            "dx",
            0.0,
        )
    )

    dy = float(
        node.get(
            "dy",
            0.0,
        )
    )

    distance = node.get(
        "distance_to_contest",
        node.get(
            "distance",
            None,
        ),
    )

    if distance is None:
        distance = math.hypot(
            dx,
            dy,
        )

    distance = float(
        distance
    )

    angle = math.atan2(
        dy,
        dx,
    )

    role = role_of(
        node
    )

    role_oh = [
        0.0
    ] * 4

    if role in ROLE_TO_INDEX:
        role_oh[
            ROLE_TO_INDEX[
                role
            ]
        ] = 1.0

    return [
        dx / 15.0,
        dy / 15.0,
        distance / 15.0,
        math.sin(angle),
        math.cos(angle),
        float(
            side_of(
                role
            )
        ),
        1.0
        if role in {
            "aerial_winner",
            "aerial_loser",
        }
        else 0.0,
        keeper_flag(
            node
        ),
        *role_oh,
    ]


REL_SELF = 0
REL_TEAM = 1
REL_OPPONENT = 2
REL_WINNER_CONTESTANT_SUPPORT = 3
REL_LOSER_CONTESTANT_SUPPORT = 4
REL_DUEL = 5

N_REL = 6
REL_EMBED = 8
EDGE_DIM = 6


def build_edges(nodes):
    n = len(
        nodes
    )

    coords = np.asarray(
        [
            [
                float(
                    node.get(
                        "dx",
                        0.0,
                    )
                ),
                float(
                    node.get(
                        "dy",
                        0.0,
                    )
                ),
            ]
            for node in nodes
        ],
        dtype=float,
    )

    roles = [
        role_of(
            node
        )
        for node in nodes
    ]

    sides = np.asarray(
        [
            side_of(
                role
            )
            for role in roles
        ],
        dtype=int,
    )

    edges = set()

    for i in range(n):
        edges.add(
            (
                i,
                i,
                REL_SELF,
            )
        )

    wi = next(
        (
            i
            for i, role in enumerate(
                roles
            )
            if role
            == "aerial_winner"
        ),
        None,
    )

    li = next(
        (
            i
            for i, role in enumerate(
                roles
            )
            if role
            == "aerial_loser"
        ),
        None,
    )

    if (
        wi is not None
        and li is not None
    ):
        edges.add(
            (
                wi,
                li,
                REL_DUEL,
            )
        )
        edges.add(
            (
                li,
                wi,
                REL_DUEL,
            )
        )

    for contestant_idx, relation in [
        (
            wi,
            REL_WINNER_CONTESTANT_SUPPORT,
        ),
        (
            li,
            REL_LOSER_CONTESTANT_SUPPORT,
        ),
    ]:
        if contestant_idx is None:
            continue

        same_side = [
            j
            for j in range(n)
            if (
                j != contestant_idx
                and sides[j]
                == sides[
                    contestant_idx
                ]
            )
        ]

        for j in same_side:
            edges.add(
                (
                    contestant_idx,
                    j,
                    relation,
                )
            )
            edges.add(
                (
                    j,
                    contestant_idx,
                    relation,
                )
            )

    for i in range(n):
        same = [
            j
            for j in range(n)
            if (
                j != i
                and sides[j]
                == sides[i]
                and sides[i]
                != 0
            )
        ]

        same.sort(
            key=lambda j:
            float(
                np.linalg.norm(
                    coords[j]
                    - coords[i]
                )
            )
        )

        for j in same[
            :SAME_TEAM_K
        ]:
            edges.add(
                (
                    i,
                    j,
                    REL_TEAM,
                )
            )

        opp = [
            j
            for j in range(n)
            if (
                j != i
                and sides[j]
                != 0
                and sides[i]
                != 0
                and sides[j]
                != sides[i]
            )
        ]

        opp.sort(
            key=lambda j:
            float(
                np.linalg.norm(
                    coords[j]
                    - coords[i]
                )
            )
        )

        for j in opp[
            :OPPONENT_K
        ]:
            edges.add(
                (
                    i,
                    j,
                    REL_OPPONENT,
                )
            )

    srcs = []
    dsts = []
    rels = []
    attrs = []

    for src, dst, rel in sorted(
        edges
    ):
        dx = (
            coords[
                src,
                0,
            ]
            - coords[
                dst,
                0,
            ]
        )

        dy = (
            coords[
                src,
                1,
            ]
            - coords[
                dst,
                1,
            ]
        )

        dist = math.hypot(
            dx,
            dy,
        )

        src_r = math.hypot(
            coords[
                src,
                0,
            ],
            coords[
                src,
                1,
            ],
        )

        dst_r = math.hypot(
            coords[
                dst,
                0,
            ],
            coords[
                dst,
                1,
            ],
        )

        if dist > 1e-8:
            ux = dx / dist
            uy = dy / dist
        else:
            ux = 0.0
            uy = 0.0

        srcs.append(
            src
        )
        dsts.append(
            dst
        )
        rels.append(
            rel
        )

        attrs.append(
            [
                dx / 15.0,
                dy / 15.0,
                dist / 15.0,
                ux,
                uy,
                (
                    src_r
                    - dst_r
                )
                / 15.0,
            ]
        )

    return (
        np.asarray(
            srcs,
            dtype=np.int64,
        ),
        np.asarray(
            dsts,
            dtype=np.int64,
        ),
        np.asarray(
            rels,
            dtype=np.int64,
        ),
        np.asarray(
            attrs,
            dtype=np.float32,
        ),
    )


# ============================================================
# PRECOMPUTE GRAPHS
# ============================================================

samples = []

print()
print(
    "Preparing graph tensors..."
)

for i, eid in enumerate(
    event_ids,
    start=1,
):
    graph = graphs_by_id[
        eid
    ]

    nodes = graph[
        "nodes"
    ]

    roles = [
        role_of(
            node
        )
        for node in nodes
    ]

    winner_mask = np.asarray(
        [
            role
            in WINNER_ROLES
            for role in roles
        ],
        dtype=bool,
    )

    loser_mask = np.asarray(
        [
            role
            in LOSER_ROLES
            for role in roles
        ],
        dtype=bool,
    )

    wi = next(
        (
            j
            for j, role in enumerate(
                roles
            )
            if role
            == "aerial_winner"
        ),
        None,
    )

    li = next(
        (
            j
            for j, role in enumerate(
                roles
            )
            if role
            == "aerial_loser"
        ),
        None,
    )

    if (
        wi is None
        or li is None
    ):
        raise RuntimeError(
            f"Missing contestant node "
            f"for {eid}"
        )

    edge_src, edge_dst, edge_rel, edge_attr = (
        build_edges(
            nodes
        )
    )

    samples.append(
        {
            "x": np.asarray(
                [
                    node_feature(
                        node
                    )
                    for node in nodes
                ],
                dtype=np.float32,
            ),
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_rel": edge_rel,
            "edge_attr": edge_attr,
            "winner_mask": winner_mask,
            "loser_mask": loser_mask,
            "winner_contestant": int(
                wi
            ),
            "loser_contestant": int(
                li
            ),
        }
    )

    if (
        i % 500 == 0
        or i == len(event_ids)
    ):
        print(
            f"  prepared "
            f"{i}/{len(event_ids)}"
        )


# ============================================================
# CONTEXT SCALER
# ============================================================

def fit_context_scaler(
    train_idx
):
    train = raw_context[
        train_idx
    ]

    mean = np.nanmean(
        train,
        axis=0,
    )

    std = np.nanstd(
        train,
        axis=0,
    )

    mean = np.where(
        np.isfinite(
            mean
        ),
        mean,
        0.0,
    )

    std = np.where(
        (
            np.isfinite(
                std
            )
            & (
                std
                > 1e-8
            )
        ),
        std,
        1.0,
    )

    return (
        mean,
        std,
    )


def transform_context(
    mean,
    std,
):
    x = raw_context.copy()

    for j in range(
        x.shape[1]
    ):
        missing = ~np.isfinite(
            x[
                :,
                j
            ]
        )

        x[
            missing,
            j
        ] = mean[
            j
        ]

    x = (
        x
        - mean
    ) / std

    return np.clip(
        x,
        -5.0,
        5.0,
    ).astype(
        np.float32
    )


# ============================================================
# BATCH COLLATION
# ============================================================

def collate(
    indices,
    context_matrix,
):
    xs = []
    srcs = []
    dsts = []
    rels = []
    attrs = []

    node_graph = []
    winner_masks = []
    loser_masks = []

    winner_contestants = []
    loser_contestants = []

    node_offset = 0

    for local_graph_idx, sample_idx in enumerate(
        indices
    ):
        s = samples[
            int(
                sample_idx
            )
        ]

        n = s[
            "x"
        ].shape[0]

        xs.append(
            s[
                "x"
            ]
        )

        srcs.append(
            s[
                "edge_src"
            ]
            + node_offset
        )

        dsts.append(
            s[
                "edge_dst"
            ]
            + node_offset
        )

        rels.append(
            s[
                "edge_rel"
            ]
        )

        attrs.append(
            s[
                "edge_attr"
            ]
        )

        node_graph.append(
            np.full(
                n,
                local_graph_idx,
                dtype=np.int64,
            )
        )

        winner_masks.append(
            s[
                "winner_mask"
            ]
        )

        loser_masks.append(
            s[
                "loser_mask"
            ]
        )

        winner_contestants.append(
            node_offset
            + s[
                "winner_contestant"
            ]
        )

        loser_contestants.append(
            node_offset
            + s[
                "loser_contestant"
            ]
        )

        node_offset += n

    return {
        "x": torch.tensor(
            np.concatenate(
                xs,
                axis=0,
            ),
            dtype=torch.float32,
            device=DEVICE,
        ),
        "edge_src": torch.tensor(
            np.concatenate(
                srcs,
                axis=0,
            ),
            dtype=torch.long,
            device=DEVICE,
        ),
        "edge_dst": torch.tensor(
            np.concatenate(
                dsts,
                axis=0,
            ),
            dtype=torch.long,
            device=DEVICE,
        ),
        "edge_rel": torch.tensor(
            np.concatenate(
                rels,
                axis=0,
            ),
            dtype=torch.long,
            device=DEVICE,
        ),
        "edge_attr": torch.tensor(
            np.concatenate(
                attrs,
                axis=0,
            ),
            dtype=torch.float32,
            device=DEVICE,
        ),
        "node_graph": torch.tensor(
            np.concatenate(
                node_graph,
                axis=0,
            ),
            dtype=torch.long,
            device=DEVICE,
        ),
        "winner_mask": torch.tensor(
            np.concatenate(
                winner_masks,
                axis=0,
            ),
            dtype=torch.bool,
            device=DEVICE,
        ),
        "loser_mask": torch.tensor(
            np.concatenate(
                loser_masks,
                axis=0,
            ),
            dtype=torch.bool,
            device=DEVICE,
        ),
        "winner_contestant": torch.tensor(
            winner_contestants,
            dtype=torch.long,
            device=DEVICE,
        ),
        "loser_contestant": torch.tensor(
            loser_contestants,
            dtype=torch.long,
            device=DEVICE,
        ),
        "context": torch.tensor(
            context_matrix[
                indices
            ],
            dtype=torch.float32,
            device=DEVICE,
        ),
        "target": torch.tensor(
            y[
                indices
            ],
            dtype=torch.float32,
            device=DEVICE,
        ),
        "num_graphs": len(
            indices
        ),
    }


# ============================================================
# MODEL
# ============================================================

class SharedMessageLayer(
    nn.Module
):
    def __init__(
        self,
        hidden,
        dropout,
    ):
        super().__init__()

        self.rel_embedding = nn.Embedding(
            N_REL,
            REL_EMBED,
        )

        self.message = nn.Sequential(
            nn.Linear(
                hidden * 2
                + EDGE_DIM
                + REL_EMBED,
                hidden,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden,
                hidden,
            ),
            nn.ReLU(),
        )

        self.update = nn.Sequential(
            nn.Linear(
                hidden * 2,
                hidden,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden,
                hidden,
            ),
        )

        self.norm = nn.LayerNorm(
            hidden
        )

    def forward(
        self,
        h,
        src,
        dst,
        rel,
        edge_attr,
    ):
        rel_emb = (
            self.rel_embedding(
                rel
            )
        )

        msg = self.message(
            torch.cat(
                [
                    h[
                        dst
                    ],
                    h[
                        src
                    ],
                    edge_attr,
                    rel_emb,
                ],
                dim=1,
            )
        )

        agg = torch.zeros_like(
            h
        )

        agg.index_add_(
            0,
            dst,
            msg,
        )

        counts = torch.zeros(
            h.shape[0],
            dtype=h.dtype,
            device=h.device,
        )

        counts.index_add_(
            0,
            dst,
            torch.ones(
                dst.shape[0],
                dtype=h.dtype,
                device=h.device,
            ),
        )

        agg = (
            agg
            / counts
            .clamp_min(
                1.0
            )
            .unsqueeze(
                1
            )
        )

        update = self.update(
            torch.cat(
                [
                    h,
                    agg,
                ],
                dim=1,
            )
        )

        return self.norm(
            h + update
        )


def masked_team_mean(
    h,
    mask,
    node_graph,
    num_graphs,
):
    selected_h = h[
        mask
    ]

    selected_graph = (
        node_graph[
            mask
        ]
    )

    out = torch.zeros(
        (
            num_graphs,
            h.shape[1],
        ),
        dtype=h.dtype,
        device=h.device,
    )

    counts = torch.zeros(
        num_graphs,
        dtype=h.dtype,
        device=h.device,
    )

    out.index_add_(
        0,
        selected_graph,
        selected_h,
    )

    counts.index_add_(
        0,
        selected_graph,
        torch.ones(
            selected_graph.shape[0],
            dtype=h.dtype,
            device=h.device,
        ),
    )

    return (
        out
        / counts
        .clamp_min(
            1.0
        )
        .unsqueeze(
            1
        )
    )


class PositioningGNN(
    nn.Module
):
    def __init__(
        self,
        context_dim,
    ):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(
                NODE_DIM,
                HIDDEN,
            ),
            nn.ReLU(),
            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [
                SharedMessageLayer(
                    HIDDEN,
                    DROPOUT,
                )
                for _ in range(
                    LAYERS
                )
            ]
        )

        graph_dim = (
            HIDDEN * 6
        )

        self.head = nn.Sequential(
            nn.Linear(
                graph_dim
                + context_dim,
                96,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                96,
                48,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                48,
                1,
            ),
        )

    def forward(
        self,
        batch,
    ):
        h = self.node_encoder(
            batch[
                "x"
            ]
        )

        for layer in self.layers:
            h = layer(
                h,
                batch[
                    "edge_src"
                ],
                batch[
                    "edge_dst"
                ],
                batch[
                    "edge_rel"
                ],
                batch[
                    "edge_attr"
                ],
            )

        num_graphs = batch[
            "num_graphs"
        ]

        w = masked_team_mean(
            h,
            batch[
                "winner_mask"
            ],
            batch[
                "node_graph"
            ],
            num_graphs,
        )

        l = masked_team_mean(
            h,
            batch[
                "loser_mask"
            ],
            batch[
                "node_graph"
            ],
            num_graphs,
        )

        diff = w - l

        abs_diff = torch.abs(
            diff
        )

        wc = h[
            batch[
                "winner_contestant"
            ]
        ]

        lc = h[
            batch[
                "loser_contestant"
            ]
        ]

        z = torch.cat(
            [
                w,
                l,
                diff,
                abs_diff,
                wc,
                lc,
                batch[
                    "context"
                ],
            ],
            dim=1,
        )

        return (
            self.head(
                z
            )
            .squeeze(
                1
            )
        )


def make_batches(
    indices,
    shuffle,
    seed,
):
    indices = np.asarray(
        indices,
        dtype=int,
    ).copy()

    if shuffle:
        rng = np.random.default_rng(
            seed
        )

        rng.shuffle(
            indices
        )

    for start in range(
        0,
        len(indices),
        BATCH_SIZE,
    ):
        yield indices[
            start:
            start
            + BATCH_SIZE
        ]


# ============================================================
# TRAIN
# ============================================================

def train_fold(
    train_idx,
    val_idx,
    fold_seed,
):
    random.seed(
        fold_seed
    )

    np.random.seed(
        fold_seed
    )

    torch.manual_seed(
        fold_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            fold_seed
        )

    mean, std = fit_context_scaler(
        train_idx
    )

    context = transform_context(
        mean,
        std,
    )

    model = PositioningGNN(
        context_dim=
        context.shape[
            1
        ]
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=
        WEIGHT_DECAY,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    for epoch in range(
        1,
        EPOCHS
        + 1,
    ):
        model.train()

        total_loss = 0.0
        total_n = 0

        for idx in make_batches(
            train_idx,
            shuffle=True,
            seed=
            fold_seed
            + epoch,
        ):
            batch = collate(
                idx,
                context,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch
            )

            loss = criterion(
                logits,
                batch[
                    "target"
                ],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            n = len(
                idx
            )

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * n
            )

            total_n += n

        if (
            epoch == 1
            or epoch % 6 == 0
            or epoch == EPOCHS
        ):
            print(
                f"    epoch "
                f"{epoch:>2}/{EPOCHS} | "
                f"train_loss="
                f"{total_loss/total_n:.4f}"
            )

    model.eval()

    probs = []

    with torch.no_grad():
        for idx in make_batches(
            val_idx,
            shuffle=False,
            seed=
            fold_seed,
        ):
            batch = collate(
                idx,
                context,
            )

            logits = model(
                batch
            )

            probs.extend(
                torch.sigmoid(
                    logits
                )
                .cpu()
                .numpy()
                .tolist()
            )

    return np.asarray(
        probs,
        dtype=float,
    )


# ============================================================
# OOF
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

p_gnn = np.full(
    len(rows),
    np.nan,
)

print()
print(
    "=" * 108
)
print(
    "5-FOLD WHOLE-MATCH OOF â€” EXACT PRE-CONTACT"
)
print(
    "=" * 108
)

for fold, (
    train_idx,
    val_idx,
) in enumerate(
    cv.split(
        np.zeros(
            len(rows)
        ),
        y,
        groups,
    ),
    start=1,
):
    print()
    print(
        f"Fold {fold}/{N_SPLITS}: "
        f"train={len(train_idx):,} | "
        f"validation={len(val_idx):,}"
    )

    p = train_fold(
        train_idx,
        val_idx,
        SEED
        + fold,
    )

    p_gnn[
        val_idx
    ] = p

    m = metrics(
        y[
            val_idx
        ],
        p,
    )

    print(
        f"    fold result: "
        f"LL={m['log_loss']:.4f} | "
        f"ROC={m['roc_auc']:.4f} | "
        f"PR={m['pr_auc']:.4f} | "
        f"Brier={m['brier']:.4f}"
    )


if np.isnan(
    p_gnn
).any():
    raise RuntimeError(
        "Missing OOF predictions."
    )


gnn_metrics = metrics(
    y,
    p_gnn,
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print(
    "=" * 108
)
print(
    "EXACT PRE-CONTACT INFORMATION-STAGE COMPARISON"
)
print(
    "=" * 108
)

print()
print(
    "Positioning GNN             "
    f"LL={gnn_metrics['log_loss']:.4f} | "
    f"ROC={gnn_metrics['roc_auc']:.4f} | "
    f"PR={gnn_metrics['pr_auc']:.4f} | "
    f"Brier={gnn_metrics['brier']:.4f}"
)


if comparator is not None:
    _, _, p_xgb = comparator

    xgb_metrics = metrics(
        y,
        p_xgb,
    )

    print(
        "Setup+incoming XGB         "
        f"LL={xgb_metrics['log_loss']:.4f} | "
        f"ROC={xgb_metrics['roc_auc']:.4f} | "
        f"PR={xgb_metrics['pr_auc']:.4f} | "
        f"Brier={xgb_metrics['brier']:.4f}"
    )

else:
    p_xgb = None

    xgb_metrics = {
        "log_loss": 0.5032,
        "roc_auc": 0.6416,
        "pr_auc": 0.3074,
        "brier": 0.1627,
    }

    print(
        "Setup+incoming XGB*        "
        "LL=0.5032 | ROC=0.6416 | "
        "PR=0.3074 | Brier=0.1627"
    )

    print(
        "* aggregate reference from script 31"
    )


# ============================================================
# SAVE
# ============================================================

oof = pd.DataFrame(
    {
        "match_id": groups,
        "winner_event_id": event_ids,
        "target": y,
        "p_positioning_gnn_exact_precontact": p_gnn,
    }
)

if p_xgb is not None:
    oof[
        "p_setup_plus_incoming_xgb"
    ] = p_xgb

oof.to_csv(
    OOF_OUT,
    index=False,
)


pd.DataFrame(
    [
        {
            "model":
            "positioning_gnn_exact_precontact",
            **gnn_metrics,
        },
        {
            "model":
            "setup_plus_incoming_xgb",
            **xgb_metrics,
        },
    ]
).to_csv(
    RESULTS_OUT,
    index=False,
)


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

if p_xgb is not None:
    by_match = defaultdict(
        list
    )

    for i, match_id in enumerate(
        groups
    ):
        by_match[
            match_id
        ].append(
            i
        )

    match_ids = list(
        by_match.keys()
    )

    rng = np.random.default_rng(
        20260919
    )

    boot_rows = []

    for b in range(
        N_BOOT
    ):
        sampled = rng.choice(
            match_ids,
            size=len(
                match_ids
            ),
            replace=True,
        )

        idx = np.concatenate(
            [
                np.asarray(
                    by_match[
                        m
                    ],
                    dtype=int,
                )
                for m in sampled
            ]
        )

        yy = y[
            idx
        ]

        pg = p_gnn[
            idx
        ]

        px = p_xgb[
            idx
        ]

        boot_rows.append(
            {
                "iteration":
                b + 1,

                "delta_log_loss_gnn_minus_xgb":
                (
                    log_loss(
                        yy,
                        pg,
                        labels=[
                            0,
                            1,
                        ],
                    )
                    -
                    log_loss(
                        yy,
                        px,
                        labels=[
                            0,
                            1,
                        ],
                    )
                ),

                "delta_brier_gnn_minus_xgb":
                (
                    brier_score_loss(
                        yy,
                        pg,
                    )
                    -
                    brier_score_loss(
                        yy,
                        px,
                    )
                ),

                "delta_pr_gnn_minus_xgb":
                (
                    safe_pr(
                        yy,
                        pg,
                    )
                    -
                    safe_pr(
                        yy,
                        px,
                    )
                ),

                "delta_roc_gnn_minus_xgb":
                (
                    safe_auc(
                        yy,
                        pg,
                    )
                    -
                    safe_auc(
                        yy,
                        px,
                    )
                ),
            }
        )

    boot = pd.DataFrame(
        boot_rows
    )

    boot.to_csv(
        BOOT_OUT,
        index=False,
    )

    print()
    print(
        "=" * 108
    )
    print(
        "PAIRED MATCH-CLUSTERED BOOTSTRAP"
    )
    print(
        "=" * 108
    )

    for col, label, lower_better in [
        (
            "delta_log_loss_gnn_minus_xgb",
            "Log loss",
            True,
        ),
        (
            "delta_brier_gnn_minus_xgb",
            "Brier",
            True,
        ),
        (
            "delta_pr_gnn_minus_xgb",
            "PR-AUC",
            False,
        ),
        (
            "delta_roc_gnn_minus_xgb",
            "ROC-AUC",
            False,
        ),
    ]:
        vals = (
            boot[col]
            .dropna()
            .to_numpy()
        )

        mean = vals.mean()

        lo, hi = np.percentile(
            vals,
            [
                2.5,
                97.5,
            ],
        )

        frac = (
            np.mean(
                vals < 0
            )
            if lower_better
            else np.mean(
                vals > 0
            )
        )

        print(
            f"{label:<9} "
            f"delta GNN-XGB="
            f"{mean:+.4f} | "
            f"95% CI "
            f"[{lo:+.4f}, {hi:+.4f}] | "
            f"GNN better in "
            f"{100*frac:.1f}%"
        )


print()
print(
    "Saved:"
)
print(
    f"  {FEATURE_AUDIT_OUT}"
)
print(
    f"  {OOF_OUT}"
)
print(
    f"  {RESULTS_OUT}"
)

if p_xgb is not None:
    print(
        f"  {BOOT_OUT}"
    )

print()
print(
    "=" * 108
)
print(
    "DONE"
)
print(
    "=" * 108
)

