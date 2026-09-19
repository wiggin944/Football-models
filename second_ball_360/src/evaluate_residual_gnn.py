import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

GRAPH_PATH = OUT / "final_graph_dataset_primary.jsonl"
FEATURE_PATH = OUT / "handcrafted_features.csv"
STRICT_PATH = OUT / "true_second_ball_strict_sample.csv"
INCOMING_CACHE = OUT / "exact_precontact_context_cache.csv"

OOF_OUT = OUT / "residual_graph_correction_oof.csv"
RESULTS_OUT = OUT / "residual_graph_correction_results.csv"
FOLDS_OUT = OUT / "residual_graph_correction_folds.csv"
BOOT_OUT = OUT / "residual_graph_correction_bootstrap.csv"

SEED = 42
N_SPLITS = 5
INNER_SPLITS = 4

EPOCHS = 18
BATCH_SIZE = 96
HIDDEN = 48
LAYERS = 2
DROPOUT = 0.30
LR = 7e-4
WEIGHT_DECAY = 1e-3
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
# EXACT PRE-CONTACT FEATURES FROM SCRIPT 31
# ============================================================

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

XGB_FEATURES = []

for c in SETUP_GEOMETRY + INCOMING_FEATURES:
    if c not in XGB_FEATURES:
        XGB_FEATURES.append(c)

# The graph gets raw player structure, so only non-graph context
# is supplied alongside it.
GNN_CONTEXT = [
    "contest_x",
    "contest_y_from_centre",
] + INCOMING_FEATURES


# ============================================================
# HELPERS
# ============================================================

def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def clip_prob(p):
    return np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )


def logit(p):
    p = clip_prob(p)
    return np.log(
        p / (1.0 - p)
    )


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
    p = clip_prob(p)

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


def new_xgb(seed):
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
        random_state=seed,
        n_jobs=-1,
    )


# ============================================================
# LOAD DATA
# ============================================================

strict = pd.read_csv(STRICT_PATH)

strict["winner_event_id"] = (
    strict["winner_event_id"]
    .astype(str)
)

strict_ids = set(
    strict["winner_event_id"]
)

features = pd.read_csv(FEATURE_PATH)

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
        f"found {len(rows)}."
    )

if "contest_y" not in rows.columns:
    raise KeyError(
        "handcrafted_features.csv is missing contest_y."
    )

rows["contest_y_from_centre"] = np.abs(
    pd.to_numeric(
        rows["contest_y"],
        errors="raise",
    ).to_numpy(dtype=float)
    - 40.0
)


if not INCOMING_CACHE.exists():
    raise FileNotFoundError(
        f"Missing exact incoming cache: {INCOMING_CACHE}"
    )

incoming = pd.read_csv(INCOMING_CACHE)

incoming["winner_event_id"] = (
    incoming["winner_event_id"]
    .astype(str)
)

missing_incoming = [
    c
    for c in INCOMING_FEATURES
    if c not in incoming.columns
]

if missing_incoming:
    raise RuntimeError(
        "Incoming cache is missing: "
        + ", ".join(missing_incoming)
    )

rows = rows.merge(
    incoming[
        ["winner_event_id"]
        + INCOMING_FEATURES
    ],
    on="winner_event_id",
    how="left",
    validate="one_to_one",
)


target_col = first_existing(
    rows.columns,
    ["target", "label", "y"],
)

match_col = first_existing(
    rows.columns,
    ["match_id", "match"],
)

if target_col is None:
    raise KeyError("Could not find target column.")

if match_col is None:
    raise KeyError("Could not find match column.")


missing_features = [
    c
    for c in XGB_FEATURES
    if c not in rows.columns
]

if missing_features:
    raise RuntimeError(
        "Missing XGB features: "
        + ", ".join(missing_features)
    )


event_ids = (
    rows["winner_event_id"]
    .astype(str)
    .to_numpy()
)

groups = (
    rows[match_col]
    .astype(str)
    .to_numpy()
)

y = (
    pd.to_numeric(
        rows[target_col],
        errors="raise",
    )
    .astype(int)
    .to_numpy()
)

is_clearance = None

if "is_clearance" in rows.columns:
    is_clearance = (
        pd.to_numeric(
            rows["is_clearance"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .to_numpy()
    )


X_xgb = rows[
    XGB_FEATURES
].apply(
    pd.to_numeric,
    errors="coerce",
).to_numpy(
    dtype=float
)

context_raw = rows[
    GNN_CONTEXT
].apply(
    pd.to_numeric,
    errors="coerce",
).to_numpy(
    dtype=float
)


print("=" * 108)
print("RESIDUAL GRAPH CORRECTION")
print("=" * 108)

print()
print(f"Strict episodes: {len(rows):,}")
print(f"Matches:         {len(set(groups)):,}")
print(f"Positive rate:   {100*y.mean():.1f}%")
print(f"XGB features:    {len(XGB_FEATURES)}")
print(f"GNN context:     {len(GNN_CONTEXT)}")
print(f"Device:          {DEVICE}")

print()
print(
    "Question: after setup+incoming XGB has made its prediction, "
    "does the raw player graph improve it?"
)


# ============================================================
# LOAD GRAPHS
# ============================================================

graphs = {}

with open(
    GRAPH_PATH,
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        if not line.strip():
            continue

        graph = json.loads(line)

        eid = str(
            graph["winner_event_id"]
        )

        if eid in strict_ids:
            graphs[eid] = graph


missing_graphs = (
    set(event_ids)
    - set(graphs)
)

if missing_graphs:
    raise RuntimeError(
        f"Missing {len(missing_graphs)} strict graphs."
    )


# ============================================================
# GRAPH REPRESENTATION
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
    for i, role in enumerate(ROLE_ORDER)
}

NODE_DIM = 12

REL_SELF = 0
REL_TEAM = 1
REL_OPPONENT = 2
REL_WINNER_CONTESTANT_SUPPORT = 3
REL_LOSER_CONTESTANT_SUPPORT = 4
REL_DUEL = 5

N_REL = 6
REL_EMBED = 8
EDGE_DIM = 6


def role_of(node):
    return str(
        node.get("role", "")
    )


def side_of(role):
    if role in WINNER_ROLES:
        return 1

    if role in LOSER_ROLES:
        return -1

    return 0


def keeper_flag(node):
    return (
        1.0
        if bool(
            node.get(
                "keeper",
                node.get(
                    "is_keeper",
                    False,
                ),
            )
        )
        else 0.0
    )


def node_feature(node):
    dx = float(
        node.get("dx", 0.0)
    )

    dy = float(
        node.get("dy", 0.0)
    )

    dist = node.get(
        "distance_to_contest",
        node.get(
            "distance",
            None,
        ),
    )

    if dist is None:
        dist = math.hypot(
            dx,
            dy,
        )

    dist = float(dist)

    angle = math.atan2(
        dy,
        dx,
    )

    role = role_of(node)

    onehot = [0.0] * 4

    if role in ROLE_TO_INDEX:
        onehot[
            ROLE_TO_INDEX[role]
        ] = 1.0

    return [
        dx / 15.0,
        dy / 15.0,
        dist / 15.0,
        math.sin(angle),
        math.cos(angle),
        float(side_of(role)),
        (
            1.0
            if role in {
                "aerial_winner",
                "aerial_loser",
            }
            else 0.0
        ),
        keeper_flag(node),
        *onehot,
    ]


def build_edges(nodes):
    n = len(nodes)

    coords = np.asarray(
        [
            [
                float(
                    node.get("dx", 0.0)
                ),
                float(
                    node.get("dy", 0.0)
                ),
            ]
            for node in nodes
        ],
        dtype=float,
    )

    roles = [
        role_of(node)
        for node in nodes
    ]

    sides = np.asarray(
        [
            side_of(role)
            for role in roles
        ],
        dtype=int,
    )

    wi = next(
        (
            i
            for i, role in enumerate(roles)
            if role == "aerial_winner"
        ),
        None,
    )

    li = next(
        (
            i
            for i, role in enumerate(roles)
            if role == "aerial_loser"
        ),
        None,
    )

    if wi is None or li is None:
        raise RuntimeError(
            "Contestant node missing."
        )

    edges = set()

    for i in range(n):
        edges.add(
            (i, i, REL_SELF)
        )

    edges.add(
        (wi, li, REL_DUEL)
    )
    edges.add(
        (li, wi, REL_DUEL)
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
        same_side = [
            j
            for j in range(n)
            if (
                j != contestant_idx
                and sides[j]
                == sides[contestant_idx]
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
                and sides[j] == sides[i]
                and sides[i] != 0
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

        for j in same[:SAME_TEAM_K]:
            edges.add(
                (i, j, REL_TEAM)
            )

        opp = [
            j
            for j in range(n)
            if (
                j != i
                and sides[j] != 0
                and sides[i] != 0
                and sides[j] != sides[i]
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

        for j in opp[:OPPONENT_K]:
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

    for src, dst, rel in sorted(edges):
        dx = (
            coords[src, 0]
            - coords[dst, 0]
        )

        dy = (
            coords[src, 1]
            - coords[dst, 1]
        )

        dist = math.hypot(
            dx,
            dy,
        )

        if dist > 1e-8:
            ux = dx / dist
            uy = dy / dist
        else:
            ux = 0.0
            uy = 0.0

        src_r = math.hypot(
            coords[src, 0],
            coords[src, 1],
        )

        dst_r = math.hypot(
            coords[dst, 0],
            coords[dst, 1],
        )

        srcs.append(src)
        dsts.append(dst)
        rels.append(rel)

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
        wi,
        li,
    )


samples = []

print()
print("Preparing graph tensors...")

for i, eid in enumerate(
    event_ids,
    start=1,
):
    nodes = graphs[
        eid
    ]["nodes"]

    roles = [
        role_of(node)
        for node in nodes
    ]

    winner_mask = np.asarray(
        [
            role in WINNER_ROLES
            for role in roles
        ],
        dtype=bool,
    )

    loser_mask = np.asarray(
        [
            role in LOSER_ROLES
            for role in roles
        ],
        dtype=bool,
    )

    (
        edge_src,
        edge_dst,
        edge_rel,
        edge_attr,
        wi,
        li,
    ) = build_edges(nodes)

    samples.append(
        {
            "x": np.asarray(
                [
                    node_feature(node)
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
            "winner_contestant": int(wi),
            "loser_contestant": int(li),
        }
    )

    if (
        i % 500 == 0
        or i == len(event_ids)
    ):
        print(
            f"  prepared {i}/{len(event_ids)}"
        )


# ============================================================
# CONTEXT SCALING
# ============================================================

def scale_context(train_idx):
    train = context_raw[
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
        np.isfinite(mean),
        mean,
        0.0,
    )

    std = np.where(
        (
            np.isfinite(std)
            & (std > 1e-8)
        ),
        std,
        1.0,
    )

    x = context_raw.copy()

    for j in range(
        x.shape[1]
    ):
        bad = ~np.isfinite(
            x[:, j]
        )

        x[
            bad,
            j
        ] = mean[j]

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
# XGB BASELINE WITH NESTED OOF TRAIN OFFSETS
# ============================================================

def nested_xgb_offsets(
    outer_train_idx,
    outer_val_idx,
    fold_seed,
):
    outer_train_idx = np.asarray(
        outer_train_idx,
        dtype=int,
    )

    outer_val_idx = np.asarray(
        outer_val_idx,
        dtype=int,
    )

    p = np.full(
        len(rows),
        np.nan,
        dtype=float,
    )

    inner_cv = StratifiedGroupKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=
        fold_seed + 1000,
    )

    inner_y = y[
        outer_train_idx
    ]

    inner_groups = groups[
        outer_train_idx
    ]

    # OOF baseline predictions for residual-GNN training rows.
    for inner_fold, (
        inner_train_pos,
        inner_hold_pos,
    ) in enumerate(
        inner_cv.split(
            np.zeros(
                len(
                    outer_train_idx
                )
            ),
            inner_y,
            inner_groups,
        ),
        start=1,
    ):
        inner_train_idx = (
            outer_train_idx[
                inner_train_pos
            ]
        )

        inner_hold_idx = (
            outer_train_idx[
                inner_hold_pos
            ]
        )

        model = new_xgb(
            fold_seed * 100
            + inner_fold
        )

        model.fit(
            X_xgb[
                inner_train_idx
            ],
            y[
                inner_train_idx
            ],
        )

        p[
            inner_hold_idx
        ] = model.predict_proba(
            X_xgb[
                inner_hold_idx
            ]
        )[:, 1]

    # Standard outer-fold XGB prediction for unseen validation.
    outer_model = new_xgb(
        fold_seed
    )

    outer_model.fit(
        X_xgb[
            outer_train_idx
        ],
        y[
            outer_train_idx
        ],
    )

    p[
        outer_val_idx
    ] = outer_model.predict_proba(
        X_xgb[
            outer_val_idx
        ]
    )[:, 1]

    needed = np.concatenate(
        [
            outer_train_idx,
            outer_val_idx,
        ]
    )

    if np.isnan(
        p[
            needed
        ]
    ).any():
        raise RuntimeError(
            "Missing nested XGB offsets."
        )

    return p


# ============================================================
# BATCHING
# ============================================================

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

        rng.shuffle(indices)

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


def collate(
    indices,
    context,
    baseline_probs,
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

    for local_graph_idx, idx in enumerate(
        indices
    ):
        s = samples[
            int(idx)
        ]

        n = s["x"].shape[0]

        xs.append(
            s["x"]
        )

        srcs.append(
            s["edge_src"]
            + node_offset
        )

        dsts.append(
            s["edge_dst"]
            + node_offset
        )

        rels.append(
            s["edge_rel"]
        )

        attrs.append(
            s["edge_attr"]
        )

        node_graph.append(
            np.full(
                n,
                local_graph_idx,
                dtype=np.int64,
            )
        )

        winner_masks.append(
            s["winner_mask"]
        )

        loser_masks.append(
            s["loser_mask"]
        )

        winner_contestants.append(
            node_offset
            + s["winner_contestant"]
        )

        loser_contestants.append(
            node_offset
            + s["loser_contestant"]
        )

        node_offset += n

    indices = np.asarray(
        indices,
        dtype=int,
    )

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
            context[
                indices
            ],
            dtype=torch.float32,
            device=DEVICE,
        ),
        "baseline_logit": torch.tensor(
            logit(
                baseline_probs[
                    indices
                ]
            ),
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
        "num_graphs": len(indices),
    }


# ============================================================
# MODEL
# ============================================================

class MessageLayer(
    nn.Module
):
    def __init__(self):
        super().__init__()

        self.rel_embedding = nn.Embedding(
            N_REL,
            REL_EMBED,
        )

        self.message = nn.Sequential(
            nn.Linear(
                HIDDEN * 2
                + EDGE_DIM
                + REL_EMBED,
                HIDDEN,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),
            nn.ReLU(),
        )

        self.update = nn.Sequential(
            nn.Linear(
                HIDDEN * 2,
                HIDDEN,
            ),
            nn.ReLU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),
        )

        self.norm = nn.LayerNorm(
            HIDDEN
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
            self.rel_embedding(rel)
        )

        msg = self.message(
            torch.cat(
                [
                    h[dst],
                    h[src],
                    edge_attr,
                    rel_emb,
                ],
                dim=1,
            )
        )

        agg = torch.zeros_like(h)

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
                len(dst),
                dtype=h.dtype,
                device=h.device,
            ),
        )

        agg = (
            agg
            / counts
            .clamp_min(1.0)
            .unsqueeze(1)
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


def masked_mean(
    h,
    mask,
    graph_idx,
    n_graphs,
):
    selected_h = h[
        mask
    ]

    selected_graph = (
        graph_idx[
            mask
        ]
    )

    out = torch.zeros(
        (
            n_graphs,
            h.shape[1],
        ),
        dtype=h.dtype,
        device=h.device,
    )

    counts = torch.zeros(
        n_graphs,
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
            len(selected_graph),
            dtype=h.dtype,
            device=h.device,
        ),
    )

    return (
        out
        / counts
        .clamp_min(1.0)
        .unsqueeze(1)
    )


class ResidualGraphModel(
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
                MessageLayer()
                for _ in range(
                    LAYERS
                )
            ]
        )

        graph_dim = (
            HIDDEN * 6
            + context_dim
        )

        self.residual_head = nn.Sequential(
            nn.Linear(
                graph_dim,
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

        # Start exactly at the XGB baseline.
        nn.init.zeros_(
            self.residual_head[
                -1
            ].weight
        )

        nn.init.zeros_(
            self.residual_head[
                -1
            ].bias
        )

    def forward(
        self,
        batch,
    ):
        h = self.node_encoder(
            batch["x"]
        )

        for layer in self.layers:
            h = layer(
                h,
                batch["edge_src"],
                batch["edge_dst"],
                batch["edge_rel"],
                batch["edge_attr"],
            )

        n_graphs = batch[
            "num_graphs"
        ]

        winner = masked_mean(
            h,
            batch["winner_mask"],
            batch["node_graph"],
            n_graphs,
        )

        loser = masked_mean(
            h,
            batch["loser_mask"],
            batch["node_graph"],
            n_graphs,
        )

        winner_contestant = h[
            batch[
                "winner_contestant"
            ]
        ]

        loser_contestant = h[
            batch[
                "loser_contestant"
            ]
        ]

        z = torch.cat(
            [
                winner,
                loser,
                winner - loser,
                torch.abs(
                    winner - loser
                ),
                winner_contestant,
                loser_contestant,
                batch["context"],
            ],
            dim=1,
        )

        correction = (
            self.residual_head(z)
            .squeeze(1)
        )

        total_logit = (
            batch["baseline_logit"]
            + correction
        )

        return (
            total_logit,
            correction,
        )


# ============================================================
# TRAIN ONE OUTER FOLD
# ============================================================

def train_residual_fold(
    train_idx,
    val_idx,
    baseline_probs,
    fold_seed,
):
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            fold_seed
        )

    context = scale_context(
        train_idx
    )

    model = ResidualGraphModel(
        context_dim=
        context.shape[1]
    ).to(DEVICE)

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
        EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0
        total_n = 0
        total_abs_corr = 0.0

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
                baseline_probs,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            total_logit, correction = (
                model(batch)
            )

            loss = criterion(
                total_logit,
                batch["target"],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            n = len(idx)

            total_loss += (
                float(
                    loss.detach().cpu()
                )
                * n
            )

            total_abs_corr += (
                float(
                    torch.mean(
                        torch.abs(
                            correction
                        )
                    )
                    .detach()
                    .cpu()
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
                f"    epoch {epoch:>2}/{EPOCHS} | "
                f"loss={total_loss/total_n:.4f} | "
                f"mean|corr|="
                f"{total_abs_corr/total_n:.4f}"
            )

    model.eval()

    probs = []
    corrections = []

    with torch.no_grad():
        for idx in make_batches(
            val_idx,
            shuffle=False,
            seed=fold_seed,
        ):
            batch = collate(
                idx,
                context,
                baseline_probs,
            )

            total_logit, correction = (
                model(batch)
            )

            probs.extend(
                torch.sigmoid(
                    total_logit
                )
                .cpu()
                .numpy()
                .tolist()
            )

            corrections.extend(
                correction
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        np.asarray(
            probs,
            dtype=float,
        ),
        np.asarray(
            corrections,
            dtype=float,
        ),
    )


# ============================================================
# 5-FOLD WHOLE-MATCH OOF
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)

p_xgb_oof = np.full(
    len(rows),
    np.nan,
)

p_residual_oof = np.full(
    len(rows),
    np.nan,
)

correction_oof = np.full(
    len(rows),
    np.nan,
)

fold_rows = []


print()
print("=" * 108)
print("5-FOLD WHOLE-MATCH OOF")
print("=" * 108)

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

    baseline_probs = nested_xgb_offsets(
        train_idx,
        val_idx,
        SEED + fold,
    )

    p_xgb_oof[
        val_idx
    ] = baseline_probs[
        val_idx
    ]

    base_fold = metrics(
        y[
            val_idx
        ],
        baseline_probs[
            val_idx
        ],
    )

    print(
        f"    baseline XGB: "
        f"LL={base_fold['log_loss']:.4f} | "
        f"ROC={base_fold['roc_auc']:.4f} | "
        f"PR={base_fold['pr_auc']:.4f}"
    )

    p_resid, corr = (
        train_residual_fold(
            train_idx,
            val_idx,
            baseline_probs,
            SEED + fold,
        )
    )

    p_residual_oof[
        val_idx
    ] = p_resid

    correction_oof[
        val_idx
    ] = corr

    residual_fold = metrics(
        y[
            val_idx
        ],
        p_resid,
    )

    print(
        f"    residual GNN: "
        f"LL={residual_fold['log_loss']:.4f} | "
        f"ROC={residual_fold['roc_auc']:.4f} | "
        f"PR={residual_fold['pr_auc']:.4f}"
    )

    print(
        f"    delta LL="
        f"{residual_fold['log_loss'] - base_fold['log_loss']:+.4f}"
    )

    fold_rows.append(
        {
            "fold": fold,
            "n": len(val_idx),
            "xgb_log_loss":
                base_fold[
                    "log_loss"
                ],
            "residual_log_loss":
                residual_fold[
                    "log_loss"
                ],
            "delta_log_loss":
                (
                    residual_fold[
                        "log_loss"
                    ]
                    - base_fold[
                        "log_loss"
                    ]
                ),
            "xgb_roc_auc":
                base_fold[
                    "roc_auc"
                ],
            "residual_roc_auc":
                residual_fold[
                    "roc_auc"
                ],
            "xgb_pr_auc":
                base_fold[
                    "pr_auc"
                ],
            "residual_pr_auc":
                residual_fold[
                    "pr_auc"
                ],
            "mean_abs_correction":
                float(
                    np.mean(
                        np.abs(corr)
                    )
                ),
        }
    )


if (
    np.isnan(
        p_xgb_oof
    ).any()
    or np.isnan(
        p_residual_oof
    ).any()
):
    raise RuntimeError(
        "Missing OOF predictions."
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

xgb_metrics = metrics(
    y,
    p_xgb_oof,
)

residual_metrics = metrics(
    y,
    p_residual_oof,
)


print()
print("=" * 108)
print("FINAL OOF COMPARISON")
print("=" * 108)

print()
print(
    "Setup+incoming XGB      "
    f"LL={xgb_metrics['log_loss']:.4f} | "
    f"ROC={xgb_metrics['roc_auc']:.4f} | "
    f"PR={xgb_metrics['pr_auc']:.4f} | "
    f"Brier={xgb_metrics['brier']:.4f}"
)

print(
    "XGB + residual GNN      "
    f"LL={residual_metrics['log_loss']:.4f} | "
    f"ROC={residual_metrics['roc_auc']:.4f} | "
    f"PR={residual_metrics['pr_auc']:.4f} | "
    f"Brier={residual_metrics['brier']:.4f}"
)

print()
print(
    f"Delta log loss: "
    f"{residual_metrics['log_loss'] - xgb_metrics['log_loss']:+.4f}"
)

print(
    f"Mean |OOF graph correction|: "
    f"{np.mean(np.abs(correction_oof)):.4f} logits"
)


# ============================================================
# OPTIONAL BRANCH DIAGNOSTICS
# ============================================================

if is_clearance is not None:
    print()
    print("BY EVENTUAL FIRST-CONTACT BRANCH")

    for value, label in [
        (0, "Pass"),
        (1, "Clearance"),
    ]:
        mask = (
            is_clearance
            == value
        )

        base = metrics(
            y[mask],
            p_xgb_oof[mask],
        )

        residual = metrics(
            y[mask],
            p_residual_oof[mask],
        )

        print(
            f"  {label:<10} "
            f"n={int(mask.sum()):>4} | "
            f"XGB LL={base['log_loss']:.4f} | "
            f"Residual LL={residual['log_loss']:.4f} | "
            f"delta="
            f"{residual['log_loss'] - base['log_loss']:+.4f}"
        )


# ============================================================
# MATCH-CLUSTERED BOOTSTRAP
# ============================================================

by_match = defaultdict(list)

for i, match_id in enumerate(groups):
    by_match[
        match_id
    ].append(i)

match_ids = list(
    by_match.keys()
)

rng = np.random.default_rng(
    20260919
)

boot_rows = []

for b in range(N_BOOT):
    sampled = rng.choice(
        match_ids,
        size=len(match_ids),
        replace=True,
    )

    idx = np.concatenate(
        [
            np.asarray(
                by_match[m],
                dtype=int,
            )
            for m in sampled
        ]
    )

    yy = y[idx]
    px = p_xgb_oof[idx]
    pr = p_residual_oof[idx]

    boot_rows.append(
        {
            "iteration": b + 1,
            "delta_log_loss_residual_minus_xgb":
                (
                    log_loss(
                        yy,
                        pr,
                        labels=[0, 1],
                    )
                    -
                    log_loss(
                        yy,
                        px,
                        labels=[0, 1],
                    )
                ),
            "delta_brier_residual_minus_xgb":
                (
                    brier_score_loss(
                        yy,
                        pr,
                    )
                    -
                    brier_score_loss(
                        yy,
                        px,
                    )
                ),
            "delta_pr_residual_minus_xgb":
                (
                    safe_pr(
                        yy,
                        pr,
                    )
                    -
                    safe_pr(
                        yy,
                        px,
                    )
                ),
            "delta_roc_residual_minus_xgb":
                (
                    safe_auc(
                        yy,
                        pr,
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
print("=" * 108)
print("PAIRED MATCH-CLUSTERED BOOTSTRAP")
print("=" * 108)

for col, label, lower_better in [
    (
        "delta_log_loss_residual_minus_xgb",
        "Log loss",
        True,
    ),
    (
        "delta_brier_residual_minus_xgb",
        "Brier",
        True,
    ),
    (
        "delta_pr_residual_minus_xgb",
        "PR-AUC",
        False,
    ),
    (
        "delta_roc_residual_minus_xgb",
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
        [2.5, 97.5],
    )

    frac_better = (
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
        f"delta residual-XGB="
        f"{mean:+.4f} | "
        f"95% CI "
        f"[{lo:+.4f}, {hi:+.4f}] | "
        f"Residual better in "
        f"{100*frac_better:.1f}%"
    )


# ============================================================
# SAVE
# ============================================================

oof = pd.DataFrame(
    {
        "winner_event_id": event_ids,
        "match_id": groups,
        "target": y,
        "p_setup_incoming_xgb":
            p_xgb_oof,
        "p_xgb_plus_residual_gnn":
            p_residual_oof,
        "graph_logit_correction":
            correction_oof,
    }
)

if is_clearance is not None:
    oof[
        "is_clearance"
    ] = is_clearance

oof.to_csv(
    OOF_OUT,
    index=False,
)


pd.DataFrame(
    [
        {
            "model":
                "setup_plus_incoming_xgb",
            **xgb_metrics,
        },
        {
            "model":
                "xgb_plus_residual_gnn",
            **residual_metrics,
        },
    ]
).to_csv(
    RESULTS_OUT,
    index=False,
)

pd.DataFrame(
    fold_rows
).to_csv(
    FOLDS_OUT,
    index=False,
)


print()
print("Saved:")
print(f"  {OOF_OUT}")
print(f"  {RESULTS_OUT}")
print(f"  {FOLDS_OUT}")
print(f"  {BOOT_OUT}")

print()
print("=" * 108)
print("DONE")
print("=" * 108)
