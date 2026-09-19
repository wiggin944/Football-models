import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures_polished"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BOOTSTRAP_PATH = OUTPUT_DIR / "model_bootstrap_results.csv"
HERO_PATH = OUTPUT_DIR / "hero_examples.csv"
GRAPH_PATH = OUTPUT_DIR / "final_graph_dataset_primary.jsonl"


# ============================================================
# HELPERS
# ============================================================

def read_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def save(fig, filename):
    path = FIGURE_DIR / filename

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# FIGURE 1 — MODEL PERFORMANCE WITH MATCH-BOOTSTRAP CIs
# ============================================================

bootstrap_rows = read_csv(
    BOOTSTRAP_PATH
)

model_order = [
    "prior",
    "event_only",
    "counts_logistic",
    "geometry_logistic",
    "xgboost",
    "gnn",
]

friendly = {
    "prior": "Prior",
    "event_only": "Event only",
    "counts_logistic": "Event + 360 counts",
    "geometry_logistic": "Full geometry",
    "xgboost": "XGBoost geometry",
    "gnn": "Relational GNN",
}

ll_rows = {
    row["model"]: row
    for row in bootstrap_rows
    if row["metric"] == "log_loss"
}

point = np.asarray(
    [
        float(
            ll_rows[m][
                "point_estimate"
            ]
        )
        for m in model_order
    ],
    dtype=float,
)

lower = np.asarray(
    [
        float(
            ll_rows[m][
                "ci_2_5"
            ]
        )
        for m in model_order
    ],
    dtype=float,
)

upper = np.asarray(
    [
        float(
            ll_rows[m][
                "ci_97_5"
            ]
        )
        for m in model_order
    ],
    dtype=float,
)

yerr = np.vstack(
    [
        point - lower,
        upper - point,
    ]
)

x = np.arange(
    len(model_order)
)

fig, ax = plt.subplots(
    figsize=(10, 5.6)
)

ax.errorbar(
    x,
    point,
    yerr=yerr,
    fmt="o",
    capsize=5,
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        friendly[m]
        for m in model_order
    ],
    rotation=22,
    ha="right",
)

ax.set_ylabel(
    "Held-out test log loss"
)

ax.set_title(
    "Model progression with 95% match-bootstrap intervals"
)

for i, value in enumerate(
    point
):
    ax.annotate(
        f"{value:.3f}",
        (
            i,
            value,
        ),
        xytext=(
            0,
            -18,
        ),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

save(
    fig,
    "01_model_progression_bootstrap.png",
)


# ============================================================
# LOAD HERO DATA
# ============================================================

hero_rows = read_csv(
    HERO_PATH
)

graphs = {}

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

        graphs[
            graph[
                "winner_event_id"
            ]
        ] = graph


role_markers = {
    "winner_support": "o",
    "loser_support": "s",
    "aerial_winner": "*",
    "aerial_loser": "X",
}


# ============================================================
# CROPPED HERO PLOTS
# ============================================================

for idx, hero in enumerate(
    hero_rows,
    start=1,
):

    event_id = hero[
        "winner_event_id"
    ]

    graph = graphs[
        event_id
    ]

    cx, cy = graph[
        "contest_location"
    ]

    # Local viewing window around the aerial contest.
    x_pad = 24
    y_pad = 20

    xmin = max(
        0,
        cx - x_pad,
    )

    xmax = min(
        120,
        cx + x_pad,
    )

    ymin = max(
        0,
        cy - y_pad,
    )

    ymax = min(
        80,
        cy + y_pad,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 6.5)
    )

    # Pitch boundaries where visible in crop.
    ax.plot(
        [0, 120, 120, 0, 0],
        [0, 0, 80, 80, 0],
        linewidth=1.0,
    )

    ax.axvline(
        60,
        linewidth=0.8,
    )

    circle = plt.Circle(
        (
            cx,
            cy,
        ),
        15,
        fill=False,
        linestyle="--",
        linewidth=1.2,
    )

    ax.add_patch(
        circle
    )

    for role in [
        "winner_support",
        "loser_support",
        "aerial_winner",
        "aerial_loser",
    ]:

        nodes = [
            node
            for node in graph[
                "nodes"
            ]
            if node[
                "role"
            ] == role
        ]

        if not nodes:
            continue

        ax.scatter(
            [
                node["x"]
                for node in nodes
            ],
            [
                node["y"]
                for node in nodes
            ],
            marker=role_markers[
                role
            ],
            s=(
                150
                if role in {
                    "aerial_winner",
                    "aerial_loser",
                }
                else 75
            ),
            label=role.replace(
                "_",
                " ",
            ).title(),
            zorder=3,
        )

    ax.scatter(
        [cx],
        [cy],
        marker="+",
        s=150,
        linewidths=2,
        label="Contest location",
        zorder=4,
    )

    ax.set_xlim(
        xmin,
        xmax,
    )

    # StatsBomb y runs top to bottom.
    ax.set_ylim(
        ymax,
        ymin,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    actual = int(
        hero[
            "target_winner_controls"
        ]
    )

    actual_text = (
        "winner controlled"
        if actual == 1
        else "opponent controlled"
    )

    category = hero[
        "hero_category"
    ].replace(
        "_",
        " ",
    ).title()

    ax.set_title(
        (
            f"{category} — "
            f"{hero['winner_event_type']}\n"
            f"XGBoost {float(hero['p_xgboost']):.2f} | "
            f"GNN {float(hero['p_gnn']):.2f} | "
            f"Actual: {actual_text}"
        )
    )

    ax.set_xlabel(
        "Pitch x"
    )

    ax.set_ylabel(
        "Pitch y"
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(
            1.02,
            1.0,
        ),
        borderaxespad=0,
        fontsize=8,
    )

    save(
        fig,
        (
            f"hero_{idx:02d}_"
            f"{hero['hero_category']}_cropped.png"
        ),
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 96)
print("POLISHED VISUALS COMPLETE")
print("=" * 96)

print()
print(
    f"Output folder: {FIGURE_DIR}"
)

print(
    f"Bootstrap model figure: 1"
)

print(
    f"Cropped hero figures:   {len(hero_rows)}"
)

print()
print(
    "No model results were changed; "
    "this script only changes presentation."
)

print()
print("=" * 96)
print("DONE")
print("=" * 96)
