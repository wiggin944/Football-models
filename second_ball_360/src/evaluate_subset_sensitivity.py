import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

BASELINE_PREDICTIONS = OUTPUT_DIR / "baseline_test_predictions.csv"
GNN_PREDICTIONS = OUTPUT_DIR / "gnn_test_predictions.csv"

OUTPUT_PATH = OUTPUT_DIR / "final_subset_sensitivity.csv"


def metric_bundle(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1 - 1e-6,
    )

    return {
        "n": len(y),
        "log_loss": log_loss(
            y,
            p,
            labels=[0, 1],
        ),
        "brier": brier_score_loss(
            y,
            p,
        ),
        "auc": roc_auc_score(
            y,
            p,
        ),
    }


with open(
    BASELINE_PREDICTIONS,
    "r",
    encoding="utf-8",
) as f:
    baseline_rows = list(csv.DictReader(f))


with open(
    GNN_PREDICTIONS,
    "r",
    encoding="utf-8",
) as f:
    gnn_rows = list(csv.DictReader(f))


gnn_by_event = {
    row["winner_event_id"]: row
    for row in gnn_rows
}


rows = []

for base in baseline_rows:

    gnn = gnn_by_event.get(
        base["winner_event_id"]
    )

    if gnn is None:
        continue

    rows.append(
        {
            "match_id": base["match_id"],
            "winner_event_id": base["winner_event_id"],
            "winner_event_type": base["winner_event_type"],
            "target": int(base["target"]),
            "is_proxy": int(
                base["loser_source_proxy"]
            ),

            "prior": float(
                base["p_prior"]
            ),

            "event_only": float(
                base["p_event-only_logistic"]
            ),

            "counts_logistic": float(
                base[
                    "p_event_plus_360_counts_logistic"
                ]
            ),

            "geometry_logistic": float(
                base[
                    "p_full_geometry_logistic"
                ]
            ),

            "xgboost": float(
                base[
                    "p_xgboost_full_geometry"
                ]
            ),

            "gnn": float(
                gnn[
                    "p_gnn_calibrated"
                ]
            ),
        }
    )


subsets = {
    "All test": lambda r: True,
    "Exact loser only": lambda r: r["is_proxy"] == 0,
    "Proxy loser only": lambda r: r["is_proxy"] == 1,
    "Pass only": lambda r: r["winner_event_type"] == "Pass",
    "Clearance only": lambda r: r["winner_event_type"] == "Clearance",
}


models = [
    "prior",
    "event_only",
    "counts_logistic",
    "geometry_logistic",
    "xgboost",
    "gnn",
]


output_rows = []


print("=" * 92)
print("FINAL SUBSET SENSITIVITY")
print("=" * 92)


for subset_name, predicate in subsets.items():

    subset = [
        row
        for row in rows
        if predicate(row)
    ]

    y = [
        row["target"]
        for row in subset
    ]

    print()
    print(
        f"{subset_name} "
        f"(n={len(subset):,})"
    )

    for model in models:

        p = [
            row[model]
            for row in subset
        ]

        result = metric_bundle(
            y,
            p,
        )

        output_rows.append(
            {
                "subset": subset_name,
                "model": model,
                **result,
            }
        )

        print(
            f"  {model:<20}"
            f"LL={result['log_loss']:.4f} | "
            f"AUC={result['auc']:.4f} | "
            f"Brier={result['brier']:.4f}"
        )


with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=output_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(output_rows)


print()
print("Saved:")
print(f"  {OUTPUT_PATH}")

print()
print("=" * 92)
print("DONE")
print("=" * 92)
