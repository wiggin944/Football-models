import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVENT_DIR = PROJECT_ROOT / "data" / "events"

FEATURE_PATH = OUTPUT_DIR / "handcrafted_features.csv"
LANDING_OOF_PATH = OUTPUT_DIR / "landing_zone_pass_oof_predictions.csv"

AUDIT_OUTPUT = OUTPUT_DIR / "true_second_ball_definition_audit.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "true_second_ball_definition_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def event_time_seconds(event):
    return (
        float(event.get("minute", 0) or 0) * 60.0
        + float(event.get("second", 0) or 0)
    )


def dist(a, b):
    return math.hypot(
        float(a[0]) - float(b[0]),
        float(a[1]) - float(b[1]),
    )


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    if len(y) == 0:
        return None

    return {
        "n": len(y),
        "control_rate": float(y.mean()),
        "log_loss": log_loss(
            y,
            p,
            labels=[0, 1],
        ),
        "auc": (
            roc_auc_score(y, p)
            if len(np.unique(y)) == 2
            else np.nan
        ),
        "brier": brier_score_loss(
            y,
            p,
        ),
    }


# ============================================================
# LOAD MODEL DATA
# ============================================================

with open(
    FEATURE_PATH,
    "r",
    encoding="utf-8",
) as f:
    feature_rows = list(
        csv.DictReader(f)
    )


pass_rows = [
    row
    for row in feature_rows
    if row[
        "winner_event_type"
    ] == "Pass"
]


with open(
    LANDING_OOF_PATH,
    "r",
    encoding="utf-8",
) as f:
    prediction_rows = list(
        csv.DictReader(f)
    )


pred_by_event = {
    row[
        "winner_event_id"
    ]: row
    for row in prediction_rows
}


# ============================================================
# LOAD EVENTS
# ============================================================

match_ids = sorted(
    set(
        str(
            row["match_id"]
        )
        for row in pass_rows
    )
)


events_by_match = {}
index_by_match = {}


for match_id in match_ids:

    path = (
        EVENT_DIR
        / f"{match_id}.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        events = json.load(f)

    events_by_match[
        match_id
    ] = events

    index_by_match[
        match_id
    ] = {
        event["id"]: i
        for i, event in enumerate(events)
    }


# ============================================================
# AUDIT EACH AERIAL PASS
# ============================================================

audit_rows = []


for row in pass_rows:

    match_id = str(
        row["match_id"]
    )

    event_id = row[
        "winner_event_id"
    ]

    events = events_by_match[
        match_id
    ]

    idx = index_by_match[
        match_id
    ][
        event_id
    ]

    event = events[idx]

    pass_data = event.get(
        "pass",
        {},
    )

    outcome_obj = pass_data.get(
        "outcome"
    )

    if isinstance(
        outcome_obj,
        dict,
    ):
        pass_outcome = outcome_obj.get(
            "name",
            "Unknown",
        )
    else:
        # StatsBomb convention: missing pass outcome = completed.
        pass_outcome = "Complete"

    end_location = pass_data.get(
        "end_location"
    )

    event_team = (
        event.get(
            "team",
            {},
        ).get(
            "id"
        )
    )

    event_time = event_time_seconds(
        event
    )

    direct_receipt = 0
    direct_receipt_dt = np.nan
    direct_receipt_error = np.nan

    next_nonreceipt_type = ""
    next_nonreceipt_team_same = np.nan

    # Search a very short window after the header for a same-team
    # receipt near the recorded pass endpoint.
    for next_event in events[
        idx + 1:
        min(
            len(events),
            idx + 12,
        )
    ]:

        dt = (
            event_time_seconds(
                next_event
            )
            - event_time
        )

        if dt < -0.01:
            continue

        if dt > 3.0:
            break

        event_type = (
            next_event
            .get(
                "type",
                {},
            )
            .get(
                "name",
                "",
            )
        )

        next_team = (
            next_event
            .get(
                "team",
                {},
            )
            .get(
                "id"
            )
        )

        if (
            next_nonreceipt_type == ""
            and event_type
            != "Ball Receipt*"
        ):
            next_nonreceipt_type = (
                event_type
            )

            next_nonreceipt_team_same = (
                1.0
                if (
                    next_team is not None
                    and event_team is not None
                    and next_team
                    == event_team
                )
                else 0.0
            )

        if (
            event_type
            == "Ball Receipt*"
            and next_team is not None
            and event_team is not None
            and next_team == event_team
        ):
            receipt_location = (
                next_event.get(
                    "location"
                )
            )

            endpoint_error = np.nan

            if (
                end_location
                and receipt_location
            ):
                endpoint_error = dist(
                    end_location,
                    receipt_location,
                )

            if (
                np.isnan(
                    endpoint_error
                )
                or endpoint_error
                <= 7.5
            ):
                direct_receipt = 1
                direct_receipt_dt = dt
                direct_receipt_error = (
                    endpoint_error
                )
                break

    target = int(
        row["target"]
    )

    pred = pred_by_event.get(
        event_id
    )

    p_base = (
        float(
            pred[
                "p_current_pass_geometry"
            ]
        )
        if pred
        else np.nan
    )

    p_landing = (
        float(
            pred[
                "p_landing_zone_geometry"
            ]
        )
        if pred
        else np.nan
    )

    # Conservative football interpretation:
    # a completed aerial pass is direct retained possession rather
    # than a loose second-ball episode.
    definite_direct_control = (
        1
        if pass_outcome == "Complete"
        else 0
    )

    # Stronger observable version: completed pass + immediate
    # same-team receipt close to its endpoint.
    confirmed_direct_receipt = (
        1
        if (
            pass_outcome == "Complete"
            and direct_receipt == 1
        )
        else 0
    )

    loose_candidate = (
        1
        if pass_outcome
        != "Complete"
        else 0
    )

    audit_rows.append(
        {
            "match_id": match_id,
            "winner_event_id": event_id,
            "target": target,
            "pass_outcome": pass_outcome,
            "direct_receipt_within_3s": (
                direct_receipt
            ),
            "direct_receipt_dt": (
                direct_receipt_dt
            ),
            "direct_receipt_endpoint_error": (
                direct_receipt_error
            ),
            "definite_direct_control": (
                definite_direct_control
            ),
            "confirmed_direct_receipt": (
                confirmed_direct_receipt
            ),
            "loose_candidate": (
                loose_candidate
            ),
            "next_nonreceipt_type": (
                next_nonreceipt_type
            ),
            "next_nonreceipt_team_same": (
                next_nonreceipt_team_same
            ),
            "p_current_pass_geometry": (
                p_base
            ),
            "p_landing_zone_geometry": (
                p_landing
            ),
        }
    )


# ============================================================
# SAVE DETAIL
# ============================================================

with open(
    AUDIT_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=audit_rows[
            0
        ].keys(),
    )

    writer.writeheader()
    writer.writerows(
        audit_rows
    )


# ============================================================
# SUMMARISE OUTCOMES
# ============================================================

outcome_counts = Counter(
    row["pass_outcome"]
    for row in audit_rows
)


print("=" * 104)
print("TRUE SECOND-BALL DEFINITION AUDIT")
print("=" * 104)

print()
print(
    f"Primary aerial-pass contests: "
    f"{len(audit_rows):,}"
)

print()
print("PASS OUTCOMES")

for outcome, n in (
    outcome_counts
    .most_common()
):
    subset = [
        row
        for row in audit_rows
        if row[
            "pass_outcome"
        ] == outcome
    ]

    control_rate = np.mean(
        [
            row["target"]
            for row in subset
        ]
    )

    receipt_rate = np.mean(
        [
            row[
                "direct_receipt_within_3s"
            ]
            for row in subset
        ]
    )

    print(
        f"  {outcome:<24}"
        f"n={n:>4} | "
        f"winner-control={100*control_rate:5.1f}% | "
        f"same-team receipt<=3s="
        f"{100*receipt_rate:5.1f}%"
    )


complete_rows = [
    row
    for row in audit_rows
    if row[
        "pass_outcome"
    ] == "Complete"
]

loose_rows = [
    row
    for row in audit_rows
    if row[
        "pass_outcome"
    ] != "Complete"
]

confirmed_rows = [
    row
    for row in audit_rows
    if row[
        "confirmed_direct_receipt"
    ] == 1
]


print()
print("=" * 104)
print("FOOTBALL PURITY CHECK")
print("=" * 104)

print()
print(
    f"Completed aerial passes: "
    f"{len(complete_rows):,}/"
    f"{len(audit_rows):,} "
    f"({100*len(complete_rows)/len(audit_rows):.1f}%)"
)

print(
    f"Completed + immediate same-team receipt: "
    f"{len(confirmed_rows):,}/"
    f"{len(audit_rows):,} "
    f"({100*len(confirmed_rows)/len(audit_rows):.1f}%)"
)

print(
    f"Non-complete aerial passes "
    f"(loose-ball candidates): "
    f"{len(loose_rows):,}/"
    f"{len(audit_rows):,} "
    f"({100*len(loose_rows)/len(audit_rows):.1f}%)"
)


# ============================================================
# MODEL PERFORMANCE BY PURITY GROUP
# ============================================================

print()
print("=" * 104)
print("LANDING-MODEL PERFORMANCE BY PASS STATUS")
print("=" * 104)


summary_rows = []


groups = [
    (
        "all_passes",
        audit_rows,
    ),
    (
        "completed_passes",
        complete_rows,
    ),
    (
        "confirmed_direct_receipt",
        confirmed_rows,
    ),
    (
        "noncomplete_loose_candidates",
        loose_rows,
    ),
]


for name, subset in groups:

    usable = [
        row
        for row in subset
        if (
            not np.isnan(
                row[
                    "p_current_pass_geometry"
                ]
            )
            and not np.isnan(
                row[
                    "p_landing_zone_geometry"
                ]
            )
        )
    ]

    if not usable:
        continue

    y = [
        row["target"]
        for row in usable
    ]

    p_base = [
        row[
            "p_current_pass_geometry"
        ]
        for row in usable
    ]

    p_landing = [
        row[
            "p_landing_zone_geometry"
        ]
        for row in usable
    ]

    base = metrics(
        y,
        p_base,
    )

    landing = metrics(
        y,
        p_landing,
    )

    print()
    print(
        f"{name} "
        f"(n={len(usable):,})"
    )

    print(
        f"  Current  "
        f"LL={base['log_loss']:.4f} | "
        f"AUC={base['auc']:.4f} | "
        f"Brier={base['brier']:.4f}"
    )

    print(
        f"  Landing  "
        f"LL={landing['log_loss']:.4f} | "
        f"AUC={landing['auc']:.4f} | "
        f"Brier={landing['brier']:.4f}"
    )

    summary_rows.append(
        {
            "subset": name,
            "n": len(
                usable
            ),
            "winner_control_rate": float(
                np.mean(
                    y
                )
            ),
            "current_log_loss": (
                base[
                    "log_loss"
                ]
            ),
            "landing_log_loss": (
                landing[
                    "log_loss"
                ]
            ),
            "current_auc": (
                base["auc"]
            ),
            "landing_auc": (
                landing["auc"]
            ),
            "landing_minus_current_ll": (
                landing[
                    "log_loss"
                ]
                -
                base[
                    "log_loss"
                ]
            ),
        }
    )


# ============================================================
# PROVISIONAL STRICT SAMPLE SIZE
# ============================================================

# Combine:
#   - every primary Clearance
#   - only non-complete aerial Passes
#
# This is NOT yet declared the final sample. It is just the
# size/class-balance audit for a more football-faithful definition.

clearance_rows = [
    row
    for row in feature_rows
    if row[
        "winner_event_type"
    ] == "Clearance"
]


strict_targets = [
    int(
        row["target"]
    )
    for row in clearance_rows
]

strict_targets.extend(
    row["target"]
    for row in loose_rows
)


print()
print("=" * 104)
print("PROVISIONAL TRUE-SECOND-BALL SAMPLE")
print("=" * 104)

print()
print(
    f"Clearances:                "
    f"{len(clearance_rows):,}"
)

print(
    f"Non-complete aerial passes:"
    f" {len(loose_rows):,}"
)

print(
    f"Combined candidate sample: "
    f"{len(strict_targets):,}"
)

print(
    f"Winner-team control rate:  "
    f"{100*np.mean(strict_targets):.1f}%"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_rows[
            0
        ].keys(),
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


print()
print("Saved:")
print(f"  {AUDIT_OUTPUT}")
print(f"  {SUMMARY_OUTPUT}")

print()
print("=" * 104)
print("DONE")
print("=" * 104)
