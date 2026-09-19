import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVENTS_DIR = PROJECT_ROOT / "data" / "events"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CONTESTS_PATH = OUTPUT_DIR / "all_aerial_contests.csv"

OUTPUT_PATH = OUTPUT_DIR / "second_ball_label_validation.csv"

HORIZONS = [2, 3, 4, 5, 6, 8]
MAX_HORIZON = max(HORIZONS)
MAX_EVENTS = 20

CORE_WINNER_ACTIONS = {
    "Pass",
    "Clearance",
}


# ============================================================
# HELPERS
# ============================================================

def timestamp_to_seconds(timestamp):
    if not timestamp:
        return None

    try:
        hh, mm, ss = timestamp.split(":")
        return (
            int(hh) * 3600
            + int(mm) * 60
            + float(ss)
        )
    except Exception:
        return None


def event_time(event):
    ts = timestamp_to_seconds(
        event.get("timestamp")
    )

    if ts is not None:
        return ts

    return (
        float(event.get("minute", 0)) * 60
        + float(event.get("second", 0))
    )


def nested_name(event, section, key):
    return (
        event.get(section, {})
        .get(key, {})
        .get("name")
    )


def control_reason(event):
    """
    Return a string if this event demonstrates that the event team
    has established meaningful control of the ball.

    This is deliberately stricter than the previous broad audit.
    """

    event_type = (
        event.get("type", {})
        .get("name")
    )

    # A completed receipt is direct evidence that the player
    # controlled the incoming ball.
    if event_type == "Ball Receipt*":

        outcome = (
            event.get("ball_receipt", {})
            .get("outcome", {})
            .get("name")
        )

        if outcome != "Incomplete":
            return "Ball Receipt* (complete)"

        return None

    # These all require the player/team to possess the ball
    # at the start of the action.
    if event_type in {
        "Carry",
        "Pass",
        "Shot",
        "Dribble",
        "Dispossessed",
    }:
        return event_type

    if event_type == "Ball Recovery":

        recovery_failure = (
            event.get("ball_recovery", {})
            .get("recovery_failure", False)
        )

        if not recovery_failure:
            return "Ball Recovery"

        return None

    if event_type == "Interception":

        outcome = (
            event.get("interception", {})
            .get("outcome", {})
            .get("name")
        )

        if outcome in {
            "Won",
            "Success In Play",
        }:
            return f"Interception ({outcome})"

        return None

    if event_type == "Goal Keeper":

        keeper_type = nested_name(
            event,
            "goalkeeper",
            "type",
        )

        # These imply possession/control rather than simply
        # a touch such as a punch or parry.
        if keeper_type in {
            "Collected",
            "Keeper Pick Up",
            "Smother",
        }:
            return f"Goal Keeper ({keeper_type})"

        return None

    return None


def is_stoppage(event):
    """
    If play terminates before either side clearly controls the ball,
    mark the situation unresolved rather than forcing a label.
    """

    event_type = (
        event.get("type", {})
        .get("name")
    )

    return event_type in {
        "Half End",
        "Period End",
        "Injury Stoppage",
        "Substitution",
        "Foul Won",
        "Foul Committed",
        "Referee Ball-Drop",
    }


def relation(team, winner_team, loser_team):
    if team == winner_team:
        return "winner_team"

    if team == loser_team:
        return "loser_team"

    return "other_or_missing"


# ============================================================
# LOAD CONTESTS
# ============================================================

if not CONTESTS_PATH.exists():
    raise FileNotFoundError(
        f"Missing:\n{CONTESTS_PATH}\n\n"
        "Run build_contest_catalogue.py first."
    )


with open(
    CONTESTS_PATH,
    "r",
    encoding="utf-8",
) as f:
    contests = list(
        csv.DictReader(f)
    )


usable = [
    row
    for row in contests
    if row.get(
        "has_winner_360",
        "",
    ).lower() == "true"
]


contests_by_match = defaultdict(list)

for row in usable:
    contests_by_match[
        row["match_id"]
    ].append(row)


print("=" * 84)
print("SECOND-BALL LABEL VALIDATION")
print("=" * 84)
print()
print(
    f"Usable canonical-360 contests: "
    f"{len(usable):,}"
)
print()


# ============================================================
# ACCUMULATORS
# ============================================================

rows = []

all_event_types = Counter()

ball_receipt_outcomes = Counter()
ball_recovery_failure = Counter()
interception_outcomes = Counter()
goalkeeper_types = Counter()

winner_action_by_label_5s = defaultdict(
    Counter
)

horizon_counts = {
    h: Counter()
    for h in HORIZONS
}

horizon_counts_core = {
    h: Counter()
    for h in HORIZONS
}

stop_reasons = Counter()
control_reasons = Counter()


# ============================================================
# PROCESS
# ============================================================

for match_num, (match_id, match_contests) in enumerate(
    sorted(
        contests_by_match.items(),
        key=lambda x: int(x[0]),
    ),
    start=1,
):

    event_path = (
        EVENTS_DIR /
        f"{match_id}.json"
    )

    with open(
        event_path,
        "r",
        encoding="utf-8",
    ) as f:
        events = json.load(f)

    event_by_id = {
        event["id"]: event
        for event in events
    }

    position_by_id = {
        event["id"]: i
        for i, event in enumerate(events)
    }

    print(
        f"[{match_num:03d}/"
        f"{len(contests_by_match):03d}] "
        f"{match_id}: "
        f"{len(match_contests)} contests"
    )

    for contest in match_contests:

        winner = event_by_id[
            contest["winner_event_id"]
        ]

        winner_pos = position_by_id[
            contest["winner_event_id"]
        ]

        winner_team = (
            contest["winner_team"]
        )

        loser_team = (
            contest["loser_team"]
        )

        winner_action = (
            contest["winner_event_type"]
        )

        start_time = event_time(
            winner
        )

        period = winner.get(
            "period"
        )

        sequence = []

        for event in events[
            winner_pos + 1:
            winner_pos + 1 + MAX_EVENTS
        ]:

            if event.get("period") != period:
                break

            dt = (
                event_time(event)
                -
                start_time
            )

            if dt < -0.001:
                continue

            if dt > MAX_HORIZON:
                break

            event_type = (
                event.get("type", {})
                .get("name")
            )

            all_event_types[
                event_type
            ] += 1

            if event_type == "Ball Receipt*":

                outcome = (
                    event
                    .get("ball_receipt", {})
                    .get("outcome", {})
                    .get("name")
                )

                ball_receipt_outcomes[
                    outcome or "Complete / no outcome"
                ] += 1

            elif event_type == "Ball Recovery":

                failure = (
                    event
                    .get("ball_recovery", {})
                    .get(
                        "recovery_failure",
                        False,
                    )
                )

                ball_recovery_failure[
                    str(bool(failure))
                ] += 1

            elif event_type == "Interception":

                outcome = (
                    event
                    .get("interception", {})
                    .get("outcome", {})
                    .get("name")
                )

                interception_outcomes[
                    outcome or "None"
                ] += 1

            elif event_type == "Goal Keeper":

                keeper_type = nested_name(
                    event,
                    "goalkeeper",
                    "type",
                )

                goalkeeper_types[
                    keeper_type or "None"
                ] += 1

            sequence.append(
                (
                    dt,
                    event,
                )
            )

        # ----------------------------------------------------
        # LABEL EACH HORIZON
        # ----------------------------------------------------

        result_by_horizon = {}

        for horizon in HORIZONS:

            label = "unresolved"
            reason = None
            label_dt = None

            for dt, event in sequence:

                if dt > horizon:
                    break

                control = control_reason(
                    event
                )

                if control is not None:

                    team = (
                        event.get("team", {})
                        .get("name")
                    )

                    label = relation(
                        team,
                        winner_team,
                        loser_team,
                    )

                    reason = control
                    label_dt = dt

                    break

                if is_stoppage(event):

                    label = "censored_stoppage"

                    reason = (
                        event.get(
                            "type",
                            {},
                        ).get("name")
                    )

                    label_dt = dt

                    break

            result_by_horizon[
                horizon
            ] = (
                label,
                reason,
                label_dt,
            )

            horizon_counts[
                horizon
            ][label] += 1

            if winner_action in CORE_WINNER_ACTIONS:

                horizon_counts_core[
                    horizon
                ][label] += 1

        label_5, reason_5, dt_5 = (
            result_by_horizon[5]
        )

        winner_action_by_label_5s[
            winner_action
        ][label_5] += 1

        if label_5 == "censored_stoppage":
            stop_reasons[
                reason_5
            ] += 1

        if label_5 in {
            "winner_team",
            "loser_team",
        }:
            control_reasons[
                reason_5
            ] += 1

        rows.append(
            {
                "match_id": match_id,
                "competition": contest["competition"],
                "season": contest["season"],
                "winner_event_id": contest["winner_event_id"],
                "winner_team": winner_team,
                "loser_team": loser_team,
                "winner_event_type": winner_action,
                "period": contest["period"],
                "minute": contest["minute"],
                "second": contest["second"],

                "label_2s": result_by_horizon[2][0],
                "label_3s": result_by_horizon[3][0],
                "label_4s": result_by_horizon[4][0],
                "label_5s": result_by_horizon[5][0],
                "label_6s": result_by_horizon[6][0],
                "label_8s": result_by_horizon[8][0],

                "reason_5s": reason_5,
                "control_dt_5s": dt_5,
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
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_horizon_table(
    title,
    counts_by_horizon,
):

    print()
    print(title)

    header = (
        f"{'Horizon':>8} | "
        f"{'Winner':>8} | "
        f"{'Loser':>8} | "
        f"{'Censored':>8} | "
        f"{'Unresolved':>10} | "
        f"{'Resolved %':>10}"
    )

    print(header)
    print("-" * len(header))

    for h in HORIZONS:

        counts = (
            counts_by_horizon[h]
        )

        winner_n = counts[
            "winner_team"
        ]

        loser_n = counts[
            "loser_team"
        ]

        censored_n = counts[
            "censored_stoppage"
        ]

        unresolved_n = counts[
            "unresolved"
        ]

        total = sum(
            counts.values()
        )

        resolved = (
            winner_n
            +
            loser_n
        )

        resolved_pct = (
            100 * resolved / total
            if total
            else 0
        )

        print(
            f"{h:>7}s | "
            f"{winner_n:>8,} | "
            f"{loser_n:>8,} | "
            f"{censored_n:>8,} | "
            f"{unresolved_n:>10,} | "
            f"{resolved_pct:>9.2f}%"
        )


print()
print("=" * 84)
print("LABEL VALIDATION SUMMARY")
print("=" * 84)

print_horizon_table(
    "ALL WINNER ACTION TYPES",
    horizon_counts,
)

print_horizon_table(
    "CORE SAMPLE ONLY: PASS + CLEARANCE",
    horizon_counts_core,
)


print()
print("5-second label by aerial-winner action:")

for action in sorted(
    winner_action_by_label_5s
):

    counts = (
        winner_action_by_label_5s[
            action
        ]
    )

    total = sum(
        counts.values()
    )

    print()
    print(
        f"  {action} "
        f"(n={total:,})"
    )

    for label in [
        "winner_team",
        "loser_team",
        "censored_stoppage",
        "unresolved",
    ]:

        value = counts[
            label
        ]

        pct = (
            100 * value / total
            if total
            else 0
        )

        print(
            f"    {label:<20}"
            f"{value:>6,} "
            f"({pct:6.2f}%)"
        )


print()
print("Control reasons at 5 seconds:")

for reason, count in (
    control_reasons.most_common()
):

    print(
        f"  {str(reason):<34}"
        f"{count:>7,}"
    )


print()
print("Censoring reasons at 5 seconds:")

if stop_reasons:

    for reason, count in (
        stop_reasons.most_common()
    ):

        print(
            f"  {str(reason):<34}"
            f"{count:>7,}"
        )

else:
    print("  None")


print()
print("Ball Receipt* outcomes observed in post-contest windows:")

for outcome, count in (
    ball_receipt_outcomes.most_common()
):

    print(
        f"  {str(outcome):<34}"
        f"{count:>7,}"
    )


print()
print("Ball Recovery failure flag:")

for value, count in (
    ball_recovery_failure.most_common()
):

    print(
        f"  {value:<34}"
        f"{count:>7,}"
    )


print()
print("Interception outcomes:")

for outcome, count in (
    interception_outcomes.most_common()
):

    print(
        f"  {str(outcome):<34}"
        f"{count:>7,}"
    )


print()
print("Goal Keeper subtypes:")

for subtype, count in (
    goalkeeper_types.most_common()
):

    print(
        f"  {str(subtype):<34}"
        f"{count:>7,}"
    )


# Label stability relative to 8 seconds.
print()
print("Agreement with final 8-second resolved label:")

resolved_8 = [
    row
    for row in rows
    if row["label_8s"] in {
        "winner_team",
        "loser_team",
    }
]

for h in [2, 3, 4, 5, 6]:

    key = f"label_{h}s"

    already_resolved = [
        row
        for row in resolved_8
        if row[key] in {
            "winner_team",
            "loser_team",
        }
    ]

    correct = sum(
        row[key] == row["label_8s"]
        for row in already_resolved
    )

    coverage = (
        100
        * len(already_resolved)
        / len(resolved_8)
        if resolved_8
        else 0
    )

    agreement = (
        100
        * correct
        / len(already_resolved)
        if already_resolved
        else 0
    )

    print(
        f"  {h}s: "
        f"{coverage:6.2f}% of 8s-resolved "
        f"already resolved; "
        f"{agreement:6.2f}% agreement"
    )


print()
print("Saved:")
print(f"  {OUTPUT_PATH}")

print()
print("=" * 84)
print("DONE")
print("=" * 84)

