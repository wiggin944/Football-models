import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVENTS_DIR = PROJECT_ROOT / "data" / "events"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CONTESTS_PATH = OUTPUT_DIR / "all_aerial_contests.csv"

AUDIT_OUTPUT = OUTPUT_DIR / "second_ball_sequence_audit.csv"
EXAMPLES_OUTPUT = OUTPUT_DIR / "second_ball_sequence_examples.txt"

# We are auditing, not yet fixing the final label definition.
MAX_SECONDS = 8.0
MAX_EVENTS = 15

# Deliberately broad "controlled action" candidates.
# We will inspect the empirical behaviour before locking these.
CONTROL_TYPES = {
    "Pass",
    "Carry",
    "Dribble",
    "Shot",
    "Ball Recovery",
    "Interception",
    "Goal Keeper",
}


# ============================================================
# HELPERS
# ============================================================

def timestamp_to_seconds(timestamp):
    """
    Parse StatsBomb HH:MM:SS.mmm timestamps into seconds.
    """
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


def event_outcome(event):
    """
    Return the first nested .outcome.name we can find.
    Useful only for audit display.
    """
    for value in event.values():

        if not isinstance(value, dict):
            continue

        outcome = value.get("outcome")

        if isinstance(outcome, dict):
            name = outcome.get("name")

            if name:
                return name

    return None


def event_subtype(event):
    """
    Small human-readable subtype for sequence inspection.
    """
    event_type = event.get("type", {}).get("name")

    if event_type == "Pass":
        return (
            event.get("pass", {})
            .get("type", {})
            .get("name")
            or event.get("pass", {})
            .get("height", {})
            .get("name")
        )

    if event_type == "Duel":
        return (
            event.get("duel", {})
            .get("type", {})
            .get("name")
        )

    if event_type == "Shot":
        return (
            event.get("shot", {})
            .get("outcome", {})
            .get("name")
        )

    if event_type == "Goal Keeper":
        return (
            event.get("goalkeeper", {})
            .get("type", {})
            .get("name")
        )

    return None


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
            + values[n // 2]
        ) / 2

    return {
        "min": values[0],
        "mean": mean(values),
        "median": med,
        "max": values[-1],
    }


# ============================================================
# LOAD CONTEST CATALOGUE
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
    contests = list(csv.DictReader(f))

# Only rows with canonical winner-side 360.
usable_contests = [
    row
    for row in contests
    if row.get("has_winner_360", "").lower() == "true"
]


print("=" * 82)
print("SECOND-BALL SEQUENCE AUDIT")
print("=" * 82)
print()
print(f"All paired contests:          {len(contests):,}")
print(f"Contests with canonical 360:  {len(usable_contests):,}")
print()


# ============================================================
# GROUP CONTESTS BY MATCH
# ============================================================

contests_by_match = defaultdict(list)

for row in usable_contests:
    contests_by_match[row["match_id"]].append(row)


# ============================================================
# AUDIT EVERY USABLE CONTEST
# ============================================================

audit_rows = []

first_next_type_counts = Counter()
first_next_team_relation_counts = Counter()
first_control_type_counts = Counter()
first_control_relation_counts = Counter()
winner_action_counts = Counter()
no_control_counts = Counter()

control_delays = []

examples = []


for match_num, (match_id, match_contests) in enumerate(
    sorted(contests_by_match.items(), key=lambda x: int(x[0])),
    start=1,
):

    event_path = EVENTS_DIR / f"{match_id}.json"

    if not event_path.exists():
        print(
            f"[{match_num:03d}/{len(contests_by_match):03d}] "
            f"{match_id}: missing events file"
        )
        continue

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
        event["id"]: pos
        for pos, event in enumerate(events)
    }

    print(
        f"[{match_num:03d}/{len(contests_by_match):03d}] "
        f"{match_id}: {len(match_contests)} contests"
    )

    for contest in match_contests:

        winner_event_id = contest["winner_event_id"]

        winner = event_by_id.get(winner_event_id)

        if winner is None:
            continue

        winner_pos = position_by_id[winner_event_id]

        winner_team = contest["winner_team"]
        loser_team = contest["loser_team"]
        winner_action = contest["winner_event_type"]

        winner_action_counts[winner_action] += 1

        period = winner.get("period")

        winner_time = timestamp_to_seconds(
            winner.get("timestamp")
        )

        if winner_time is None:
            winner_time = (
                float(winner.get("minute", 0)) * 60
                + float(winner.get("second", 0))
            )

        subsequent = []

        for event in events[
            winner_pos + 1:
            winner_pos + 1 + MAX_EVENTS
        ]:

            if event.get("period") != period:
                break

            event_time = timestamp_to_seconds(
                event.get("timestamp")
            )

            if event_time is None:
                event_time = (
                    float(event.get("minute", 0)) * 60
                    + float(event.get("second", 0))
                )

            dt = event_time - winner_time

            if dt < -0.001:
                continue

            if dt > MAX_SECONDS:
                break

            subsequent.append(
                {
                    "event": event,
                    "dt": dt,
                }
            )

        # ----------------------------------------------------
        # IMMEDIATE NEXT EVENT
        # ----------------------------------------------------

        if subsequent:

            next_event = subsequent[0]["event"]
            next_dt = subsequent[0]["dt"]

            next_type = next_event.get(
                "type", {}
            ).get("name")

            next_team = next_event.get(
                "team", {}
            ).get("name")

            next_possession_team = (
                next_event
                .get("possession_team", {})
                .get("name")
            )

            next_possession = next_event.get(
                "possession"
            )

            first_next_type_counts[next_type] += 1

            if next_team == winner_team:
                next_relation = "winner_team"
            elif next_team == loser_team:
                next_relation = "loser_team"
            else:
                next_relation = "other_or_missing"

            first_next_team_relation_counts[
                next_relation
            ] += 1

        else:

            next_event = None
            next_dt = None
            next_type = None
            next_team = None
            next_possession_team = None
            next_possession = None
            next_relation = "none"

            first_next_team_relation_counts["none"] += 1

        # ----------------------------------------------------
        # FIRST CONTROL CANDIDATE
        # ----------------------------------------------------

        first_control = None

        for item in subsequent:

            event = item["event"]

            event_type = (
                event.get("type", {})
                .get("name")
            )

            if event_type in CONTROL_TYPES:

                first_control = item
                break

        if first_control is not None:

            control_event = first_control["event"]
            control_dt = first_control["dt"]

            control_type = (
                control_event.get("type", {})
                .get("name")
            )

            control_team = (
                control_event.get("team", {})
                .get("name")
            )

            control_possession_team = (
                control_event
                .get("possession_team", {})
                .get("name")
            )

            control_possession = control_event.get(
                "possession"
            )

            if control_team == winner_team:
                control_relation = "winner_team"
            elif control_team == loser_team:
                control_relation = "loser_team"
            else:
                control_relation = "other_or_missing"

            first_control_type_counts[
                control_type
            ] += 1

            first_control_relation_counts[
                control_relation
            ] += 1

            control_delays.append(
                control_dt
            )

        else:

            control_event = None
            control_dt = None
            control_type = None
            control_team = None
            control_possession_team = None
            control_possession = None
            control_relation = "no_control_in_window"

            first_control_relation_counts[
                control_relation
            ] += 1

            no_control_counts[
                winner_action
            ] += 1

        # ----------------------------------------------------
        # COMPACT SEQUENCE STRING
        # ----------------------------------------------------

        sequence_parts = []

        for item in subsequent:

            event = item["event"]
            dt = item["dt"]

            event_type = (
                event.get("type", {})
                .get("name")
            )

            team = (
                event.get("team", {})
                .get("name")
            )

            subtype = event_subtype(
                event
            )

            outcome = event_outcome(
                event
            )

            relation = (
                "W"
                if team == winner_team
                else "L"
                if team == loser_team
                else "?"
            )

            label = (
                f"+{dt:.2f}s "
                f"{relation}:"
                f"{event_type}"
            )

            if subtype:
                label += f"[{subtype}]"

            if outcome and outcome != subtype:
                label += f"({outcome})"

            sequence_parts.append(
                label
            )

        sequence_string = " -> ".join(
            sequence_parts
        )

        # ----------------------------------------------------
        # SAVE ROW
        # ----------------------------------------------------

        audit_row = {
            "match_id": match_id,
            "competition": contest["competition"],
            "season": contest["season"],
            "winner_event_id": winner_event_id,
            "winner_index": contest["winner_index"],
            "winner_team": winner_team,
            "loser_team": loser_team,
            "winner_event_type": winner_action,
            "period": contest["period"],
            "minute": contest["minute"],
            "second": contest["second"],

            "winner_possession": winner.get(
                "possession"
            ),
            "winner_possession_team": (
                winner.get("possession_team", {})
                .get("name")
            ),

            "n_events_in_window": len(
                subsequent
            ),

            "next_event_type": next_type,
            "next_event_team": next_team,
            "next_event_relation": next_relation,
            "next_event_dt": next_dt,
            "next_event_possession": next_possession,
            "next_event_possession_team": next_possession_team,

            "first_control_type": control_type,
            "first_control_team": control_team,
            "first_control_relation": control_relation,
            "first_control_dt": control_dt,
            "first_control_possession": control_possession,
            "first_control_possession_team": control_possession_team,

            "sequence": sequence_string,
        }

        audit_rows.append(
            audit_row
        )

        # Keep interesting examples:
        # especially first-control losses, no-control cases,
        # plus a limited spread of winner-retained cases.
        if (
            control_relation == "loser_team"
            or control_relation == "no_control_in_window"
            or len(examples) < 40
        ):

            examples.append(
                audit_row
            )


# ============================================================
# SAVE CSV
# ============================================================

if audit_rows:

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


# ============================================================
# SAVE HUMAN-READABLE EXAMPLES
# ============================================================

# Prioritise loser-team recoveries, then no-control,
# then winner-team cases.
relation_order = {
    "loser_team": 0,
    "no_control_in_window": 1,
    "winner_team": 2,
    "other_or_missing": 3,
}

examples = sorted(
    examples,
    key=lambda r: (
        relation_order.get(
            r["first_control_relation"],
            9,
        ),
        r["winner_event_type"],
        int(r["match_id"]),
        int(r["winner_index"]),
    ),
)

# Cap file size while preserving lots of diagnostic cases.
examples = examples[:120]

with open(
    EXAMPLES_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    for i, row in enumerate(
        examples,
        start=1,
    ):

        f.write(
            "=" * 90
            + "\n"
        )

        f.write(
            f"EXAMPLE {i}\n"
        )

        f.write(
            f"{row['competition']} "
            f"{row['season']} | "
            f"match {row['match_id']} | "
            f"{row['minute']}:{int(row['second']):02d}\n"
        )

        f.write(
            f"Aerial winner: {row['winner_team']} | "
            f"loser: {row['loser_team']} | "
            f"winner action: {row['winner_event_type']}\n"
        )

        f.write(
            f"First control candidate: "
            f"{row['first_control_relation']} | "
            f"{row['first_control_type']} | "
            f"dt={row['first_control_dt']}\n"
        )

        f.write(
            f"Sequence:\n"
            f"{row['sequence']}\n\n"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 82)
print("SECOND-BALL AUDIT SUMMARY")
print("=" * 82)

print()
print(
    f"Usable contests audited: "
    f"{len(audit_rows):,}"
)

print()
print("Aerial winner action types:")

for key, value in winner_action_counts.most_common():
    print(
        f"  {key:<24}"
        f"{value:>7,}"
    )


print()
print("Immediate next event types:")

for key, value in first_next_type_counts.most_common(15):
    print(
        f"  {str(key):<24}"
        f"{value:>7,}"
    )


print()
print("Immediate next-event team relation:")

for key, value in first_next_team_relation_counts.most_common():
    print(
        f"  {key:<24}"
        f"{value:>7,}"
    )


print()
print("First broad control-candidate event type:")

for key, value in first_control_type_counts.most_common():
    print(
        f"  {str(key):<24}"
        f"{value:>7,}"
    )


print()
print("First control-candidate team relation:")

for key, value in first_control_relation_counts.most_common():
    pct = (
        100 * value / len(audit_rows)
        if audit_rows
        else 0
    )

    print(
        f"  {key:<24}"
        f"{value:>7,} "
        f"({pct:6.2f}%)"
    )


delay_summary = describe(
    control_delays
)

if delay_summary:

    print()
    print(
        f"Time to first control candidate "
        f"(window <= {MAX_SECONDS:.0f}s):"
    )

    print(
        f"  Min:    "
        f"{delay_summary['min']:.3f}s"
    )

    print(
        f"  Mean:   "
        f"{delay_summary['mean']:.3f}s"
    )

    print(
        f"  Median: "
        f"{delay_summary['median']:.3f}s"
    )

    print(
        f"  Max:    "
        f"{delay_summary['max']:.3f}s"
    )


print()
print("No control candidate within window by winner action:")

if no_control_counts:

    for key, value in no_control_counts.most_common():
        print(
            f"  {key:<24}"
            f"{value:>7,}"
        )

else:
    print("  None")


print()
print("Saved:")
print(f"  {AUDIT_OUTPUT}")
print(f"  {EXAMPLES_OUTPUT}")

print()
print("=" * 82)
print("DONE")
print("=" * 82)

