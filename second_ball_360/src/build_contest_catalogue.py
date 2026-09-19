import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVENTS_DIR = PROJECT_ROOT / "data" / "events"
THREE_SIXTY_DIR = PROJECT_ROOT / "data" / "three_sixty"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MATCH_CATALOGUE = METADATA_DIR / "mens_360_matches.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATALOGUE_OUTPUT = OUTPUT_DIR / "all_aerial_contests.csv"
FRAMES_OUTPUT = OUTPUT_DIR / "all_aerial_contest_frames.jsonl"
FAILURES_OUTPUT = OUTPUT_DIR / "all_aerial_contest_failures.csv"


# ============================================================
# HELPERS
# ============================================================

def get_aerial_won_section(event):
    """
    Find which event subsection contains aerial_won=True.
    """
    for key, value in event.items():
        if isinstance(value, dict) and value.get("aerial_won") is True:
            return key

    return None


def safe_name(obj, key):
    """
    Safely extract nested StatsBomb name fields.
    """
    value = obj.get(key)

    if isinstance(value, dict):
        return value.get("name")

    return None


def load_match_metadata():
    if not MATCH_CATALOGUE.exists():
        raise FileNotFoundError(
            f"Missing match catalogue:\n{MATCH_CATALOGUE}"
        )

    with open(
        MATCH_CATALOGUE,
        "r",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    return {
        str(row["match_id"]): row
        for row in rows
    }


# ============================================================
# LOAD MATCH METADATA
# ============================================================

match_metadata = load_match_metadata()

match_ids = sorted(
    match_metadata.keys(),
    key=int,
)

print("=" * 80)
print("FULL 300-MATCH AERIAL CONTEST BUILD")
print("=" * 80)
print()
print(f"Matches in metadata catalogue: {len(match_ids)}")
print()


# ============================================================
# ACCUMULATORS
# ============================================================

rows = []
frame_records = []
failures = []

winner_action_counts = Counter()
competition_counts = Counter()

match_contest_counts = {}
match_usable_counts = {}

total_aerial_lost = 0
total_paired = 0
total_unpaired = 0
total_ambiguous = 0
total_winner_360 = 0
total_no_winner_360 = 0

visible_player_counts = []
winner_team_visible_counts = []
opponent_visible_counts = []


# ============================================================
# PROCESS EVERY MATCH
# ============================================================

for match_number, match_id in enumerate(match_ids, start=1):

    meta = match_metadata[match_id]

    event_path = EVENTS_DIR / f"{match_id}.json"
    frame_path = THREE_SIXTY_DIR / f"{match_id}.json"

    fixture = (
        f"{meta.get('home_team', '?')} v "
        f"{meta.get('away_team', '?')}"
    )

    competition_label = (
        f"{meta.get('competition', '?')} "
        f"{meta.get('season', '?')}"
    )

    print(
        f"[{match_number:03d}/{len(match_ids):03d}] "
        f"{fixture} — {competition_label}"
    )

    if not event_path.exists():
        failures.append(
            {
                "match_id": match_id,
                "kind": "missing_events_file",
                "event_id": "",
                "details": str(event_path),
            }
        )
        print("    ERROR: missing events file")
        continue

    if not frame_path.exists():
        failures.append(
            {
                "match_id": match_id,
                "kind": "missing_360_file",
                "event_id": "",
                "details": str(frame_path),
            }
        )
        print("    ERROR: missing 360 file")
        continue

    with open(
        event_path,
        "r",
        encoding="utf-8",
    ) as f:
        events = json.load(f)

    with open(
        frame_path,
        "r",
        encoding="utf-8",
    ) as f:
        frames = json.load(f)

    event_by_id = {
        event["id"]: event
        for event in events
    }

    frame_by_id = {
        frame["event_uuid"]: frame
        for frame in frames
    }

    match_total = 0
    match_usable = 0

    for loser in events:

        duel_type = (
            loser.get("duel", {})
            .get("type", {})
            .get("name")
        )

        if duel_type != "Aerial Lost":
            continue

        total_aerial_lost += 1
        match_total += 1

        candidate_winners = []

        for related_id in loser.get("related_events", []):

            related = event_by_id.get(related_id)

            if related is None:
                continue

            aerial_section = get_aerial_won_section(related)

            if aerial_section is not None:
                candidate_winners.append(
                    (related, aerial_section)
                )

        if len(candidate_winners) == 0:

            total_unpaired += 1

            failures.append(
                {
                    "match_id": match_id,
                    "kind": "unpaired_aerial_lost",
                    "event_id": loser["id"],
                    "details": (
                        f"index={loser.get('index')} "
                        f"related={loser.get('related_events', [])}"
                    ),
                }
            )

            continue

        if len(candidate_winners) > 1:

            total_ambiguous += 1

            failures.append(
                {
                    "match_id": match_id,
                    "kind": "ambiguous_multiple_winners",
                    "event_id": loser["id"],
                    "details": (
                        f"index={loser.get('index')} "
                        f"winner_ids="
                        f"{[x[0]['id'] for x in candidate_winners]}"
                    ),
                }
            )

            continue

        winner, aerial_section = candidate_winners[0]

        total_paired += 1

        winner_event_id = winner["id"]

        winner_frame = frame_by_id.get(
            winner_event_id
        )

        has_winner_360 = winner_frame is not None

        if has_winner_360:
            total_winner_360 += 1
            match_usable += 1
        else:
            total_no_winner_360 += 1

        winner_action = winner["type"]["name"]

        winner_action_counts[
            winner_action
        ] += 1

        competition_counts[
            competition_label
        ] += 1

        winner_location = winner.get("location")

        # ----------------------------------------------------
        # CANONICAL 360 FRAME SUMMARY
        # ----------------------------------------------------

        n_visible = None
        n_winner_team_visible = None
        n_opponent_visible = None
        n_actor = None
        n_keeper = None
        visible_area = None

        if winner_frame is not None:

            freeze_frame = winner_frame.get(
                "freeze_frame",
                [],
            )

            visible_area = winner_frame.get(
                "visible_area"
            )

            n_visible = len(
                freeze_frame
            )

            n_winner_team_visible = sum(
                player.get("teammate") is True
                for player in freeze_frame
            )

            n_opponent_visible = sum(
                player.get("teammate") is False
                for player in freeze_frame
            )

            n_actor = sum(
                player.get("actor") is True
                for player in freeze_frame
            )

            n_keeper = sum(
                player.get("keeper") is True
                for player in freeze_frame
            )

            visible_player_counts.append(
                n_visible
            )

            winner_team_visible_counts.append(
                n_winner_team_visible
            )

            opponent_visible_counts.append(
                n_opponent_visible
            )

            # Full canonical graph material for downstream use.
            frame_records.append(
                {
                    "match_id": int(match_id),
                    "loser_event_id": loser["id"],
                    "winner_event_id": winner_event_id,
                    "winner_team": winner["team"]["name"],
                    "loser_team": loser["team"]["name"],
                    "winner_event_type": winner_action,
                    "aerial_won_section": aerial_section,
                    "period": winner.get("period"),
                    "minute": winner.get("minute"),
                    "second": winner.get("second"),
                    "contest_location": winner_location,
                    "freeze_frame": freeze_frame,
                    "visible_area": visible_area,
                }
            )

        # ----------------------------------------------------
        # EVENT-SPECIFIC CONTEXT
        # ----------------------------------------------------

        pass_data = winner.get(
            "pass",
            {},
        )

        clearance_data = winner.get(
            "clearance",
            {},
        )

        shot_data = winner.get(
            "shot",
            {},
        )

        miscontrol_data = winner.get(
            "miscontrol",
            {},
        )

        row = {
            "match_id": match_id,
            "competition": meta.get("competition"),
            "season": meta.get("season"),
            "match_date": meta.get("match_date"),
            "home_team": meta.get("home_team"),
            "away_team": meta.get("away_team"),

            "loser_event_id": loser["id"],
            "winner_event_id": winner_event_id,

            "loser_index": loser.get("index"),
            "winner_index": winner.get("index"),

            "period": winner.get("period"),
            "minute": winner.get("minute"),
            "second": winner.get("second"),

            "loser_team": loser["team"]["name"],
            "winner_team": winner["team"]["name"],

            "winner_event_type": winner_action,
            "aerial_won_section": aerial_section,

            "contest_x": (
                winner_location[0]
                if winner_location
                else None
            ),
            "contest_y": (
                winner_location[1]
                if winner_location
                else None
            ),

            "has_winner_360": has_winner_360,

            "n_visible": n_visible,
            "n_winner_team_visible": n_winner_team_visible,
            "n_opponent_visible": n_opponent_visible,
            "n_actor": n_actor,
            "n_keeper": n_keeper,

            "possession": winner.get("possession"),
            "possession_team": (
                winner.get("possession_team", {})
                .get("name")
            ),

            "pass_height": safe_name(
                pass_data,
                "height",
            ),

            "pass_type": safe_name(
                pass_data,
                "type",
            ),

            "pass_length": pass_data.get(
                "length"
            ),

            "clearance_head": clearance_data.get(
                "head"
            ),

            "shot_body_part": safe_name(
                shot_data,
                "body_part",
            ),

            "shot_outcome": safe_name(
                shot_data,
                "outcome",
            ),

            "miscontrol_aerial_won": miscontrol_data.get(
                "aerial_won"
            ),
        }

        rows.append(row)

    match_contest_counts[
        match_id
    ] = match_total

    match_usable_counts[
        match_id
    ] = match_usable

    print(
        f"    physical contests: {match_total:>3} | "
        f"canonical 360 usable: {match_usable:>3}"
    )


# ============================================================
# SAVE CONTEST CATALOGUE
# ============================================================

if rows:

    with open(
        CATALOGUE_OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# SAVE FULL CANONICAL FRAMES AS JSONL
# ============================================================

with open(
    FRAMES_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    for record in frame_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# SAVE FAILURES
# ============================================================

if failures:

    with open(
        FAILURES_OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "match_id",
                "kind",
                "event_id",
                "details",
            ],
        )

        writer.writeheader()
        writer.writerows(
            failures
        )

else:

    if FAILURES_OUTPUT.exists():
        FAILURES_OUTPUT.unlink()


# ============================================================
# SUMMARY HELPERS
# ============================================================

def describe(values):

    if not values:
        return None

    values = sorted(
        values
    )

    n = len(values)

    return {
        "min": values[0],
        "mean": sum(values) / n,
        "median": (
            values[n // 2]
            if n % 2 == 1
            else (
                values[n // 2 - 1]
                +
                values[n // 2]
            ) / 2
        ),
        "max": values[-1],
    }


visible_summary = describe(
    visible_player_counts
)

contest_per_match = describe(
    list(
        match_contest_counts.values()
    )
)

usable_per_match = describe(
    list(
        match_usable_counts.values()
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("FULL-DATASET SUMMARY")
print("=" * 80)

print()
print(
    f"Matches processed:                  "
    f"{len(match_contest_counts)}"
)

print(
    f"Physical aerial-contest anchors:    "
    f"{total_aerial_lost:,}"
)

print(
    f"Uniquely paired contests:           "
    f"{total_paired:,}"
)

print(
    f"Unpaired contests:                  "
    f"{total_unpaired:,}"
)

print(
    f"Ambiguous contests:                 "
    f"{total_ambiguous:,}"
)

print()
print(
    f"Paired contests with winner 360:    "
    f"{total_winner_360:,}"
)

print(
    f"Paired contests without winner 360: "
    f"{total_no_winner_360:,}"
)

if total_paired:

    print(
        f"Canonical 360 coverage:             "
        f"{100 * total_winner_360 / total_paired:.2f}%"
    )


print()
print("Winner action types:")

for action, count in winner_action_counts.most_common():

    print(
        f"  {action:<24}"
        f"{count:>7,}"
    )


print()
print("Contests by competition-season:")

for competition, count in sorted(
    competition_counts.items()
):

    print(
        f"  {competition:<42}"
        f"{count:>7,}"
    )


if contest_per_match:

    print()
    print("Physical contests per match:")

    print(
        f"  Min:    "
        f"{contest_per_match['min']:.0f}"
    )

    print(
        f"  Mean:   "
        f"{contest_per_match['mean']:.2f}"
    )

    print(
        f"  Median: "
        f"{contest_per_match['median']:.2f}"
    )

    print(
        f"  Max:    "
        f"{contest_per_match['max']:.0f}"
    )


if usable_per_match:

    print()
    print("Canonical 360 contests per match:")

    print(
        f"  Min:    "
        f"{usable_per_match['min']:.0f}"
    )

    print(
        f"  Mean:   "
        f"{usable_per_match['mean']:.2f}"
    )

    print(
        f"  Median: "
        f"{usable_per_match['median']:.2f}"
    )

    print(
        f"  Max:    "
        f"{usable_per_match['max']:.0f}"
    )


if visible_summary:

    print()
    print("Visible players in canonical frames:")

    print(
        f"  Min:    "
        f"{visible_summary['min']:.0f}"
    )

    print(
        f"  Mean:   "
        f"{visible_summary['mean']:.2f}"
    )

    print(
        f"  Median: "
        f"{visible_summary['median']:.2f}"
    )

    print(
        f"  Max:    "
        f"{visible_summary['max']:.0f}"
    )


print()
print(
    f"Failure / exception records:        "
    f"{len(failures):,}"
)

print()
print("Saved:")
print(f"  {CATALOGUE_OUTPUT}")
print(f"  {FRAMES_OUTPUT}")

if failures:
    print(f"  {FAILURES_OUTPUT}")

print()
print("=" * 80)
print("DONE")
print("=" * 80)
