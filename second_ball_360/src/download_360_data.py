import csv
import json
import time
from pathlib import Path

import requests


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://raw.githubusercontent.com/hudl/open-data/master/data"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVENTS_DIR = PROJECT_ROOT / "data" / "events"
THREE_SIXTY_DIR = PROJECT_ROOT / "data" / "three_sixty"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

CATALOGUE_PATH = METADATA_DIR / "mens_360_matches.csv"

TIMEOUT = 60
MAX_RETRIES = 4
SLEEP_BETWEEN_REQUESTS = 0.05


# ============================================================
# SETUP
# ============================================================

for directory in [EVENTS_DIR, THREE_SIXTY_DIR, METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


if not CATALOGUE_PATH.exists():
    raise FileNotFoundError(
        f"Could not find match catalogue:\n{CATALOGUE_PATH}\n\n"
        "Run download_open_data.py first."
    )


session = requests.Session()

session.headers.update(
    {
        "User-Agent": "Second-Ball-360-Research"
    }
)


# ============================================================
# HELPERS
# ============================================================

def get_json(url):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.get(
                url,
                timeout=TIMEOUT,
            )

            response.raise_for_status()

            return response.json()

        except (
            requests.RequestException,
            json.JSONDecodeError,
        ) as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                wait = attempt * 1.5

                print(
                    f"      Request failed "
                    f"(attempt {attempt}/{MAX_RETRIES}); "
                    f"retrying in {wait:.1f}s..."
                )

                time.sleep(wait)

    raise RuntimeError(
        f"Failed after {MAX_RETRIES} attempts:\n"
        f"{url}\n"
        f"{last_error}"
    )


def save_json(data, path):
    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
        )

    temp_path.replace(path)


def load_catalogue():
    with open(
        CATALOGUE_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(f)
        )


# ============================================================
# LOAD THE 300-MATCH MANIFEST
# ============================================================

matches = load_catalogue()

print("=" * 72)
print("SECOND-BALL-360 FULL DATA DOWNLOAD")
print("=" * 72)
print()
print(f"Matches in catalogue: {len(matches)}")
print()


# ============================================================
# DOWNLOAD
# ============================================================

completed = 0
downloaded_events = 0
downloaded_360 = 0
skipped_events = 0
skipped_360 = 0
failures = []


for i, match in enumerate(
    matches,
    start=1,
):

    match_id = str(
        match["match_id"]
    )

    event_path = (
        EVENTS_DIR /
        f"{match_id}.json"
    )

    frame_path = (
        THREE_SIXTY_DIR /
        f"{match_id}.json"
    )

    fixture = (
        f"{match.get('home_team', '?')} v "
        f"{match.get('away_team', '?')}"
    )

    competition = (
        f"{match.get('competition', '?')} "
        f"{match.get('season', '?')}"
    )

    print(
        f"[{i:03d}/{len(matches):03d}] "
        f"{fixture} â€” {competition}"
    )

    match_failed = False

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    if event_path.exists():

        skipped_events += 1

        print(
            "    Events: already present"
        )

    else:

        event_url = (
            f"{BASE_URL}/events/"
            f"{match_id}.json"
        )

        try:
            events = get_json(
                event_url
            )

            save_json(
                events,
                event_path,
            )

            downloaded_events += 1

            print(
                f"    Events: downloaded "
                f"({len(events):,})"
            )

        except Exception as exc:

            match_failed = True

            failures.append(
                {
                    "match_id": match_id,
                    "file": "events",
                    "error": str(exc),
                }
            )

            print(
                f"    Events: FAILED â€” {exc}"
            )


    # --------------------------------------------------------
    # 360
    # --------------------------------------------------------

    if frame_path.exists():

        skipped_360 += 1

        print(
            "    360:    already present"
        )

    else:

        frame_url = (
            f"{BASE_URL}/three-sixty/"
            f"{match_id}.json"
        )

        try:
            frames = get_json(
                frame_url
            )

            save_json(
                frames,
                frame_path,
            )

            downloaded_360 += 1

            print(
                f"    360:    downloaded "
                f"({len(frames):,} frames)"
            )

        except Exception as exc:

            match_failed = True

            failures.append(
                {
                    "match_id": match_id,
                    "file": "three_sixty",
                    "error": str(exc),
                }
            )

            print(
                f"    360:    FAILED â€” {exc}"
            )


    if not match_failed:
        completed += 1

    time.sleep(
        SLEEP_BETWEEN_REQUESTS
    )


# ============================================================
# VERIFY LOCAL FILE COUNTS
# ============================================================

local_event_ids = {
    p.stem
    for p in EVENTS_DIR.glob("*.json")
}

local_frame_ids = {
    p.stem
    for p in THREE_SIXTY_DIR.glob("*.json")
}

catalogue_ids = {
    str(match["match_id"])
    for match in matches
}

missing_events = sorted(
    catalogue_ids - local_event_ids
)

missing_360 = sorted(
    catalogue_ids - local_frame_ids
)


# ============================================================
# SAVE FAILURE LOG
# ============================================================

failure_path = (
    METADATA_DIR /
    "download_failures.csv"
)

if failures:

    with open(
        failure_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "match_id",
                "file",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(
            failures
        )

else:

    if failure_path.exists():
        failure_path.unlink()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 72)
print("DOWNLOAD SUMMARY")
print("=" * 72)

print()
print(
    f"Catalogue matches:        "
    f"{len(matches)}"
)

print(
    f"Complete matches locally: "
    f"{len(catalogue_ids & local_event_ids & local_frame_ids)}"
)

print()
print(
    f"Events downloaded now:    "
    f"{downloaded_events}"
)

print(
    f"Events already present:   "
    f"{skipped_events}"
)

print(
    f"360 files downloaded now: "
    f"{downloaded_360}"
)

print(
    f"360 files already present:"
    f" {skipped_360}"
)

print()
print(
    f"Missing event files:      "
    f"{len(missing_events)}"
)

print(
    f"Missing 360 files:        "
    f"{len(missing_360)}"
)

print(
    f"Recorded failures:        "
    f"{len(failures)}"
)


if missing_events:
    print()
    print(
        "Missing event match IDs:"
    )

    for match_id in missing_events:
        print(
            f"  {match_id}"
        )


if missing_360:
    print()
    print(
        "Missing 360 match IDs:"
    )

    for match_id in missing_360:
        print(
            f"  {match_id}"
        )


print()
print("=" * 72)

if not missing_events and not missing_360:

    print(
        "SUCCESS: all catalogue matches "
        "have both event and 360 files."
    )

else:

    print(
        "INCOMPLETE: rerun this same script. "
        "Existing files will be skipped."
    )

print("=" * 72)

