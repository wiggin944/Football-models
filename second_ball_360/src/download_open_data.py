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

# Keep this at 1 for our first test.
# Change to None later to download every men's 360 match.
DOWNLOAD_LIMIT = 1

TIMEOUT = 30


# ============================================================
# SETUP
# ============================================================

for directory in [EVENTS_DIR, THREE_SIXTY_DIR, METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_json(url):
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ============================================================
# 1. LOAD COMPETITION INDEX
# ============================================================

print("Loading StatsBomb competition index...")

competitions_url = f"{BASE_URL}/competitions.json"
competitions = get_json(competitions_url)

print(f"Competition-season entries found: {len(competitions)}")


# ============================================================
# 2. FIND ALL MEN'S MATCHES WITH 360
# ============================================================

matches_360 = []

for competition in competitions:

    # Adult men's football only
    if competition.get("competition_gender") != "male":
        continue

    if competition.get("competition_youth", False):
        continue

    competition_id = competition["competition_id"]
    season_id = competition["season_id"]

    matches_url = (
        f"{BASE_URL}/matches/"
        f"{competition_id}/"
        f"{season_id}.json"
    )

    try:
        matches = get_json(matches_url)
    except requests.HTTPError:
        print(
            f"Could not read "
            f"{competition['competition_name']} "
            f"{competition['season_name']}"
        )
        continue

    available_matches = [
        match
        for match in matches
        if str(match.get("match_status_360", "")).lower() == "available"
    ]

    if available_matches:
        print(
            f"{competition['competition_name']} "
            f"{competition['season_name']}: "
            f"{len(available_matches)} 360 matches"
        )

    for match in available_matches:
        matches_360.append(
            {
                "match_id": match["match_id"],
                "competition_id": competition_id,
                "season_id": season_id,
                "competition": competition["competition_name"],
                "season": competition["season_name"],
                "home_team": match["home_team"]["home_team_name"],
                "away_team": match["away_team"]["away_team_name"],
                "match_date": match["match_date"],
            }
        )


print()
print("=" * 60)
print(f"TOTAL MEN'S 360 MATCHES FOUND: {len(matches_360)}")
print("=" * 60)


# ============================================================
# 3. SAVE MATCH CATALOGUE
# ============================================================

catalogue_path = METADATA_DIR / "mens_360_matches.csv"

with open(catalogue_path, "w", newline="", encoding="utf-8") as f:

    fieldnames = [
        "match_id",
        "competition_id",
        "season_id",
        "competition",
        "season",
        "home_team",
        "away_team",
        "match_date",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(matches_360)


print(f"\nSaved match catalogue to:\n{catalogue_path}")


# ============================================================
# 4. DOWNLOAD EVENT + 360 FILES
# ============================================================

matches_to_download = matches_360

if DOWNLOAD_LIMIT is not None:
    matches_to_download = matches_to_download[:DOWNLOAD_LIMIT]


print()
print(f"Downloading {len(matches_to_download)} match(es)...")
print()


for i, match in enumerate(matches_to_download, start=1):

    match_id = match["match_id"]

    event_path = EVENTS_DIR / f"{match_id}.json"
    three_sixty_path = THREE_SIXTY_DIR / f"{match_id}.json"

    print(
        f"[{i}/{len(matches_to_download)}] "
        f"{match['home_team']} v {match['away_team']} "
        f"({match['competition']} {match['season']})"
    )

    # --------------------------------------------------------
    # EVENT DATA
    # --------------------------------------------------------

    if not event_path.exists():

        events_url = f"{BASE_URL}/events/{match_id}.json"

        events = get_json(events_url)

        save_json(events, event_path)

        print(f"    Events: {len(events):,}")

    else:
        print("    Events: already downloaded")

    # --------------------------------------------------------
    # 360 DATA
    # --------------------------------------------------------

    if not three_sixty_path.exists():

        three_sixty_url = (
            f"{BASE_URL}/three-sixty/{match_id}.json"
        )

        frames = get_json(three_sixty_url)

        save_json(frames, three_sixty_path)

        print(f"    360 frames: {len(frames):,}")

    else:
        print("    360: already downloaded")

    time.sleep(0.1)


print()
print("Download complete.")