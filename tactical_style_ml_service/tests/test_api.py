from fastapi.testclient import TestClient

from src.api.main import app
from src.tactical_model.inference import load_artifact


client = TestClient(app)


def payload():
    artifact = load_artifact()

    rows = []

    for team_id, multiplier in [
        (1, 0.95),
        (2, 1.00),
        (3, 1.05),
        (4, 1.10),
    ]:
        row = {
            "team_id": team_id,
            "competition_id": 1,
            "season_id": 999,
            "squad": f"Demo Team {team_id}",
            "comp": "Demo League",
            "season": "2099-00",
        }

        for column in artifact["feature_columns"]:
            median = artifact[
                "training_reference"
            ][column]["q50"]

            row[column] = float(median) * multiplier

        rows.append(row)

    return {"rows": rows}


def test_single_team_is_rejected():
    body = payload()
    body["rows"] = body["rows"][:1]

    response = client.post(
        "/predict/cohort",
        json=body,
    )

    assert response.status_code == 422


def test_unknown_feature_is_rejected():
    body = payload()

    body["rows"][0]["made_up_feature"] = 123

    response = client.post(
        "/predict/cohort",
        json=body,
    )

    assert response.status_code == 422


def test_oversized_cohort_is_rejected():
    body = payload()

    template = body["rows"][0].copy()
    body["rows"] = []

    for team_id in range(1, 502):
        row = template.copy()
        row["team_id"] = team_id
        row["squad"] = f"Demo Team {team_id}"
        body["rows"].append(row)

    response = client.post(
        "/predict/cohort",
        json=body,
    )

    assert response.status_code == 422
