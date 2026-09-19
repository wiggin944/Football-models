import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.tactical_model.inference import load_artifact


client = TestClient(app)


def synthetic_payload():
    artifact = load_artifact()

    rows = []

    for team_id, multiplier in [
        (1, 0.95),
        (2, 1.05),
        (3, 1.10),
        (4, 0.90),
    ]:
        row = {
            "team_id": team_id,
            "competition_id": 1,
            "season_id": 999,
            "squad": f"Synthetic Team {team_id}",
            "comp": "Synthetic League",
            "season": "2099-00",
        }

        for column in artifact[
            "feature_columns"
        ]:
            reference = artifact[
                "training_reference"
            ][column]

            value = (
                float(reference["q50"])
                * multiplier
            )

            row[column] = value

        rows.append(row)

    return {
        "rows": rows
    }


def test_ci_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ci_model_contract():
    response = client.get("/model")

    assert response.status_code == 200

    body = response.json()

    assert body["n_components"] == 6
    assert body["feature_count"] == 29


def test_ci_prediction_contract():
    response = client.post(
        "/predict/cohort",
        json=synthetic_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["predictions"]) == 4

    tactical = np.array(
        [
            [
                row["tac0"],
                row["tac1"],
                row["tac2"],
                row["tac3"],
                row["tac4"],
                row["tac5"],
            ]
            for row in body["predictions"]
        ],
        dtype=float,
    )

    assert np.isfinite(tactical).all()

    np.testing.assert_allclose(
        tactical.sum(axis=1),
        1.0,
        atol=1e-8,
    )
