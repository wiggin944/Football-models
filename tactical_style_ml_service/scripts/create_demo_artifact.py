"""Build a deterministic synthetic model artefact for demos and CI.

This does NOT contain or reconstruct the dissertation training data.
The real fitted artefact remains private.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.tactical_model.model import fit_nmf
from src.tactical_model.preprocessing import prepare_training_matrix
from src.tactical_model.settings import (
    DOMINANCE_FEATURES,
    N_NMF,
    RNG,
    TRAIN_SEASONS,
)


FEATURE_COLUMNS = [
    "passes_per_match",
    "prog_pass_rate",
    "final_third_rate",
    "through_ball_rate",
    "switch_rate",
    "cross_rate",
    "prog_carries",
    "directness",
    "dribble_rate",
    "carries_into_box",
    "box_touch_share",
    "sca",
    "terr_ilr1",
    "terr_ilr2",
    "press_ilr1",
    "press_ilr2",
    "high_press_actions",
    "tackles",
    "interceptions",
    "blocks",
    "recoveries",
    "aerial_win_rate",
    "ppda_allowed",
    "deep",
    "deep_allowed",
    "xg_per_match",
    "setpiece_xg_share",
    "transition_threat_proxy",
    "press_intensity",
]


def build_synthetic_features():
    rng = np.random.default_rng(20260914)

    rows = []

    team_id = 1

    for season_index, season in enumerate(TRAIN_SEASONS):
        for team_index in range(24):
            latent = rng.normal(size=5)

            row = {
                "team_id": team_id,
                "competition_id": 1,
                "season_id": season_index + 1,
                "squad": f"Synthetic Team {team_index + 1}",
                "comp": "Synthetic League",
                "season": season,
            }

            for feature_index, column in enumerate(FEATURE_COLUMNS):
                base = 10.0 + feature_index * 0.75
                noise_scale = 0.6 + 0.15 * (feature_index % 5)

                row[column] = float(
                    base
                    + 0.7 * latent[feature_index % len(latent)]
                    + 0.25 * season_index
                    + rng.normal(0.0, noise_scale)
                )

            rows.append(row)
            team_id += 1

    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parents[1]

    frame = build_synthetic_features()

    (
        output,
        matrix,
        _,
        preprocessing_state,
    ) = prepare_training_matrix(
        frame,
        FEATURE_COLUMNS,
        TRAIN_SEASONS,
    )

    (
        _,
        nmf,
        component_scale,
    ) = fit_nmf(matrix)

    training_reference = {}

    for column in FEATURE_COLUMNS:
        values = output[column].to_numpy(float)

        q01, q05, q50, q95, q99 = np.quantile(
            values,
            [0.01, 0.05, 0.50, 0.95, 0.99],
        )

        training_reference[column] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "q01": float(q01),
            "q05": float(q05),
            "q50": float(q50),
            "q95": float(q95),
            "q99": float(q99),
        }

    artifact = {
        "model_version": "stage1-tactical-nmf-demo-v1",
        "feature_columns": FEATURE_COLUMNS,
        "dominance_features": list(DOMINANCE_FEATURES),
        "train_seasons": list(TRAIN_SEASONS),
        "n_components": N_NMF,
        "random_state": RNG,
        "preprocessing": preprocessing_state,
        "training_reference": training_reference,
        "component_scale": component_scale,
        "nmf": nmf,
        "demo_artifact": True,
    }

    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    model_path = artifact_dir / "stage1_model.joblib"
    metadata_path = artifact_dir / "stage1_metadata.json"

    joblib.dump(
        artifact,
        model_path,
    )

    metadata = {
        "model_version": artifact["model_version"],
        "demo_artifact": True,
        "data": "deterministic synthetic data",
        "rows": len(frame),
        "features": len(FEATURE_COLUMNS),
        "n_components": N_NMF,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Created synthetic demo artifact:")
    print(model_path)
    print("Rows:", len(frame))
    print("Features:", len(FEATURE_COLUMNS))


if __name__ == "__main__":
    main()
