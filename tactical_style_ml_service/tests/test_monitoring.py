import pandas as pd

from src.tactical_model.inference import load_artifact
from src.tactical_model.monitoring import (
    assess_input_distribution,
)


def reference_frame(artifact):
    rows = []

    for _ in range(10):
        rows.append(
            {
                column: artifact[
                    "training_reference"
                ][column]["q50"]
                for column in artifact[
                    "feature_columns"
                ]
            }
        )

    return pd.DataFrame(rows)


def test_monitoring_returns_diagnostics():
    artifact = load_artifact()

    diagnostics = assess_input_distribution(
        reference_frame(artifact),
        artifact,
    )

    assert "ood_cell_fraction" in diagnostics
    assert "drifted_features" in diagnostics
    assert "max_mean_shift_sd" in diagnostics


def test_extreme_distribution_shift_is_detected():
    artifact = load_artifact()

    frame = reference_frame(artifact)

    for column in artifact["feature_columns"]:
        stats = artifact[
            "training_reference"
        ][column]

        frame[column] = (
            float(stats["q99"])
            + 10.0
            * max(float(stats["std"]), 1.0)
        )

    diagnostics = assess_input_distribution(
        frame,
        artifact,
    )

    assert diagnostics["ood_cell_fraction"] > 0.90
    assert diagnostics["drifted_features"] > 0
    assert diagnostics["max_mean_shift_sd"] > 2.0
