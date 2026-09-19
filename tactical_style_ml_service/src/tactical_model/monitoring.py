import numpy as np


def assess_input_distribution(
    frame,
    artifact,
):
    feature_columns = artifact[
        "feature_columns"
    ]

    reference = artifact.get(
        "training_reference"
    )

    if not reference:
        raise RuntimeError(
            "Model artifact has no training "
            "distribution reference."
        )

    total_cells = 0
    ood_cells = 0

    drifted = []
    max_mean_shift_sd = 0.0

    for column in feature_columns:
        values = frame[
            column
        ].to_numpy(float)

        stats = reference[column]

        below = values < stats["q01"]
        above = values > stats["q99"]

        feature_ood = (
            below | above
        )

        feature_ood_fraction = float(
            feature_ood.mean()
        )

        total_cells += len(values)
        ood_cells += int(
            feature_ood.sum()
        )

        std = max(
            float(stats["std"]),
            1e-9,
        )

        mean_shift_sd = abs(
            float(values.mean())
            - float(stats["mean"])
        ) / std

        max_mean_shift_sd = max(
            max_mean_shift_sd,
            mean_shift_sd,
        )

        if (
            feature_ood_fraction > 0.10
            or mean_shift_sd > 2.0
        ):
            drifted.append(column)

    return {
        "ood_cells": int(
            ood_cells
        ),
        "ood_cell_fraction": float(
            ood_cells / total_cells
            if total_cells
            else 0.0
        ),
        "drifted_features": int(
            len(drifted)
        ),
        "drifted_feature_names": (
            ",".join(drifted)
            if drifted
            else ""
        ),
        "max_mean_shift_sd": float(
            max_mean_shift_sd
        ),
    }
