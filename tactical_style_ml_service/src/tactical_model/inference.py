from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import make_engine
from .features import (
    build_features,
    zscore_within_season,
)
from .settings import ARTIFACT_DIR
from .monitoring import assess_input_distribution


DEFAULT_ARTIFACT = (
    ARTIFACT_DIR / "stage1_model.joblib"
)

ID_COLUMNS = [
    "team_id",
    "competition_id",
    "season_id",
    "squad",
    "comp",
    "season",
]


@lru_cache(maxsize=4)
def load_artifact(
    artifact_path=DEFAULT_ARTIFACT,
):
    artifact_path = Path(artifact_path)

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: "
            f"{artifact_path}"
        )

    return joblib.load(artifact_path)


def _validate_feature_frame(
    frame,
    artifact,
):
    feature_columns = artifact[
        "feature_columns"
    ]

    required = (
        ID_COLUMNS
        + feature_columns
    )

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Missing required inference columns: "
            + ", ".join(missing)
        )

    if frame.empty:
        raise ValueError(
            "Inference cohort is empty."
        )

    duplicate_keys = frame.duplicated(
        [
            "team_id",
            "competition_id",
            "season_id",
        ]
    )

    if duplicate_keys.any():
        raise ValueError(
            "Duplicate team-competition-season "
            "rows in inference cohort."
        )


def prepare_inference_matrix(
    frame,
    artifact,
):
    _validate_feature_frame(
        frame,
        artifact,
    )

    feature_columns = artifact[
        "feature_columns"
    ]

    output = frame.copy()

    output[feature_columns] = (
        output[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # Mirror the original structure:
    # 1. incoming competition-season median
    # 2. stored training global median
    for column in feature_columns:
        output[column] = (
            output[column]
            .fillna(
                output.groupby(
                    ["comp", "season"]
                )[column]
                .transform("median")
            )
        )

        training_median = artifact[
            "preprocessing"
        ]["global_medians"][column]

        output[column] = (
            output[column]
            .fillna(training_median)
        )

    remaining_missing = (
        output[feature_columns]
        .isna()
        .sum()
    )

    if (remaining_missing > 0).any():
        raise ValueError(
            "Inference imputation left "
            "missing values."
        )

    dominance = (
        zscore_within_season(
            output,
            artifact[
                "dominance_features"
            ],
        )
        .mean(axis=1)
        .to_numpy(float)
    )

    ranks = (
        output.groupby("season")[
            feature_columns
        ]
        .rank(pct=True)
    )

    residuals = ranks.copy()

    slopes = artifact[
        "preprocessing"
    ]["residual_slopes"]

    intercepts = artifact[
        "preprocessing"
    ]["residual_intercepts"]

    residual_minimum = artifact[
        "preprocessing"
    ]["residual_minimum"]

    for column in feature_columns:
        residuals[column] = (
            ranks[column].to_numpy(float)
            - (
                slopes[column]
                * dominance
                + intercepts[column]
            )
        )

    shift = np.array(
        [
            residual_minimum[column]
            for column
            in feature_columns
        ],
        dtype=float,
    )

    matrix = (
        residuals.to_numpy(float)
        - shift
    )

    below_training_support = (
        matrix < 0
    )

    clipped_cells = int(
        below_training_support.sum()
    )

    clipped_rows = int(
        below_training_support.any(
            axis=1
        ).sum()
    )

    # NMF requires non-negative inputs.
    # Values below the training residual
    # support are clipped and surfaced
    # explicitly as an OOD diagnostic.
    matrix = np.clip(
        matrix,
        0.0,
        None,
    )

    diagnostics = {
        "rows": int(len(output)),
        "clipped_cells": clipped_cells,
        "clipped_rows": clipped_rows,
        "clipped_cell_fraction": float(
            clipped_cells
            / matrix.size
        ),
    }

    diagnostics.update(
        assess_input_distribution(
            output,
            artifact,
        )
    )

    return (
        output,
        matrix,
        dominance,
        diagnostics,
    )


def predict_feature_cohort(
    frame,
    artifact_path=DEFAULT_ARTIFACT,
):
    artifact = load_artifact(
        artifact_path
    )

    (
        output,
        matrix,
        dominance,
        diagnostics,
    ) = prepare_inference_matrix(
        frame,
        artifact,
    )

    scores = artifact[
        "nmf"
    ].transform(matrix)

    score_total = scores.sum(
        axis=1,
        keepdims=True,
    )

    normalised = np.divide(
        scores,
        score_total,
        out=np.full_like(
            scores,
            1.0 / scores.shape[1],
        ),
        where=score_total > 1e-12,
    )

    result = output[
        [
            "team_id",
            "competition_id",
            "season_id",
            "squad",
            "comp",
            "season",
        ]
    ].copy()

    result["team_dom_z"] = dominance

    for component in range(
        artifact["n_components"]
    ):
        result[
            f"tac{component}"
        ] = normalised[
            :,
            component,
        ]

    tactical_columns = [
        f"tac{i}"
        for i in range(
            artifact["n_components"]
        )
    ]

    if not np.isfinite(
        result[
            ["team_dom_z"]
            + tactical_columns
        ].to_numpy(float)
    ).all():
        raise RuntimeError(
            "Inference produced "
            "non-finite outputs."
        )

    row_sums = (
        result[tactical_columns]
        .sum(axis=1)
        .to_numpy(float)
    )

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "Tactical component shares "
            "do not sum to one."
        )

    diagnostics[
        "model_version"
    ] = artifact[
        "model_version"
    ]

    return result, diagnostics


def predict_database_cohort(
    db_path,
    seasons=None,
    artifact_path=DEFAULT_ARTIFACT,
):
    engine = make_engine(
        db_path
    )

    frame, feature_columns = (
        build_features(engine)
    )

    artifact = load_artifact(
        artifact_path
    )

    if list(feature_columns) != list(
        artifact["feature_columns"]
    ):
        raise RuntimeError(
            "Database feature schema "
            "does not match model artifact."
        )

    if seasons is not None:
        seasons = {
            str(season)
            for season in seasons
        }

        frame = frame[
            frame["season"]
            .astype(str)
            .isin(seasons)
        ].copy()

    return predict_feature_cohort(
        frame,
        artifact_path,
    )
