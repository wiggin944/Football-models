import numpy as np

from .features import zscore_within_season
from .settings import (
    DOMINANCE_FEATURES,
    TEST_SEASON,
)


def assert_test_excluded(seasons):
    seasons = {str(season) for season in seasons}

    if TEST_SEASON in seasons:
        raise RuntimeError(
            f"Temporal leakage: test season {TEST_SEASON} "
            "was included in a model-fitting dataset."
        )


def prepare_training_matrix(
    all_features,
    feature_columns,
    train_seasons,
):
    assert_test_excluded(train_seasons)

    output = all_features[
        all_features["season"].isin(train_seasons)
    ].copy()

    if output.empty:
        raise RuntimeError(
            f"No tactics rows found for training seasons "
            f"{train_seasons}."
        )

    unexpected = sorted(
        set(output["season"].astype(str))
        - set(train_seasons)
    )

    if unexpected:
        raise RuntimeError(
            f"Unexpected tactics fitting seasons: {unexpected}"
        )

    group_medians = (
        output.groupby(["comp", "season"])[
            feature_columns
        ]
        .median()
        .copy()
    )

    global_medians = {}

    for column in feature_columns:
        output[column] = output[column].fillna(
            output.groupby(
                ["comp", "season"]
            )[column].transform("median")
        )

        global_median = output[column].median()
        global_medians[column] = float(global_median)

        output[column] = output[column].fillna(
            global_median
        )

    remaining_missing = (
        output[feature_columns]
        .isna()
        .sum()
    )

    if (remaining_missing > 0).any():
        raise RuntimeError(
            "Training-only tactics imputation left missing values:\n"
            + remaining_missing[
                remaining_missing > 0
            ].to_string()
        )

    dominance = (
        zscore_within_season(
            output,
            DOMINANCE_FEATURES,
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

    slopes = {}
    intercepts = {}

    for column in feature_columns:
        slope, intercept = np.polyfit(
            dominance,
            ranks[column].to_numpy(float),
            1,
        )

        slopes[column] = float(slope)
        intercepts[column] = float(intercept)

        residuals[column] = (
            ranks[column].to_numpy(float)
            - (
                slope * dominance
                + intercept
            )
        )

    residual_minimum = residuals.min(axis=0)

    matrix = residuals.to_numpy(float)
    matrix = (
        matrix
        - residual_minimum.to_numpy(float)
    )

    state = {
        "group_medians": group_medians,
        "global_medians": global_medians,
        "residual_slopes": slopes,
        "residual_intercepts": intercepts,
        "residual_minimum": {
            column: float(
                residual_minimum[column]
            )
            for column in feature_columns
        },
    }

    return (
        output,
        matrix,
        dominance,
        state,
    )
