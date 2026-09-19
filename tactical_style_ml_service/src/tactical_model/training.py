import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib

from .data import make_engine
from .features import build_features
from .model import fit_nmf
from .preprocessing import prepare_training_matrix
from .settings import (
    ARTIFACT_DIR,
    DOMINANCE_FEATURES,
    N_NMF,
    RNG,
    SOURCE_DB,
    TRAIN_SEASONS,
)


MODEL_VERSION = "stage1-tactical-nmf-v1"


def train(
    db_path=SOURCE_DB,
    artifact_dir=ARTIFACT_DIR,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    engine = make_engine(db_path)

    all_features, feature_columns = (
        build_features(engine)
    )

    (
        output,
        matrix,
        dominance,
        preprocessing_state,
    ) = prepare_training_matrix(
        all_features,
        feature_columns,
        TRAIN_SEASONS,
    )

    (
        tactical_scores,
        nmf,
        component_scale,
    ) = fit_nmf(matrix)

    for component in range(N_NMF):
        output[f"tac{component}"] = (
            tactical_scores[:, component]
        )

    output["team_dom_z"] = dominance

    result_columns = [
        "team_id",
        "competition_id",
        "season_id",
        "team_dom_z",
        *[
            f"tac{i}"
            for i in range(N_NMF)
        ],
    ]

    results = (
        output[result_columns]
        .copy()
        .sort_values(
            [
                "team_id",
                "competition_id",
                "season_id",
            ]
        )
        .reset_index(drop=True)
    )

    training_reference = {}

    for column in feature_columns:
        values = output[column].to_numpy(float)

        q01, q05, q50, q95, q99 = (
            __import__("numpy").quantile(
                values,
                [0.01, 0.05, 0.50, 0.95, 0.99],
            )
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
        "model_version": MODEL_VERSION,
        "feature_columns": list(
            feature_columns
        ),
        "dominance_features": list(
            DOMINANCE_FEATURES
        ),
        "train_seasons": list(
            TRAIN_SEASONS
        ),
        "n_components": N_NMF,
        "random_state": RNG,
        "preprocessing": preprocessing_state,
        "training_reference": training_reference,
        "component_scale": component_scale,
        "nmf": nmf,
    }

    model_path = (
        artifact_dir
        / "stage1_model.joblib"
    )

    results_path = (
        artifact_dir
        / "training_team_style.csv"
    )

    metadata_path = (
        artifact_dir
        / "stage1_metadata.json"
    )

    joblib.dump(
        artifact,
        model_path,
    )

    results.to_csv(
        results_path,
        index=False,
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "training_rows": len(results),
        "training_seasons": list(
            TRAIN_SEASONS
        ),
        "feature_count": len(
            feature_columns
        ),
        "feature_columns": list(
            feature_columns
        ),
        "n_components": N_NMF,
        "random_state": RNG,
        "reconstruction_error": float(
            nmf.reconstruction_err_
        ),
        "versions": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version(
                "scikit-learn"
            ),
            "sqlalchemy": version(
                "sqlalchemy"
            ),
            "joblib": version("joblib"),
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[train] seasons:",
        list(TRAIN_SEASONS),
    )
    print(
        "[train] rows:",
        len(results),
    )
    print(
        "[train] features:",
        len(feature_columns),
    )
    print(
        "[train] reconstruction error:",
        nmf.reconstruction_err_,
    )
    print(
        "[artifact]",
        model_path,
    )
    print(
        "[metadata]",
        metadata_path,
    )

    return results


if __name__ == "__main__":
    train()
