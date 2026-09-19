import json

import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
)

from src.api.schemas import (
    CohortPredictionResponse,
    CohortRequest,
    HealthResponse,
    ModelInfoResponse,
)
from src.tactical_model.inference import (
    load_artifact,
    predict_feature_cohort,
)
from src.tactical_model.settings import (
    ARTIFACT_DIR,
)


MODEL_PATH = (
    ARTIFACT_DIR / "stage1_model.joblib"
)


app = FastAPI(
    title="Football Tactical Style API",
    description=(
        "Production service for the Stage 1 "
        "continuous NMF tactical-style model. "
        "Inference is cohort-based because "
        "the model uses season-relative ranks "
        "and dominance normalisation."
    ),
    version="1.0.0",
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    try:
        artifact = load_artifact(
            MODEL_PATH
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable.",
        ) from exc

    return {
        "status": "ok",
        "model_version": artifact[
            "model_version"
        ],
    }


@app.get(
    "/model",
    response_model=ModelInfoResponse,
)
def model_info():
    artifact = load_artifact(
        MODEL_PATH
    )

    return {
        "model_version": artifact[
            "model_version"
        ],
        "n_components": artifact[
            "n_components"
        ],
        "feature_count": len(
            artifact["feature_columns"]
        ),
        "feature_columns": artifact[
            "feature_columns"
        ],
        "train_seasons": artifact[
            "train_seasons"
        ],
        "random_state": artifact[
            "random_state"
        ],
    }


@app.post(
    "/predict/cohort",
    response_model=CohortPredictionResponse,
)
def predict_cohort(
    request: CohortRequest,
):
    frame = pd.DataFrame(
        [
            row.model_dump()
            for row in request.rows
        ]
    )

    try:
        predictions, diagnostics = (
            predict_feature_cohort(
                frame,
                artifact_path=MODEL_PATH,
            )
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    records = json.loads(
        predictions.to_json(
            orient="records"
        )
    )

    return {
        "predictions": records,
        "diagnostics": diagnostics,
    }
