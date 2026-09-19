from collections import Counter

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class TeamFeatureRow(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    team_id: int = Field(gt=0)
    competition_id: int = Field(gt=0)
    season_id: int = Field(gt=0)

    squad: str = Field(min_length=1)
    comp: str = Field(min_length=1)
    season: str = Field(min_length=1)

    passes_per_match: float | None = None
    prog_pass_rate: float | None = None
    final_third_rate: float | None = None
    through_ball_rate: float | None = None
    switch_rate: float | None = None
    cross_rate: float | None = None
    prog_carries: float | None = None
    directness: float | None = None
    dribble_rate: float | None = None
    carries_into_box: float | None = None
    box_touch_share: float | None = None
    sca: float | None = None

    terr_ilr1: float | None = None
    terr_ilr2: float | None = None
    press_ilr1: float | None = None
    press_ilr2: float | None = None

    high_press_actions: float | None = None
    tackles: float | None = None
    interceptions: float | None = None
    blocks: float | None = None
    recoveries: float | None = None
    aerial_win_rate: float | None = None

    ppda_allowed: float | None = None
    deep: float | None = None
    deep_allowed: float | None = None
    xg_per_match: float | None = None

    setpiece_xg_share: float | None = None
    transition_threat_proxy: float | None = None
    press_intensity: float | None = None


class CohortRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    rows: list[TeamFeatureRow] = Field(
        min_length=2,
        max_length=500,
    )

    @model_validator(mode="after")
    def validate_cohort(self):
        keys = [
            (
                row.team_id,
                row.competition_id,
                row.season_id,
            )
            for row in self.rows
        ]

        if len(keys) != len(set(keys)):
            raise ValueError(
                "Duplicate team-competition-season rows."
            )

        season_counts = Counter(
            row.season
            for row in self.rows
        )

        too_small = [
            season
            for season, count
            in season_counts.items()
            if count < 2
        ]

        if too_small:
            raise ValueError(
                "Each season requires cohort context "
                "from at least two teams. "
                f"Too small: {too_small}"
            )

        return self


class TeamPrediction(BaseModel):
    team_id: int
    competition_id: int
    season_id: int

    squad: str
    comp: str
    season: str

    team_dom_z: float
    tac0: float
    tac1: float
    tac2: float
    tac3: float
    tac4: float
    tac5: float


class CohortPredictionResponse(BaseModel):
    predictions: list[TeamPrediction]

    diagnostics: dict[
        str,
        int | float | str
    ]


class HealthResponse(BaseModel):
    status: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    n_components: int
    feature_count: int
    feature_columns: list[str]
    train_seasons: list[str]
    random_state: int
