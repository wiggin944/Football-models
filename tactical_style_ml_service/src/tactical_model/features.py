import numpy as np
import pandas as pd

from .data import (
    read_panel,
    read_setpiece,
    read_team_match_metrics,
)
from .settings import AGG_STATS


def ilr3(a, b, c, eps=1e-6):
    a, b, c = a + eps, b + eps, c + eps

    return (
        np.sqrt(0.5) * np.log(a / b),
        np.sqrt(2 / 3) * np.log(np.sqrt(a * b) / c),
    )


def zscore_within_season(df, cols):
    output = df[cols].copy()

    for season in df.season.unique():
        rows = df.season == season
        block = df.loc[rows, cols]

        output.loc[rows] = (
            block - block.mean()
        ) / (
            block.std(ddof=0) + 1e-9
        )

    return output


def team_match_totals(panel):
    frame = panel[
        (panel.n90 > 0)
        & panel.tkl.notna()
    ].copy()

    frame = frame[
        ~frame.pos.astype(str).str.startswith("GK")
    ]

    keys = [
        "team_id",
        "competition_id",
        "season_id",
        "squad",
        "comp",
        "season",
    ]

    weighted = frame[AGG_STATS].multiply(
        frame.n90,
        axis=0,
    )

    weighted[keys] = frame[keys]

    sums = weighted.groupby(keys)[AGG_STATS].sum()
    minutes = frame.groupby(keys).n90.sum()

    return sums.div(minutes, axis=0) * 10.0


def build_style_features(frame):
    output = pd.DataFrame(index=frame.index)

    output["passes_per_match"] = frame.pass_att
    output["prog_pass_rate"] = frame.prog_pass / frame.pass_att
    output["final_third_rate"] = frame.pass_1_3 / frame.pass_att
    output["through_ball_rate"] = frame.tb / frame.pass_att
    output["switch_rate"] = frame.sw / frame.pass_att
    output["cross_rate"] = frame.crs / frame.pass_att
    output["prog_carries"] = frame.prog_carries
    output["directness"] = frame.car_prg_dist / frame.touches
    output["dribble_rate"] = frame.drib_att / frame.touches * 100
    output["carries_into_box"] = frame.cpa
    output["box_touch_share"] = frame.tou_att_pen / frame.tou_att3
    output["sca"] = frame.sca

    (
        output["terr_ilr1"],
        output["terr_ilr2"],
    ) = ilr3(
        frame.tou_att3,
        frame.tou_mid3,
        frame.tou_def3,
    )

    (
        output["press_ilr1"],
        output["press_ilr2"],
    ) = ilr3(
        frame.tkl_att3,
        frame.tkl_mid3,
        frame.tkl_def3,
    )

    output["high_press_actions"] = frame.tkl_att3
    output["tackles"] = (
        frame.tkl_def3
        + frame.tkl_mid3
        + frame.tkl_att3
    )
    output["interceptions"] = frame["int"]
    output["blocks"] = frame.blocks
    output["recoveries"] = frame.recov

    output["aerial_win_rate"] = (
        frame.aer_won
        / (frame.aer_won + frame.aer_lost)
    )

    return output.reset_index()


def build_features(engine):
    style = build_style_features(
        team_match_totals(
            read_panel(engine)
        )
    )

    output = style.merge(
        read_team_match_metrics(engine),
        on=[
            "team_id",
            "competition_id",
            "season_id",
        ],
        how="left",
    )

    output = output.merge(
        read_setpiece(engine),
        on=[
            "team_id",
            "season_id",
        ],
        how="left",
    )

    output["press_intensity"] = -output["ppda"]
    output = output.drop(columns=["ppda"])

    id_columns = (
        "team_id",
        "competition_id",
        "season_id",
        "squad",
        "comp",
        "season",
    )

    feature_columns = [
        column
        for column in output.columns
        if column not in id_columns
    ]

    output[feature_columns] = (
        output[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return output, feature_columns
