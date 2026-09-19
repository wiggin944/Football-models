from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

SOURCE_DB = DATA_DIR / "stage1_source.db"

RNG = 7
N_NMF = 6

TRAIN_SEASONS = (
    "2021-22",
    "2022-23",
    "2023-24",
)

TEST_SEASON = "2024-25"

AGG_STATS = [
    "prog_pass",
    "prog_carries",
    "pass_1_3",
    "tb",
    "sw",
    "crs",
    "cpa",
    "sca",
    "drib_att",
    "recov",
    "int",
    "blocks",
    "touches",
    "tou_def3",
    "tou_mid3",
    "tou_att3",
    "tou_att_pen",
    "pass_att",
    "pass_cmp",
    "tkl_def3",
    "tkl_mid3",
    "tkl_att3",
    "aer_won",
    "aer_lost",
    "car_prg_dist",
]

DOMINANCE_FEATURES = [
    "passes_per_match",
    "prog_pass_rate",
    "final_third_rate",
    "box_touch_share",
]
