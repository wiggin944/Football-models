import pytest

from src.tactical_model.preprocessing import (
    assert_test_excluded,
)


def test_test_season_cannot_enter_training():
    with pytest.raises(
        RuntimeError,
        match="Temporal leakage",
    ):
        assert_test_excluded(
            [
                "2021-22",
                "2022-23",
                "2023-24",
                "2024-25",
            ]
        )


def test_training_seasons_are_allowed():
    assert_test_excluded(
        [
            "2021-22",
            "2022-23",
            "2023-24",
        ]
    )
