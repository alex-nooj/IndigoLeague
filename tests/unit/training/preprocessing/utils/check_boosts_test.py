import typing

import pytest
from poke_env.environment import Pokemon
from unittest.mock import patch, PropertyMock

from indigo_league.training.preprocessing.utils import check_boosts


@pytest.mark.parametrize(
    "defense,special_defense,expected",
    [
        (0, 0, 1.0),
        (-3, 0, -1.0),
        (0, -3, -1.0),
        (-3, -3, -1.0),
        (-4, 0, -1.0),
    ],
)
def test_check_defense(defense: int, special_defense: int, expected: float):
    mon = Pokemon(species="charizard")
    mon._boosts["def"] = defense
    mon._boosts["spd"] = special_defense

    assert check_boosts.check_defense(mon) == expected


@pytest.mark.parametrize(
    "boosts,stats,expected",
    [
        (
            {"atk": 0.0, "def": 0.0, "spa": 0.0, "spd": 0.0, "spe": 0.0},
            {"atk": 0.0, "def": 0.0, "spa": 0.0, "spd": 0.0, "spe": 0.0},
            1.0,
        ),
        (
            {"atk": -3.0, "def": 0.0, "spa": 0.0, "spd": 0.0, "spe": 0.0},
            {"atk": 200.0, "def": 0.0, "spa": 200.0, "spd": 0.0, "spe": 0.0},
            -1.0,
        ),
        (
            {"atk": 0.0, "def": 0.0, "spa": -3.0, "spd": 0.0, "spe": 0.0},
            {"atk": 0.0, "def": 0.0, "spa": 0.0, "spd": 0.0, "spe": 0.0},
            -1.0,
        ),
        (
            {"atk": -3.0, "def": 0.0, "spa": 0.0, "spd": 0.0, "spe": 0.0},
            {"atk": 0.0, "def": 0.0, "spa": 1.0, "spd": 0.0, "spe": 0.0},
            1.0,
        ),
    ],
)
def test_check_attack(
    boosts: typing.Dict[str, float], stats: typing.Dict[str, float], expected: float
):
    with patch("poke_env.environment.Pokemon.stats", new_callable=PropertyMock) as mock_method:
        mock_method.return_value = stats

        mon = Pokemon(species="charizard")
        mon._boosts = boosts
        assert check_boosts.check_attack(mon) == expected
