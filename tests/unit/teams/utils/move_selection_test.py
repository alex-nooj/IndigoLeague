import typing

import pytest
from poke_env.environment import Move
from poke_env.environment import MoveCategory

from indigo_league.teams.utils import move_selection


def test_safe_sample_moves():
    assert move_selection.safe_sample_moves("ditto", "limber", "leftovers", [], "modest", []) == [
        "transform"
    ]


@pytest.mark.parametrize(
    "category", [MoveCategory.STATUS, MoveCategory.PHYSICAL, MoveCategory.SPECIAL]
)
def test_remove_move_category(category: MoveCategory):
    moves = {
        "flamethrower": 0,
        "machpunch": 1,
        "lightscreen": 2,
    }

    results = move_selection.remove_move_category(moves, category)

    for k, v in results.items():
        assert Move(k).category != category
        assert v == moves[k]


@pytest.mark.parametrize(
    "pokemon_name,nature,evs,moves,expected",
    [
        (
            "shuckle",
            "lonely",
            ["0", "0", "0", "0", "0", "0"],
            {"flamethrower": 0, "machpunch": 1, "lightscreen": 2},
            {"machpunch": 1, "lightscreen": 2},
        ),
        (
            "shuckle",
            "hardy",
            ["0", "12", "0", "0", "0", "0"],
            {"flamethrower": 0, "machpunch": 1, "lightscreen": 2},
            {"machpunch": 1, "lightscreen": 2},
        ),
        (
            "shuckle",
            "hardy",
            ["0", "0", "0", "0", "0", "0"],
            {"flamethrower": 0, "machpunch": 1, "lightscreen": 2},
            {"flamethrower": 0, "machpunch": 1, "lightscreen": 2},
        ),
        (
            "shuckle",
            "hardy",
            ["0", "0", "0", "12", "0", "0"],
            {"flamethrower": 0, "machpunch": 1, "lightscreen": 2},
            {"flamethrower": 0, "lightscreen": 2},
        ),
        (
            "shuckle",
            "modest",
            ["0", "0", "0", "0", "0", "0"],
            {"flamethrower": 0, "machpunch": 1, "lightscreen": 2},
            {"flamethrower": 0, "lightscreen": 2},
        ),
    ],
)
def test_remove_bad_offensive_moves(
    pokemon_name: str,
    nature: str,
    evs: typing.List[str],
    moves: typing.Dict[str, float],
    expected: typing.Dict[str, float],
):
    results = move_selection.remove_bad_offensive_moves(pokemon_name, nature, evs, moves)

    assert len(results) == len(expected)

    for k, v in results.items():
        assert k in expected
        assert v == expected[k]


@pytest.mark.parametrize(
    "item,moves,expected",
    [
        ("leftovers", {"flamethrower": 0, "lightscreen": 1}, {"flamethrower": 0, "lightscreen": 1}),
        ("assault vest", {"flamethrower": 0, "lightscreen": 1}, {"flamethrower": 0}),
        ("choice scarf", {"flamethrower": 0, "lightscreen": 1}, {"flamethrower": 0}),
        ("choice specs", {"flamethrower": 0, "lightscreen": 1}, {"flamethrower": 0}),
        ("choice band", {"flamethrower": 0, "lightscreen": 1}, {"flamethrower": 0}),
    ],
)
def test_remove_moves_based_on_item(
    item: str, moves: typing.Dict[str, float], expected: typing.Dict[str, float]
):
    results = move_selection.remove_moves_based_on_item(item, moves)

    assert len(results) == len(expected)

    for k, v in results.items():
        assert k in expected
        assert v == expected[k]


@pytest.mark.parametrize(
    "ability,moves,expected",
    [
        (
            "snowwarning",
            {"hail": 0, "sunnyday": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
            {"sunnyday": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
        ),
        (
            "snowwarning",
            {"rockthrow": 0, "sunnyday": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
            {"rockthrow": 0, "sunnyday": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
        ),
        (
            "drought",
            {"hail": 0, "sunnyday": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
            {"hail": 0, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
        ),
        (
            "drought",
            {"hail": 0, "rockthrow": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
            {"hail": 0, "rockthrow": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
        ),
        (
            "drizzle",
            {"hail": 0, "sunnyday": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
            {"hail": 0, "sunnyday": 1, "sandstorm": 3, "flamethrower": 4},
        ),
        (
            "drizzle",
            {"hail": 0, "sunnyday": 1, "rockthrow": 2, "sandstorm": 3, "flamethrower": 4},
            {"hail": 0, "sunnyday": 1, "rockthrow": 2, "sandstorm": 3, "flamethrower": 4},
        ),
        (
            "sand stream",
            {"hail": 0, "sunnyday": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
            {"hail": 0, "sunnyday": 1, "raindance": 2, "flamethrower": 4},
        ),
        (
            "sand stream",
            {"hail": 0, "sunnyday": 1, "raindance": 2, "rock throw": 3, "flamethrower": 4},
            {"hail": 0, "sunnyday": 1, "raindance": 2, "rock throw": 3, "flamethrower": 4},
        ),
        (
            "blaze",
            {"hail": 0, "sunnyday": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
            {"hail": 0, "sunnyday": 1, "raindance": 2, "sandstorm": 3, "flamethrower": 4},
        ),
    ],
)
def test_remove_weather_moves(
    ability: str, moves: typing.Dict[str, float], expected: typing.Dict[str, float]
):
    results = move_selection.remove_weather_moves(ability, moves)

    assert len(results) == len(expected)

    for k, v in results.items():
        assert k in expected
        assert v == expected[k]
