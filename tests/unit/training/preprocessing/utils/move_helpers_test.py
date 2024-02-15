import typing

import pytest
from poke_env.environment import Field
from poke_env.environment import Move
from poke_env.environment import Pokemon
from poke_env.environment import SideCondition
from poke_env.environment import Status
from poke_env.environment import Weather

from indigo_league.training.preprocessing.utils import move_helpers


@pytest.mark.parametrize(
    "names",
    [
        list(move_helpers.ENTRY_HAZARDS.keys()),
        list(move_helpers.SETUP_MOVES.keys()),
        list(move_helpers.ANTI_HAZARDS_MOVES),
    ],
)
def test_constants(names: typing.List[str]):
    for k in names:
        m = Move(k)
    assert True


@pytest.mark.parametrize(
    "move,tgt_side_conditions,expected",
    [
        (Move("flamethrower"), {}, 0.0),
        (Move("flamethrower"), {SideCondition.SPIKES: 1.0}, 0.0),
        (Move("spikes"), {}, 1.0),
        (Move("spikes"), {SideCondition.SPIKES: 1.0}, 0.0),
        (Move("stealthrock"), {}, 1.0),
        (Move("stealthrock"), {SideCondition.STEALTH_ROCK: 1.0}, 0.0),
        (Move("stickyweb"), {}, 1.0),
        (Move("stickyweb"), {SideCondition.STICKY_WEB: 1.0}, 0.0),
        (Move("toxicspikes"), {}, 1.0),
        (Move("toxicspikes"), {SideCondition.TOXIC_SPIKES: 1.0}, 0.0),
        (
            Move("spikes"),
            {
                SideCondition.TOXIC_SPIKES: 1.0,
                SideCondition.STEALTH_ROCK: 1.0,
                SideCondition.STICKY_WEB: 1.0,
            },
            1.0,
        ),
    ],
)
def test_check_hazard_move(
    move: Move, tgt_side_conditions: typing.Dict[SideCondition, int], expected: float
):
    assert move_helpers.check_hazard_move(move, tgt_side_conditions) == expected


@pytest.mark.parametrize(
    "move,usr_side_conditions,expected",
    [
        (Move("lightscreen"), {}, 1.0),
        (Move("lightscreen"), {SideCondition.LIGHT_SCREEN: 1.0}, 0.0),
        (Move("reflect"), {}, 1.0),
        (Move("reflect"), {SideCondition.REFLECT: 1.0}, 0.0),
        (Move("auroraveil"), {}, 1.0),
        (Move("auroraveil"), {SideCondition.AURORA_VEIL: 1.0}, 0.0),
        (Move("tailwind"), {}, 1.0),
        (Move("tailwind"), {SideCondition.TAILWIND: 1.0}, 0.0),
        (
            Move("lightscreen"),
            {
                SideCondition.TOXIC_SPIKES: 1.0,
                SideCondition.STEALTH_ROCK: 1.0,
                SideCondition.STICKY_WEB: 1.0,
                SideCondition.REFLECT: 1.0,
            },
            1.0,
        ),
        (
            Move("flamethrower"),
            {
                SideCondition.TOXIC_SPIKES: 1.0,
                SideCondition.STEALTH_ROCK: 1.0,
                SideCondition.STICKY_WEB: 1.0,
                SideCondition.REFLECT: 1.0,
            },
            0.0,
        ),
        (
            Move("flamethrower"),
            {},
            0.0,
        ),
    ],
)
def test_check_setup_move(
    move: Move, usr_side_conditions: typing.Dict[SideCondition, int], expected: float
):
    assert move_helpers.check_setup_move(move, usr_side_conditions) == expected


@pytest.mark.parametrize(
    "move,usr_side_conditions,expected",
    [
        (Move("rapidspin"), {}, 0.0),
        (Move("rapidspin"), {SideCondition.SPIKES: 1.0}, 1.0),
        (Move("rapidspin"), {}, 0.0),
        (Move("rapidspin"), {SideCondition.LIGHT_SCREEN: 1.0}, 0.0),
        (Move("defog"), {}, 0.0),
        (Move("defog"), {SideCondition.STEALTH_ROCK: 1.0}, 1.0),
        (Move("defog"), {SideCondition.LIGHT_SCREEN: 1.0}, 0.0),
        (Move("flamethrower"), {SideCondition.STEALTH_ROCK: 1.0}, 0.0),
        (Move("flamethrower"), {SideCondition.LIGHT_SCREEN: 1.0}, 0.0),
        (
            Move("spikes"),
            {
                SideCondition.TOXIC_SPIKES: 1.0,
                SideCondition.STEALTH_ROCK: 1.0,
                SideCondition.STICKY_WEB: 1.0,
            },
            0.0,
        ),
    ],
)
def test_check_removal_move(
    move: Move, usr_side_conditions: typing.Dict[SideCondition, int], expected: float
):
    assert move_helpers.check_removal_move(move, usr_side_conditions) == expected


@pytest.mark.parametrize(
    "move,boosts,expected",
    [
        (Move("poweruppunch"), {"atk": 0.0}, 1.0),
        (Move("swordsdance"), {"atk": -1.0}, 1.0),
        (Move("swordsdance"), {"atk": 6.0}, 0.0),
        (Move("screech"), {"atk": 0.0, "def": 0.0, "spa": 0.0, "spd": 0.0, "spe": 0.0}, 0.0),
        (Move("swordsdance"), {"atk": 0.0, "def": 3.0, "spe": 3.0}, 1.0),
    ],
)
def test_check_boost_move(move: Move, boosts: typing.Dict[str, int], expected: float):
    assert move_helpers.check_boost_move(move, boosts) == expected


@pytest.mark.parametrize(
    "tgt,ability,expected",
    [
        (Pokemon(species="charizard"), None, False),
        (Pokemon(species="marill"), "comatose", False),
        (Pokemon(species="marill"), "thermalexchange", False),
        (Pokemon(species="marill"), "waterbubble", False),
        (Pokemon(species="marill"), "waterveil", False),
        (Pokemon(species="marill"), "flareboost", False),
        (Pokemon(species="marill"), "overgrown", True),
        (Pokemon(species="marill"), None, True),
    ],
)
def test_burn_possible(tgt: Pokemon, ability: typing.Optional[str], expected: bool):
    tgt._ability = ability

    assert move_helpers.burn_possible(tgt) == expected


@pytest.mark.parametrize(
    "tgt,weather,ability,expected",
    [
        (Pokemon(species="lapras"), {}, None, False),
        (Pokemon(species="beldum"), {Weather.SUNNYDAY: 1}, None, False),
        (Pokemon(species="beldum"), {Weather.DESOLATELAND: 1}, None, False),
        (Pokemon(species="beldum"), {Weather.RAINDANCE: 1}, None, True),
        (Pokemon(species="beldum"), {}, "comatose", False),
        (Pokemon(species="beldum"), {}, "magmaarmor", False),
        (Pokemon(species="beldum"), {}, None, True),
    ],
)
def test_freeze_possible(
    tgt: Pokemon, weather: typing.Dict[Weather, int], ability: typing.Optional[str], expected: bool
):
    tgt._ability = ability

    assert move_helpers.freeze_possible(tgt, weather) == expected


@pytest.mark.parametrize(
    "move,tgt,ability,expected",
    [
        (Move("thunderwave"), Pokemon(species="rotom"), None, False),
        (Move("thunderwave"), Pokemon(species="quagsire"), None, False),
        (Move("bodyslam"), Pokemon(species="quagsire"), "comatose", False),
        (Move("bodyslam"), Pokemon(species="quagsire"), "limber", False),
        (Move("bodyslam"), Pokemon(species="quagsire"), None, True),
        (Move("bodyslam"), Pokemon(species="spinda"), "limber", False),
        (Move("bodyslam"), Pokemon(species="spinda"), "owntempo", True),
    ],
)
def test_paralysis_possible(
    move: Move, tgt: Pokemon, ability: typing.Optional[str], expected: bool
):
    tgt._ability = ability

    assert move_helpers.paralysis_possible(move, tgt) == expected


@pytest.mark.parametrize(
    "usr,usr_ability,tgt,tgt_ability,expected",
    [
        (Pokemon(species="snorlax"), None, Pokemon(species="ekans"), None, False),
        (Pokemon(species="snorlax"), None, Pokemon(species="beldum"), None, False),
        (Pokemon(species="snorlax"), "corrosion", Pokemon(species="ekans"), None, True),
        (Pokemon(species="snorlax"), "corrosion", Pokemon(species="beldum"), None, True),
        (Pokemon(species="snorlax"), None, Pokemon(species="snorlax"), "comatose", False),
        (Pokemon(species="snorlax"), None, Pokemon(species="snorlax"), "immunity", False),
        (Pokemon(species="snorlax"), None, Pokemon(species="snorlax"), "poisonheal", False),
        (Pokemon(species="snorlax"), None, Pokemon(species="snorlax"), "limber", True),
        (Pokemon(species="snorlax"), "corrosion", Pokemon(species="snorlax"), "comatose", False),
        (Pokemon(species="snorlax"), "corrosion", Pokemon(species="snorlax"), "immunity", False),
        (Pokemon(species="snorlax"), "corrosion", Pokemon(species="snorlax"), "poisonheal", False),
        (Pokemon(species="snorlax"), "corrosion", Pokemon(species="snorlax"), "limber", True),
    ],
)
def test_poison_possible(
    usr: Pokemon,
    usr_ability: typing.Optional[str],
    tgt: Pokemon,
    tgt_ability: typing.Optional[str],
    expected: bool,
):
    usr._ability = usr_ability
    tgt._ability = tgt_ability

    assert move_helpers.poison_possible(usr, tgt) == expected


@pytest.mark.parametrize(
    "ability,field,team,expected",
    [
        (None, {}, [False, False, False], True),
        ("limber", {}, [False, False, False], True),
        ("insomnia", {}, [False, False, False], False),
        ("vital spirit", {}, [False, False, False], False),
        ("limber", {Field.ELECTRIC_TERRAIN: 1}, [False, False, False], False),
        ("limber", {Field.PSYCHIC_TERRAIN: 1}, [False, False, False], True),
        ("limber", {}, [False, False, True], False),
    ],
)
def test_sleep_possible(
    ability: typing.Optional[str],
    field: typing.Dict[Field, int],
    team: typing.List[bool],
    expected: bool,
):
    tgt_team = [Pokemon(species="snorlax") for _ in team]
    for ix, slp in enumerate(team):
        if slp:
            tgt_team[ix]._status = Status.SLP
    assert move_helpers.sleep_possible(ability, tgt_team, field) == expected
