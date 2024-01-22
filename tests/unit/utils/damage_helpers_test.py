import typing

import pytest
from poke_env.environment import Move
from poke_env.environment import MoveCategory
from poke_env.environment import Pokemon
from poke_env.environment import SideCondition
from poke_env.environment import Status
from poke_env.environment import Weather

from utils import damage_helpers


def test_attack_defense_ratio():
    # Move category not Physical or Special
    # Move category physical, defense category physical
    # Move category physical, defense category special
    # Move category special, defense category physical
    # Move category special, defense category special
    pass


@pytest.mark.parametrize(
    "move_category,status,ability,expected",
    [
        (MoveCategory.PHYSICAL, Status.PSN, "blaze", 1.0),
        (MoveCategory.SPECIAL, Status.PSN, "blaze", 1.0),
        (MoveCategory.PHYSICAL, None, "blaze", 1.0),
        (MoveCategory.SPECIAL, None, "blaze", 1.0),
        (MoveCategory.PHYSICAL, None, None, 1.0),
        (MoveCategory.SPECIAL, None, None, 1.0),
        (MoveCategory.PHYSICAL, Status.BRN, "guts", 1.0),
        (MoveCategory.SPECIAL, Status.BRN, "guts", 1.0),
        (MoveCategory.PHYSICAL, Status.BRN, "guts", 1.0),
        (MoveCategory.SPECIAL, Status.BRN, "guts", 1.0),
        (MoveCategory.PHYSICAL, Status.BRN, "blaze", 0.5),
        (MoveCategory.SPECIAL, Status.BRN, "blaze", 1.0),
        (MoveCategory.PHYSICAL, Status.BRN, "GUTS", 1.0),
    ],
)
def test_burn_multiplier(
    move_category: MoveCategory,
    status: typing.Optional[Status],
    ability: str,
    expected: float,
):
    burn_mult = damage_helpers.burn_multiplier(move_category, status, ability)
    assert type(burn_mult) is float
    assert burn_mult == expected


@pytest.mark.parametrize("level,expected", [(50, 22.0), (100, 42.0)])
def test_level_multiplier(level: int, expected: float):
    assert damage_helpers.level_multiplier(level) == expected


@pytest.mark.parametrize(
    "move_category,side_conditions,expected",
    [
        (MoveCategory.PHYSICAL, [SideCondition.AURORA_VEIL], 0.5),
        (MoveCategory.SPECIAL, [SideCondition.AURORA_VEIL], 0.5),
        (MoveCategory.PHYSICAL, [SideCondition.LIGHT_SCREEN], 1.0),
        (MoveCategory.SPECIAL, [SideCondition.LIGHT_SCREEN], 0.5),
        (MoveCategory.PHYSICAL, [SideCondition.REFLECT], 0.5),
        (MoveCategory.SPECIAL, [SideCondition.REFLECT], 1.0),
        (
            MoveCategory.PHYSICAL,
            [SideCondition.REFLECT, SideCondition.LIGHT_SCREEN],
            0.5,
        ),
        (
            MoveCategory.SPECIAL,
            [SideCondition.REFLECT, SideCondition.LIGHT_SCREEN],
            0.5,
        ),
    ],
)
def test_screens_multiplier(
    move_category: MoveCategory,
    side_conditions: typing.List[SideCondition],
    expected: float,
):
    assert damage_helpers.screens_multiplier(move_category, side_conditions) == expected


@pytest.mark.parametrize(
    "move_id,usr_ability,tgt_ability,expected",
    [
        ("vine whip", "soundproof", "punkrock", 1.0),
        ("overdrive", "blaze", "blaze", 1.0),
        ("boomburst", "blaze", "blaze", 1.0),
        ("overdrive", "punk rock", "blaze", 1.3),
        ("boomburst", "punk rock", "blaze", 1.3),
        ("overdrive", "blaze", "punk rock", 0.5),
        ("boomburst", "blaze", "punk rock", 0.5),
        ("overdrive", "punk rock", "punk rock", 0.65),
        ("boomburst", "punk rock", "punk rock", 0.65),
        ("overdrive", "punk rock", "soundproof", 0.0),
        ("boomburst", "punk rock", "soundproof", 0.0),
        ("OVERDRIVE", "punk-Rock", "punk ROck", 0.65),
    ],
)
def test_sound_multiplier(
    move_id: str, usr_ability: str, tgt_ability: typing.Optional[str], expected: float
):
    assert (
        damage_helpers.sound_multiplier(move_id, usr_ability, tgt_ability) == expected
    )


@pytest.mark.parametrize(
    "move_category,tgt_ability,expected",
    [
        (MoveCategory.SPECIAL, "icescales", 0.5),
        (MoveCategory.SPECIAL, "ice -Scales", 0.5),
        (MoveCategory.PHYSICAL, "icescales", 1.0),
        (MoveCategory.PHYSICAL, "blaze", 1.0),
        (MoveCategory.PHYSICAL, None, 1.0),
    ],
)
def test_icescales_multiplier(
    move_category: MoveCategory, tgt_ability: typing.Optional[str], expected: float
):
    assert damage_helpers.icescales_multiplier(move_category, tgt_ability) == expected


@pytest.mark.parametrize(
    "damage,hp,expected",
    [
        (100.0, 100.0, 1.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 1.0),
        (200.0, 100.0, 1.0),
    ],
)
def test_normalize_damage(damage: float, hp: float, expected: float):
    assert damage_helpers.normalize_damage(damage, hp) == expected


def test_stab_multiplier():
    pass


def test_ability_immunities():
    pass


def test_type_multiplier():
    pass


def test_weather_multiplier():
    pass


def test_item_multiplier():
    pass


def test_opponent_item_multiplier():
    pass


class TestCalcMoveDamage:
    def test_earthquake(self):
        move = Move("earthquake")
        usr = Pokemon(species="groudon")
        tgt = Pokemon(species="articuno")
        tgt._current_hp = 128.0
        assert damage_helpers.calc_move_damage(move, usr, tgt, None, None) == 0.0

    def test_flash_fire(self):
        move = Move("magmastorm")
        usr = Pokemon(species="heatran")
        tgt = Pokemon(species="heatran")
        tgt._current_hp = 128.0
        tgt._ability = "flashfire"
        assert damage_helpers.calc_move_damage(move, usr, tgt, None, None) == 0.0


def test_embed_moves():
    pass


def test_embed_move():
    pass
