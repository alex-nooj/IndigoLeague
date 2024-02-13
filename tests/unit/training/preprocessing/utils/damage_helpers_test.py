import typing
from unittest.mock import MagicMock

import pytest
from poke_env.environment import Move
from poke_env.environment import MoveCategory
from poke_env.environment import Pokemon
from poke_env.environment import PokemonType
from poke_env.environment import SideCondition
from poke_env.environment import Status
from poke_env.environment import Weather

from indigo_league.training.preprocessing.utils import damage_helpers


def test_attack_defense_ratio():
    assert (
        damage_helpers.attack_defense_ratio(
            MoveCategory.STATUS,
            MoveCategory.PHYSICAL,
            Pokemon(species="shuckle"),
            Pokemon(species="marill"),
        )
        == 0.0
    )


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


@pytest.mark.parametrize(
    "species,ability,move,expected",
    [
        ("Charizard", "blaze", "flamethrower", 1.5),
        ("Charizard", "blaze", "airslash", 1.5),
        ("Charizard", "blaze", "dragonbreath", 1.0),
        ("Charizard", "Adaptability", "flamethrower", 2.0),
        ("Charizard", "Adaptability", "airslash", 2.0),
        ("Charizard", "Adaptability", "dragonbreath", 1.0),
    ],
)
def test_stab_multiplier(species: str, ability: str, move: str, expected: float):
    pkm = Pokemon(species=species)
    pkm._ability = ability

    assert damage_helpers.stab_multiplier(pkm, Move(move)) == expected


@pytest.mark.parametrize(
    "move_type,tgt_ability,expected",
    [
        (PokemonType.WATER, "Dry Skin", 0.0),
        (PokemonType.WATER, "Storm Drain", 0.0),
        (PokemonType.WATER, "Water Absorb", 0.0),
        (PokemonType.WATER, "Blaze", 1.0),
        (PokemonType.FIRE, "Dry Skin", 2.0),
        (PokemonType.FIRE, "Flash Fire", 0.0),
        (PokemonType.FIRE, "Blaze", 1.0),
        (PokemonType.ELECTRIC, "Lightning Rod", 0.0),
        (PokemonType.ELECTRIC, "Motor Drive", 0.0),
        (PokemonType.ELECTRIC, "Volt Absorb", 0.0),
        (PokemonType.ELECTRIC, "Blaze", 1.0),
        (PokemonType.GRASS, "Sap Sipper", 0.0),
        (PokemonType.GRASS, "Blaze", 1.0),
        (PokemonType.GROUND, "Levitate", 0.0),
        (PokemonType.GROUND, "Blaze", 1.0),
        (PokemonType.ICE, "Dry Skin", 1.0),
        (PokemonType.ICE, "Storm Drain", 1.0),
        (PokemonType.ICE, "Water Absorb", 1.0),
        (PokemonType.ICE, "Flash Fire", 1.0),
        (PokemonType.ICE, "Lightning Rod", 1.0),
        (PokemonType.ICE, "Motor Drive", 1.0),
        (PokemonType.ICE, "Volt Absorb", 1.0),
        (PokemonType.ICE, "Sap Sipper", 1.0),
        (PokemonType.ICE, "Levitate", 1.0),
        (PokemonType.ICE, "Blaze", 1.0),
        (PokemonType.ICE, None, 1.0),
    ],
)
def test_ability_immunities(
    move_type: PokemonType, tgt_ability: typing.Optional[str], expected: float
):
    assert (
        damage_helpers.ability_immunities(move_type=move_type, tgt_ability=tgt_ability)
        == expected
    )


@pytest.mark.parametrize(
    "move_id,move_type,tgt,expected",
    [
        ("freeze dry", PokemonType.WATER, Pokemon(species="Ludicolo"), 2.0),
        ("freeze dry", PokemonType.WATER, Pokemon(species="Pikachu"), 1.0),
        ("flamethrower", PokemonType.FIRE, Pokemon(species="Scizor"), 4.0),
        ("flamethrower", PokemonType.FIRE, Pokemon(species="Bellsprout"), 2.0),
        ("flamethrower", PokemonType.FIRE, Pokemon(species="Slaking"), 1.0),
        ("flamethrower", PokemonType.FIRE, Pokemon(species="Golem"), 0.5),
        ("flamethrower", PokemonType.FIRE, Pokemon(species="Relicanth"), 0.25),
        ("mach punch", PokemonType.FIGHTING, Pokemon(species="Gastly"), 0.0),
    ],
)
def test_type_multiplier(
    move_id: str, move_type: PokemonType, tgt: Pokemon, expected: float
):
    assert (
        damage_helpers.type_multiplier(move_id=move_id, move_type=move_type, tgt=tgt)
        == expected
    )


@pytest.mark.parametrize(
    "move_type,weather,usr_ability,tgt_ability,expected",
    [
        (PokemonType.FIRE, Weather.RAINDANCE, "blaze", "blaze", 0.5),
        (PokemonType.WATER, Weather.RAINDANCE, "waterabsorb", "blaze", 1.5),
        (PokemonType.GRASS, Weather.RAINDANCE, "waterabsorb", "blaze", 1.0),
        (PokemonType.FIRE, Weather.SUNNYDAY, "blaze", "blaze", 1.5),
        (PokemonType.WATER, Weather.SUNNYDAY, "waterabsorb", "blaze", 0.5),
        (PokemonType.GRASS, Weather.SUNNYDAY, "waterabsorb", "blaze", 1.0),
        (PokemonType.GRASS, None, "waterabsorb", "blaze", 1.0),
        (PokemonType.WATER, Weather.RAINDANCE, "cloudnine", "blaze", 1.0),
        (PokemonType.WATER, Weather.RAINDANCE, "airlock", "blaze", 1.0),
        (PokemonType.WATER, Weather.RAINDANCE, "blaze", "cloudnine", 1.0),
        (PokemonType.WATER, Weather.RAINDANCE, "blaze", "airlock", 1.0),
        (PokemonType.WATER, Weather.RAINDANCE, "blaze", "None", 1.5),
    ],
)
def test_weather_multiplier(
    move_type: PokemonType,
    weather: Weather,
    usr_ability: str,
    tgt_ability: str,
    expected: float,
):
    assert (
        damage_helpers.weather_multiplier(
            move_type=move_type,
            weather=weather,
            usr_ability=usr_ability,
            tgt_ability=tgt_ability,
        )
        == expected
    )


@pytest.mark.parametrize(
    "item,expected", [("lifeorb", 5324 / 4096), ("leftovers", 1.0), (None, 1.0)]
)
def test_item_multiplier(item: typing.Optional[str], expected: float):
    assert damage_helpers.item_multiplier(item=item) == expected


@pytest.mark.parametrize(
    "item,move,expected",
    [
        (None, Move("flamethrower"), 1.0),
        ("leftovers", Move("earthquake"), 1.0),
        ("airballoon", Move("earthquake"), 0.0),
        ("air balloon", Move("earthquake"), 0.0),
        ("airballoon", Move("flamethrower"), 1.0),
    ],
)
def test_opponent_item_multiplier(
    item: typing.Optional[str], move: Move, expected: float
):
    assert damage_helpers.opponent_item_multiplier(item=item, move=move) == expected


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

    @pytest.mark.parametrize(
        "usr,tgt,move",
        [(Pokemon(species="alakazam"), Pokemon(species="tyranitar"), Move("psychic"))],
    )
    def test_immunities(self, usr: Pokemon, tgt: Pokemon, move: Move):
        tgt._current_hp = 128.0
        assert damage_helpers.calc_move_damage(move, usr, tgt, None, None) == 0.0

    def test_zero_power(self):
        assert (
            damage_helpers.calc_move_damage(
                Move("lightscreen"),
                Pokemon(species="alakazam"),
                Pokemon(species="tyranitar"),
                None,
                None,
            )
            == -1.0
        )


@pytest.mark.parametrize("n_moves", [1, 2, 3, 4, 5])
def test_embed_moves(n_moves: int):
    moves = [Move("flamethrower") for _ in range(n_moves)]
    usr = Pokemon(species="charizard")
    tgt = Pokemon(species="porygon")

    result = len(damage_helpers.embed_moves(moves, usr, tgt, None, []))
    expected = 4 * len(damage_helpers.embed_move(moves[0], usr, tgt, None, []))

    assert result == expected


@pytest.mark.parametrize("current_pp", [0, 1, Move("flamethrower").max_pp])
def test_embed_move(current_pp: int):
    move = Move("flamethrower")
    move._current_pp = current_pp

    usr = Pokemon(species="charizard")
    tgt = Pokemon(species="blastoise")

    result = damage_helpers.embed_move(move, usr, tgt, None, [])

    assert result[-1] == current_pp / move.max_pp


def test_embed_move_no_pp():
    move = Move("flamethrower")
    move.entry["pp"] = 0

    usr = Pokemon(species="charizard")
    tgt = Pokemon(species="blastoise")

    result = damage_helpers.embed_move(move, usr, tgt, None, [])

    assert result[-1] == 0.0
