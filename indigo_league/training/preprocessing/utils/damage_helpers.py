import typing

from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.environment.weather import Weather

from indigo_league.training.preprocessing.utils.normalize_stats import stat_estimation
from indigo_league.utils.constants import NUM_MOVES
from indigo_league.utils.str_helpers import format_str


def attack_defense_ratio(
    move_category: MoveCategory,
    defensive_category: MoveCategory,
    usr: Pokemon,
    tgt: Pokemon,
) -> float:
    if move_category == MoveCategory.PHYSICAL:
        attack = stat_estimation(usr, "atk")
    elif move_category == MoveCategory.SPECIAL:
        attack = stat_estimation(usr, "spa")
    else:
        return 0.0

    if defensive_category == MoveCategory.PHYSICAL:
        defense = stat_estimation(tgt, "def")
    else:
        defense = stat_estimation(tgt, "spd")

    return attack / defense


def burn_multiplier(
    move_category: MoveCategory, status: typing.Optional[Status], ability: str
) -> float:
    return (
        0.5
        if status == Status.BRN
        and move_category == MoveCategory.PHYSICAL
        and ability is not None
        and format_str(ability) != "guts"
        else 1.0
    )


def level_multiplier(level: int) -> float:
    return 2 * level / 5.0 + 2


def screens_multiplier(
    move_category: MoveCategory, side_conditions: typing.List[SideCondition]
) -> float:
    if SideCondition.AURORA_VEIL in side_conditions:
        return 0.5
    elif (
        SideCondition.LIGHT_SCREEN in side_conditions
        and move_category == MoveCategory.SPECIAL
    ):
        return 0.5
    elif (
        SideCondition.REFLECT in side_conditions
        and move_category == MoveCategory.PHYSICAL
    ):
        return 0.5
    else:
        return 1.0


def sound_multiplier(
    move_id: str, usr_ability: str, tgt_ability: typing.Optional[str]
) -> float:
    multiplier = 1.0
    if format_str(move_id) in ["overdrive", "boomburst"]:
        if tgt_ability is not None and format_str(tgt_ability) == "punkrock":
            multiplier *= 0.5
        elif tgt_ability is not None and format_str(tgt_ability) == "soundproof":
            return 0.0

        if usr_ability is not None and format_str(usr_ability) == "punkrock":
            multiplier *= 1.3
    return multiplier


def icescales_multiplier(
    move_category: MoveCategory, tgt_ability: typing.Optional[str]
) -> float:
    if (
        tgt_ability is not None
        and format_str(tgt_ability) == "icescales"
        and move_category == MoveCategory.SPECIAL
    ):
        return 0.5
    else:
        return 1.0


def normalize_damage(damage: float, hp: float) -> float:
    return 1 - max((hp - damage), 0) / hp if hp != 0.0 else 1.0


def stab_multiplier(usr: Pokemon, move: Move) -> float:
    if usr.type_1 == move.type or usr.type_2 == move.type:
        if usr.ability is not None and format_str(usr.ability) == "adaptability":
            return 2.0
        else:
            return 1.5
    else:
        return 1.0


def ability_immunities(
    move_type: PokemonType, tgt_ability: typing.Optional[str]
) -> float:
    if tgt_ability is None:
        return 1.0
    else:
        ability = format_str(tgt_ability)

    if move_type == PokemonType.WATER:
        if ability in ["dryskin", "stormdrain", "waterabsorb"]:
            return 0.0
    elif move_type == PokemonType.FIRE:
        if ability == "dryskin":
            return 2.0
        elif ability == "flashfire":
            return 0.0
    elif move_type == PokemonType.ELECTRIC:
        if ability in ["lightningrod", "motordrive", "voltabsorb"]:
            return 0.0
    elif move_type == PokemonType.GRASS:
        if ability == "sapsipper":
            return 0.0
    elif move_type == PokemonType.GROUND:
        if ability == "levitate":
            return 0.0
    return 1.0


def type_multiplier(move_id: str, move_type: PokemonType, tgt: Pokemon) -> float:
    if format_str(move_id) == "freezedry" and PokemonType.WATER in [
        tgt.type_1,
        tgt.type_2,
    ]:
        return 2.0
    else:
        return tgt.damage_multiplier(move_type)


def weather_multiplier(
    move_type: PokemonType,
    weather: typing.Union[Weather, None],
    usr_ability: str,
    tgt_ability: typing.Union[str, None],
) -> float:
    if (
        weather is None
        or (
            usr_ability is not None
            and format_str(usr_ability) in ["cloudnine", "airlock"]
        )
        or (
            tgt_ability is not None
            and format_str(tgt_ability) in ["cloudnine", "airlock"]
        )
    ):
        return 1.0
    elif weather == Weather.RAINDANCE:
        if move_type == PokemonType.FIRE:
            return 0.5
        elif move_type == PokemonType.WATER:
            return 1.5
    elif weather == Weather.SUNNYDAY:
        if move_type == PokemonType.FIRE:
            return 1.5
        elif move_type == PokemonType.WATER:
            return 0.5
    return 1.0


def item_multiplier(item: typing.Union[str, None]) -> float:
    if item is None:
        return 1.0
    elif format_str(item) == "lifeorb":
        return 5324 / 4096
    else:
        return 1.0


def opponent_item_multiplier(item: typing.Optional[str], move: Move) -> float:
    if (
        item is not None
        and format_str(item) == "airballoon"
        and move.type == PokemonType.GROUND
    ):
        return 0.0
    return 1.0


def calc_move_damage(
    move: Move,
    usr: Pokemon,
    tgt: Pokemon,
    weather: typing.Optional[Weather] = None,
    side_conditions: typing.Optional[typing.List[SideCondition]] = None,
) -> float:
    if side_conditions is None:
        side_conditions = []
    lvl_mult = level_multiplier(usr.level)
    power = move.base_power
    if power == 0:
        return -1.0
    ad_ratio = attack_defense_ratio(move.category, move.defensive_category, usr, tgt)
    weather_mult = weather_multiplier(
        move.type,
        weather,
        usr.ability,
        tgt.ability,
    )
    stab = stab_multiplier(usr, move)
    burn_mult = burn_multiplier(move.category, usr.status, usr.ability)

    screens_mult = screens_multiplier(move.category, side_conditions)

    sound_mult = sound_multiplier(move.id, usr.ability, tgt.ability)

    icescales_mult = icescales_multiplier(move.category, tgt.ability)

    ability_mult = ability_immunities(move.type, tgt.ability)

    type_mult = type_multiplier(move.id, move.type, tgt)

    item_mult = item_multiplier(usr.item)
    # opponent_item_mult = opponent_item_multiplier(tgt.item, move)
    damage = (
        ((lvl_mult * power * ad_ratio) / 50 + 2)
        * weather_mult
        * stab
        * type_mult
        * burn_mult
        * screens_mult
        * sound_mult
        * icescales_mult
        * ability_mult
        * item_mult
    )
    dmg_normed = normalize_damage(damage, tgt.current_hp)
    return dmg_normed


def embed_moves(
    moves: typing.List[Move],
    usr: Pokemon,
    tgt: Pokemon,
    weather: typing.Union[typing.List[Weather], None],
    side_conditions: typing.List[SideCondition],
) -> typing.List[float]:
    """Gets the damage estimates for each move

    Args:
        moves:
        usr:
        tgt:
        weather:
        side_conditions:

    Returns:

    """
    move_embedding = []

    if len(moves) > NUM_MOVES:
        moves = moves[:NUM_MOVES]

    for move in moves:
        move_embedding += embed_move(move, usr, tgt, weather, side_conditions)

    if len(moves) < NUM_MOVES:
        for _ in range(NUM_MOVES - len(moves)):
            move_embedding += [0.0 for _ in range(2 + 1 + 1 + 1)]
    return move_embedding


def embed_move(
    move: Move,
    usr: Pokemon,
    tgt: Pokemon,
    weather: typing.Union[typing.List[Weather], None],
    side_conditions: typing.List[SideCondition],
) -> typing.List[float]:
    if move.max_pp == 0:
        pp_ratio = 0.0
    else:
        pp_ratio = min(max((move.current_pp / move.max_pp, 0.0)), move.max_pp)
    return [
        type_multiplier("", poke_type, usr) / 4.0 if poke_type else -1
        for poke_type in tgt.types
    ] + [
        move.base_power,
        calc_move_damage(
            move=move,
            usr=usr,
            tgt=tgt,
            weather=weather,
            side_conditions=side_conditions,
        ),
        pp_ratio,
    ]
