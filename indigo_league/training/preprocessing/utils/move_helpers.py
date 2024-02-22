import typing

from poke_env.environment import Field
from poke_env.environment import Move
from poke_env.environment import Pokemon
from poke_env.environment import PokemonType
from poke_env.environment import SideCondition
from poke_env.environment import Status
from poke_env.environment import Weather

from indigo_league.utils.str_helpers import format_str

ENTRY_HAZARDS = {
    "spikes": SideCondition.SPIKES,
    "stealthrock": SideCondition.STEALTH_ROCK,
    "stickyweb": SideCondition.STICKY_WEB,
    "toxicspikes": SideCondition.TOXIC_SPIKES,
}

SETUP_MOVES = {
    "lightscreen": SideCondition.LIGHT_SCREEN,
    "reflect": SideCondition.REFLECT,
    "auroraveil": SideCondition.AURORA_VEIL,
    "tailwind": SideCondition.TAILWIND,
}

ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}


def check_hazard_move(
    move: Move, tgt_side_conditions: typing.Dict[SideCondition, int]
) -> float:
    return float(
        move.id in ENTRY_HAZARDS and ENTRY_HAZARDS[move.id] not in tgt_side_conditions
    )


def check_setup_move(
    move: Move, usr_side_conditions: typing.Dict[SideCondition, int]
) -> float:
    return float(
        move.id in SETUP_MOVES and SETUP_MOVES[move.id] not in usr_side_conditions
    )


def check_removal_move(
    move: Move, usr_side_conditions: typing.Dict[SideCondition, int]
) -> float:
    return float(
        move.id in ANTI_HAZARDS_MOVES
        and any(k in usr_side_conditions for k in ENTRY_HAZARDS.values())
    )


def check_boost_move(move: Move, boosts: typing.Dict[str, int]) -> float:
    if (
        move.boosts is not None
        and sum(move.boosts.values()) >= 2.0
        and move.target == "self"
        and min([boosts[s] for s, v in move.boosts.items() if v > 0]) < 6
    ):
        return 1.0

    if (
        move.self_boost is not None
        and sum(move.self_boost.values()) >= 2.0
        and min([boosts[s] for s, v in move.boosts.items() if v > 0]) < 6
    ):
        return 1.0

    for effect in move.secondary:
        if "self" in effect and "boosts" in effect["self"]:
            if (
                min([boosts[s] for s, v in effect["self"]["boosts"].items() if v > 0])
                < 6
            ):
                return 1.0

    return 0.0


def check_status_move(
    move: Move,
    usr: Pokemon,
    tgt: Pokemon,
    tgt_team: typing.List[Pokemon],
    field: typing.Dict[Field, int],
    weather: typing.Dict[Weather, int],
) -> float:
    # This is not a status inflicting move, or the target already has a status
    if move.status is None or tgt.status is not None:
        return 0.0

    # Next, we check each status' immune conditions
    if move.status == Status.BRN and not burn_possible(tgt):
        return 0.0
    if move.status == Status.FRZ and not freeze_possible(tgt, weather):
        return 0.0
    if move.status == Status.PAR and not paralysis_possible(move, tgt):
        return 0.0
    if move.status in [Status.PSN, Status.TOX] and not poison_possible(usr, tgt):
        return 0.0
    if move.status == Status.SLP and not sleep_possible(tgt.ability, tgt_team, field):
        return 0.0

    # The move can inflict a status
    return 1.0


def burn_possible(tgt: Pokemon) -> bool:
    if PokemonType.FIRE in tgt.types:
        return False

    if tgt.ability is not None and format_str(tgt.ability) in [
        "comatose",
        "thermalexchange",
        "waterbubble",
        "waterveil",
        "flareboost",
    ]:
        return False

    return True


def freeze_possible(tgt: Pokemon, weather: typing.Dict[Weather, int]) -> bool:
    if (
        PokemonType.ICE in tgt.types
        or Weather.SUNNYDAY in weather
        or Weather.DESOLATELAND in weather
    ):
        return False

    if tgt.ability is not None and format_str(tgt.ability) in [
        "comatose",
        "magmaarmor",
    ]:
        return False

    return True


def paralysis_possible(move: Move, tgt: Pokemon) -> bool:
    if PokemonType.ELECTRIC in tgt.types:
        return False

    if move.type == PokemonType.ELECTRIC and PokemonType.GROUND in tgt.types:
        return False

    if tgt.ability is not None and format_str(tgt.ability) in ["comatose", "limber"]:
        return False

    return True


def poison_possible(usr: Pokemon, tgt: Pokemon) -> bool:
    if (PokemonType.POISON in tgt.types or PokemonType.STEEL in tgt.types) and (
        usr.ability is None or format_str(usr.ability) != "corrosion"
    ):
        return False

    if tgt.ability is not None and format_str(tgt.ability) in [
        "comatose",
        "immunity",
        "poisonheal",
    ]:
        return False

    return True


def sleep_possible(
    ability: typing.Optional[str],
    tgt_team: typing.List[Pokemon],
    field: typing.Dict[Field, int],
) -> bool:
    if Field.ELECTRIC_TERRAIN in field:
        return False

    if ability is not None and format_str(ability) in ["insomnia", "vitalspirit"]:
        return False

    if any(mon.status == Status.SLP for mon in tgt_team):
        return False

    return True
