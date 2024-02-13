import typing

from poke_env import NATURES
from poke_env.environment import Move
from poke_env.environment import MoveCategory
from poke_env.environment import Pokemon

from indigo_league.utils.choose_from_dict import choose_from_dict
from indigo_league.utils.constants import NUM_MOVES
from indigo_league.utils.str_helpers import format_str


def safe_sample_moves(
    pokemon_name: str, ability: str, item: str, evs: typing.List[str], nature: str, moves: typing.Dict[str, float]
) -> typing.List[str]:
    moves = {k: v for k, v in moves.items() if k not in ["", "teleport", "zapcannon"]}
    moves = smart_filter_moves(
        name=pokemon_name, ability=ability, item=item, evs=evs, nature=nature, moves=moves
    )
    if pokemon_name.lower() == "ditto":
        return ["transform"]
    else:
        return choose_from_dict(moves, NUM_MOVES)


def remove_move_category(
    moves: typing.Dict[str, float], category: MoveCategory
) -> typing.Dict[str, float]:
    return {k: v for k, v in moves.items() if Move(k).category != category}


def remove_bad_offensive_moves(
    pokemon_name: str, nature: str, evs: typing.List[str], moves: typing.Dict[str, float]
) -> typing.Dict[str, float]:
    pokemon = Pokemon(species=pokemon_name)

    atk = NATURES[nature]["atk"] * pokemon.base_stats["atk"] + int(evs[1]) // 4
    spa = NATURES[nature]["spa"] * pokemon.base_stats["spa"] + int(evs[3]) // 4

    if atk > spa:
        return remove_move_category(moves=moves, category=MoveCategory.SPECIAL)
    elif spa < atk:
        return remove_move_category(moves=moves, category=MoveCategory.PHYSICAL)

    return moves


def remove_moves_based_on_item(
    item: str, moves: typing.Dict[str, float]
) -> typing.Dict[str, float]:
    if format_str(item) in ["assaultvest", "choicescarf", "choicespecs", "choiceband"]:
        return remove_move_category(moves, MoveCategory.STATUS)
    return moves


def remove_weather_moves(ability: str, moves: typing.Dict[str, float]) -> typing.Dict[str, float]:
    # If ability causes weather, don't need a move to do it
    if format_str(ability) == "snowwarning" and "hail" in moves:
        del moves["hail"]
    elif format_str(ability) == "drought" and "sunnyday" in moves:
        del moves["sunnyday"]
    elif format_str(ability) == "drizzle" and "raindance" in moves:
        del moves["raindance"]
    elif format_str(ability) == "sandstream" and "sandstorm" in moves:
        del moves["sandstorm"]

    return moves


def smart_filter_moves(
    name: str,
    ability: str,
    item: str,
    evs: typing.List[str],
    nature: str,
    moves: typing.Dict[str, float],
) -> typing.Dict[str, float]:
    """Narrows down the move list based on stats and held-item

    Args:
        name: Species name of the pokemon
        ability: Pokemon's ability
        item: Name of the pokemon's held item
        evs: List of effort values for the pokemon
        nature: The Pokemon's nature
        moves: Full possible move list (from Smogon)

    Returns:
        The reduced move list
    """
    moves = {k: v for k, v in moves.items() if k not in ["", "teleport", "zapcannon"]}

    # Remove moves that don't mesh with our offensive category
    moves = remove_bad_offensive_moves(pokemon_name=name, nature=nature, evs=evs, moves=moves)

    # Remove moves based on the held item
    moves = remove_moves_based_on_item(item=item, moves=moves)

    moves = remove_weather_moves(ability, moves)

    return moves
