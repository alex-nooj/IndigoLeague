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
    """Will filter moves based on stats and items, then choose 4 unique moves.

    Args:
        pokemon_name: Species of the pokemon
        ability: Pokemon's ability
        item: Pokemon's held item
        evs: List of Pokemon's effort values in stat order
        nature: Pokemon's nature
        moves: Dict of possible moves with {move name, frequency}

    Returns:
        List of move names of length NUM_MOVES
    """
    if format_str(pokemon_name) == "ditto":
        return ["transform"]

    moves = {k: v for k, v in moves.items() if k not in ["", "teleport", "zapcannon"]}
    moves = smart_filter_moves(
        name=pokemon_name, ability=ability, item=item, evs=evs, nature=nature, moves=moves
    )
    return choose_from_dict(moves, NUM_MOVES)


def remove_move_category(
    moves: typing.Dict[str, float], category: MoveCategory
) -> typing.Dict[str, float]:
    """Removes moves from move pool if they have the given category.

    Args:
        moves: Dict of possible moves with {move name, frequency}
        category: Move category to filter by

    Returns:
        The moves Dict but without any moves belonging to the given MoveCategory
    """
    return {k: v for k, v in moves.items() if Move(k).category != category}


def remove_bad_offensive_moves(
    pokemon_name: str, nature: str, evs: typing.List[str], moves: typing.Dict[str, float]
) -> typing.Dict[str, float]:
    """Determines the Pokemon's worst offensive stat, then removes moves that use that stat

    Args:
        pokemon_name: Species of the pokemon
        nature: Pokemon's nature
        evs: List of Pokemon's effort values in stat order
        moves: Dict of possible moves with {move name, frequency}

    Returns:
        The moves Dict but without any moves belonging to the bad offensive stat category
    """
    pokemon = Pokemon(species=pokemon_name)

    atk = NATURES[nature]["atk"] * pokemon.base_stats["atk"] + int(evs[1]) // 4
    spa = NATURES[nature]["spa"] * pokemon.base_stats["spa"] + int(evs[3]) // 4

    if atk > spa:
        return remove_move_category(moves=moves, category=MoveCategory.SPECIAL)
    elif atk < spa:
        return remove_move_category(moves=moves, category=MoveCategory.PHYSICAL)

    return moves


def remove_moves_based_on_item(
    item: str, moves: typing.Dict[str, float]
) -> typing.Dict[str, float]:
    """Checks if the pokemon is holding a specific item then removes moves based on that.

    Assault vest prevents the holder from using status moves, while "choice" items make status moves
    a poor choice (since the pokemon is locked into the move). This function removes these moves
    from the possible move pool.

    Args:
        item: Pokemon's held item
        moves: Dict of possible moves with {move name, frequency}

    Returns:
        The moves Dict but without any status moves if the pokemon has the named items
    """
    if format_str(item) in ["assaultvest", "choicescarf", "choicespecs", "choiceband"]:
        return remove_move_category(moves, MoveCategory.STATUS)
    return moves


def remove_weather_moves(ability: str, moves: typing.Dict[str, float]) -> typing.Dict[str, float]:
    """The moves Dict but without any moves belonging to the bad offensive stat category


    Args:
        ability: Pokemon's ability
        moves: Dict of possible moves with {move name, frequency}

    Returns:
        The moves Dict but without any weather moves if the pokemon has the corresponding ability
    """

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
