import typing

import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle
from poke_env.environment import Effect
from poke_env.environment import Move
from poke_env.environment import Pokemon
from poke_env.environment import Status

from indigo_league.utils.constants import NUM_MOVES
from indigo_league.utils.constants import NUM_POKEMON
from indigo_league.utils.str_helpers import format_str


def moves_mask(
    active_pokemon: typing.Optional[Pokemon], available_moves: typing.List[Move]
) -> npt.NDArray:
    """Creates an encoding of valid moves to select by index.

    Args:
        active_pokemon: The current pokemon on the field or None if there isn't one
        available_moves: List of moves that the active pokemon can choose from

    Returns:
        A numpy array where a "1" at index i means available_moves[i] is a valid move choice.
    """
    moves = np.zeros(NUM_MOVES)
    if active_pokemon is None:
        return moves

    for ix, move in enumerate(active_pokemon.moves.values()):
        if move.id in [m.id for m in available_moves]:
            if format_str(move.id) == "substitute" and Effect.SUBSTITUTE in active_pokemon.effects:
                moves[ix] = 0.0
            else:
                moves[ix] = 1.0 if move.current_pp != 0 else 0.0
    return moves


def switch_mask(
    available_switches: typing.List[Pokemon], team_mons: typing.List[Pokemon]
) -> npt.NDArray:
    """Creates an encoding of valid pokemon to switch into by index.

    Args:
        available_switches: List of switches deemed available by the environment
        team_mons: List of pokemon on the team

    Returns:
        A numpy array where a "1" at index i means team_mons[i] is a valid choice.
    """
    team = np.zeros(NUM_POKEMON)
    if len(available_switches) == 0:
        return team

    for ix, mon in enumerate(team_mons):
        team[ix] = float(mon.status != Status.FNT and not mon.active)

    return team


def action_masks(battle: typing.Optional[AbstractBattle]) -> npt.NDArray:
    """Creates a mask for MaskablePPO to determine which actions are valid.

    Args:
        battle: The currently active battle, provided by the environment.

    Returns:
        Numpy array with a 1 at index i if that action is valid, or 0 if that
        action is invalid.
    """
    if battle is None:
        return np.zeros(NUM_MOVES + NUM_POKEMON)
    moves = moves_mask(battle.active_pokemon, battle.available_moves)
    team = switch_mask(battle.available_switches, list(battle.team.values()))
    return np.concatenate([moves, team])
