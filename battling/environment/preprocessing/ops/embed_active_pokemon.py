import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle
from poke_env.environment import Move
from poke_env.environment import Pokemon
from poke_env.environment import SideCondition
from poke_env.environment import Status
from poke_env.environment import Weather

from battling.environment.preprocessing.op import Op
from utils.damage_helpers import calc_move_damage
from utils.normalize_stats import normalize_stats


class EmbedActivePokemon(Op):
    def __init__(self):
        self._n_features = 2 * (8 + 7 + 4 + 4)

    def embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.Dict[str, npt.NDArray]:
        """Embeds active and opponent active pokemon in the current state.

        Args:
            battle: Current state of the battle (functional)
            state: Current state of the battle (observation)

        Returns:
            Dict[str, NDArray]: The updated observational state.
        """
        stats = normalize_stats(battle.active_pokemon)
        status = self._embed_status(battle.active_pokemon)
        moves = self._embed_moves(
            moves=battle.available_moves,
            usr=battle.active_pokemon,
            tgt=battle.opponent_active_pokemon,
            weather=list(battle.weather.keys())[0] if len(battle.weather) > 0 else None,
            side_conditions=list(battle.opponent_side_conditions.keys()),
        )

        moves_dmg_multiplier = [0, 0, 0, 0]
        for ix, move in enumerate(battle.available_moves):
            if move.type:
                moves_dmg_multiplier[ix] = (
                    move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                    )
                    / 2
                    - 1
                )

        opp_stats = normalize_stats(battle.opponent_active_pokemon)
        opp_status = self._embed_status(battle.opponent_active_pokemon)
        opp_moves = self._embed_moves(
            moves=list(battle.opponent_active_pokemon.moves.values()),
            usr=battle.opponent_active_pokemon,
            tgt=battle.active_pokemon,
            weather=list(battle.weather.keys())[0] if len(battle.weather) > 0 else None,
            side_conditions=list(battle.side_conditions.keys()),
        )

        opp_moves_dmg_multiplier = [0, 0, 0, 0]
        for ix, move in enumerate(battle.opponent_active_pokemon.moves.values()):
            if ix > 3:
                break
            if move.type:
                opp_moves_dmg_multiplier[ix] = (
                    move.type.damage_multiplier(
                        battle.active_pokemon.type_1,
                        battle.active_pokemon.type_2,
                    )
                    / 2
                    - 1
                )

        state["active_pokemon"] = np.asarray(
            stats
            + status
            + moves
            + moves_dmg_multiplier
            + opp_stats
            + opp_status
            + opp_moves
            + opp_moves_dmg_multiplier
        )
        return state

    def _embed_moves(
        self,
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
        move_dmgs = [-1.0 for _ in range(4)]
        for ix, move in enumerate(moves):
            move_dmgs[ix] = calc_move_damage(
                move=move,
                usr=usr,
                tgt=tgt,
                weather=weather,
                side_conditions=side_conditions,
            )
        return move_dmgs

    def _embed_status(self, pokemon: Pokemon) -> typing.List[float]:
        status = [0.0 for _ in range(len(Status))]
        if pokemon.status is not None:
            status[int(pokemon.status.value - 1)] = 1.0
        return status

    def _embed_types(self, pokemon: Pokemon) -> typing.List[float]:
        poke_types = [0.0 for _ in range(18)]
        poke_types[pokemon.type_1.value - 1] = 1.0
        if pokemon.type_2 is not None:
            poke_types[pokemon.type_2.value - 1] = 1.0
        return poke_types

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                "active_pokemon": gym.spaces.Box(
                    np.asarray([-1 for _ in range(self._n_features)]),
                    np.asarray([1 for _ in range(self._n_features)]),
                    dtype=np.float32,
                )
            }
        )
