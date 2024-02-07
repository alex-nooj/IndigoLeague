import typing

import gym
import numpy as np
from numpy import typing as npt
from poke_env.environment import AbstractBattle

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import calc_move_damage
from indigo_league.utils.constants import NUM_MOVES
from indigo_league.utils.constants import NUM_POKEMON


class SimpleOp(Op):
    def __init__(self, seq_len: int):
        low = [-1, -1, -1, -1, 0, 0]
        high = [1, 1, 1, 1, 1, 1]
        super().__init__(seq_len=seq_len, n_features=len(low), key="SimpleOp")
        self.low = []
        self.high = []
        for _ in range(self.seq_len):
            self.low += low
            self.high += high

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        moves_base_power = [-1.0 for _ in range(NUM_MOVES)]
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = calc_move_damage(
                move=move,
                usr=battle.active_pokemon,
                tgt=battle.opponent_active_pokemon,
                weather=list(battle.weather.keys())[0]
                if len(battle.weather) > 0
                else None,
                side_conditions=list(battle.opponent_side_conditions.keys()),
            )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / NUM_POKEMON
        )
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted])
            / NUM_POKEMON
        )

        # Final vector with 10 components
        final_vector = moves_base_power + [fainted_mon_team, fainted_mon_opponent]

        return final_vector

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """

        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.array(self.low, dtype=np.float32),
                    np.array(self.high, dtype=np.float32),
                    dtype=np.float32,
                )
            }
        )
