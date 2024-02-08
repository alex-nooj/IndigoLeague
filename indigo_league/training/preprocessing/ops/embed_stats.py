import typing

import gym
import numpy as np
from numpy import typing as npt
from poke_env.environment import AbstractBattle

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import normalize_stats


class EmbedStats(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len,
            n_features=2 * 8 - 1,
            key=__name__,
        )

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        active_stats = normalize_stats(active)[1:]  # Own HP is handled by EmbedSwitchOptions
        opponent_stats = normalize_stats(opponent)
        return active_stats + opponent_stats

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """

        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.array(
                        [-1.0 for _ in range(self.n_features * self.seq_len)],
                        dtype=np.float32,
                    ),
                    np.array(
                        [-1.0 for _ in range(self.n_features * self.seq_len)],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                )
            }
        )
