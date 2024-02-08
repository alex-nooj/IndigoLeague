import typing

import gym
import numpy as np
from numpy import typing as npt
from poke_env.environment import AbstractBattle

from indigo_league.training.preprocessing.op import Op

STATS = ["atk", "def", "spa", "spd", "spe"]


class EmbedBoosts(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len,
            n_features=2*len(STATS),
            key=__name__
        )

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        active_boosts = [active.boosts[stat] / 6.0 for stat in STATS]
        opponent_boosts = [opponent.boosts[stat] / 6.0 for stat in STATS]

        return active_boosts + opponent_boosts

    def describe_embedding(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.array([-1.0 for _ in range(self.n_features * self.seq_len)]),
                    np.array([1.0 for _ in range(self.n_features * self.seq_len)]),
                    dtype=np.float32,
                )
            }
        )
