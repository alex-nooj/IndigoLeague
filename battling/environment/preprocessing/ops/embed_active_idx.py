import typing

import gym
import numpy as np
from numpy import typing as npt
from poke_env.environment import AbstractBattle

from battling.environment.preprocessing.op import Op


class EmbedActiveIdx(Op):
    def __init__(self, seq_len: int):
        super().__init__(seq_len=seq_len, n_features=6, key="active_idx")

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        active_pokemon = battle.active_pokemon
        indices = [0.0 for _ in range(6)]
        if active_pokemon is not None:
            all_pokemon = [mon.species for mon in battle.team.values()]
            all_pokemon.sort()
            for ix in range(len(all_pokemon)):
                if all_pokemon == active_pokemon.species:
                    indices[ix] = 1.0
                    break
        return indices

    def describe_embedding(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.asarray([0.0 for _ in range(self.n_features * self.seq_len)]),
                    np.asarray([1.0 for _ in range(self.n_features * self.seq_len)]),
                )
            }
        )
