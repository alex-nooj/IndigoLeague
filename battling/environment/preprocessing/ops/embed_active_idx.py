import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle

from battling.environment.preprocessing.op import Op


class EmbedActiveIdx(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len,
            n_features=6,
            key="active_idx",
        )

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        """Embeds active and opponent active pokemon in the current state.

        Args:
            battle: Current state of the battle (functional)
            state: Current state of the battle (observation)

        Returns:
            Dict[str, NDArray]: The updated observational state.
        """
        idx = [0.0 for _ in range(self.n_features)]

        if battle.active_pokemon is not None:
            active_pkm = battle.active_pokemon.species
            team_mons = [pkm.species for pkm in battle.team.values()]
            team_mons.sort()
            for ix, pkm in enumerate(team_mons):
                if pkm == active_pkm:
                    idx[ix] = 1.0
                    break
        return idx

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.asarray([0.0 for _ in range(self.n_features * self.seq_len)]),
                    np.asarray([1.0 for _ in range(self.n_features * self.seq_len)]),
                    dtype=np.float32,
                )
            }
        )
