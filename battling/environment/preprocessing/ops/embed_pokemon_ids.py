import typing

import gym
import numpy as np
import numpy.typing as npt
import poke_env
from poke_env.environment import AbstractBattle

from battling.environment.preprocessing.op import Op
from utils.embedding_lut import EmbeddingLUT
from utils.gather_opponent_team import gather_opponent_team
from utils.gather_opponent_team import gather_team


class EmbedPokemonIDs(Op):
    def __init__(self, embedding_size: int, seq_len: int):
        """Constructor for Op.

        Args:
            embedding_size: Size of the output of the embedding layer.  Rule of thumb is (size of codex)^(1/4).
        """
        super().__init__(seq_len=seq_len, n_features=12, key="pokemon_ids")
        self.id_lut = {}
        self._embedding_size = embedding_size

        all_mons = {species: info["num"] for species, info in poke_env.GEN8_POKEDEX.items() if info["num"] > 0}
        self.id_lut = EmbeddingLUT(["fainted"] + list(all_mons.keys()), values=[0] + list(all_mons.values()))
        self.poke_lut = EmbeddingLUT(self.id_lut.values())

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        mons = [mon.species for mon in gather_team(battle)]
        opp_mons = [mon.species for mon in gather_opponent_team(battle)]
        mon_vec = self._embed_pokemon_ids(mons) + self._embed_pokemon_ids(opp_mons)
        return mon_vec

    def _embed_pokemon_ids(self, mons: typing.List[str]) -> typing.List[int]:
        if len(mons) > 6:
            all_mons = mons[:6]
        elif len(mons) < 6:
            all_mons = mons + ["fainted" for _ in range(6 - len(mons))]
        else:
            all_mons = mons

        return [self.poke_lut[self.id_lut[mon]] for mon in all_mons]

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.array(
                        [0 for _ in range(self.seq_len * self.n_features)],
                        dtype=np.int64,
                    ),
                    np.array(
                        [
                            len(self.poke_lut)
                            for _ in range(self.seq_len * self.n_features)
                        ]
                    ),
                    dtype=np.int64,
                )
            }
        )

    def embedding_infos(self) -> typing.Dict[str, typing.Tuple[int, int, int]]:
        """Describes what the embedding layer should look like after this op.

        Returns:
            Dict[str, Tuple[int, int, int]]: The number of items in the codex, the embedding size, and the number of
                features
        """
        return {
            self.key: (
                len(self.poke_lut),
                self._embedding_size,
                self.seq_len * self.n_features,
            )
        }
