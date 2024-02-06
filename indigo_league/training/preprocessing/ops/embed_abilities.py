import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import EmbeddingLUT
from indigo_league.utils.smogon_data import SmogonData


class EmbedAbilities(Op):
    """Operation for converting abilities to an index."""

    def __init__(self, embedding_size: int, seq_len: int):
        """Constructor for Op.

        Args:
            embedding_size: Size of the output of the embedding layer.  Rule of thumb is (size of codex)^(1/4).
        """
        super().__init__(seq_len=seq_len, n_features=2, key="ability_ids")
        self.abilities_lut = {}
        self._embedding_size = embedding_size

        data = SmogonData()
        abilities = {}
        for mon in data.smogon_data["data"].values():
            for ability in mon["Abilities"]:
                abilities[ability] = True
        self.abilities_lut = EmbeddingLUT(["insomnia"] + sorted(list(abilities.keys())))
        self.abilities_lut = EmbeddingLUT(["none"] + sorted(list(abilities.keys())))

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        """Convert the ability strings to an integer index value.

        Args:
            battle: Current state of the battle (functional)
            state: Current state of the battle (observation)

        Returns:
            List[float]: The updated observational state.
        """
        ids = [
            self.abilities_lut[
                battle.active_pokemon.ability
                if battle.active_pokemon.ability
                else "none"
            ]
        ]
        if battle.opponent_active_pokemon.ability:
            ids.append(self.abilities_lut[battle.opponent_active_pokemon.ability])
        else:
            ids.append(self.abilities_lut["none"])
        return ids

    def _embed_abilities(self, abilities: typing.List[str]) -> typing.List[int]:
        if len(abilities) < 6:
            full_abilities = abilities + ["none" for _ in range(6 - len(abilities))]
        elif len(abilities) > 6:
            full_abilities = abilities[:6]
        else:
            full_abilities = abilities
        return [self.abilities_lut[ability] for ability in full_abilities]

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
                            len(self.abilities_lut)
                            for _ in range(self.seq_len * self.n_features)
                        ],
                        dtype=np.int64,
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
                len(self.abilities_lut),
                self._embedding_size,
                self.seq_len * self.n_features,
            )
        }
