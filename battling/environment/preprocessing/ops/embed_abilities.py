import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle

from battling.environment.preprocessing.op import Op
from utils.embedding_lut import EmbeddingLUT
from utils.gather_opponent_team import gather_opponent_team
from utils.gather_opponent_team import gather_team
from utils.smogon_data import SmogonData


class EmbedAbilities(Op):
    """Operation for converting abilities to an index."""

    def __init__(self, embedding_size: int, seq_len: int):
        """Constructor for Op.

        Args:
            embedding_size: Size of the output of the embedding layer.  Rule of thumb is (size of codex)^(1/4).
        """
        super().__init__(seq_len=seq_len, n_features=12, key="ability_ids")
        self.abilities_lut = {}
        self._embedding_size = embedding_size

        data = SmogonData()
        abilities = {}
        for mon in data.smogon_data["data"].values():
            for ability in mon["Abilities"]:
                abilities[ability] = True
        self.abilities_lut = EmbeddingLUT(["none"] + sorted(list(abilities.keys())))

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> npt.NDArray:
        """Convert the ability strings to an integer index value.

        Args:
            battle: Current state of the battle (functional)
            state: Current state of the battle (observation)

        Returns:
            Dict[str, NDArray]: The updated observational state.
        """
        # Gather all the abilities in the specific order of available switches.
        abilities = [mon.ability if mon.ability else "none" for mon in gather_team(battle)]

        # Gather the abilities of the opponent team.
        opp_abilities = [mon.ability if mon.ability else "none" for mon in gather_opponent_team(battle)]

        ids = self._embed_abilities(abilities) + self._embed_abilities(opp_abilities)

        return np.asarray(ids)

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
                    np.array([0 for _ in range(self.seq_len * self.n_features)], dtype=np.int64),
                    np.array(
                        [len(self.abilities_lut) for _ in range(self.seq_len * self.n_features)], dtype=np.int64
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
        return {self.key: (len(self.abilities_lut), self._embedding_size, self.seq_len * self.n_features)}
