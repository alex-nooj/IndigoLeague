import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import EmbeddingLUT
from indigo_league.training.preprocessing.utils import gather_opponent_team
from indigo_league.training.preprocessing.utils import gather_team
from indigo_league.utils.constants import NUM_POKEMON
from indigo_league.teams.smogon_data import SmogonData


class EmbedItems(Op):
    def __init__(self, embedding_size: int, seq_len: int):
        """Constructor for Op.

        Args:
            embedding_size: Size of the output of the embedding layer.  Rule of thumb is (size of codex)^(1/4).
        """
        super().__init__(seq_len=seq_len, n_features=12, key="item_ids")
        data = SmogonData()
        items = {}
        for mon in data.smogon_data["data"].values():
            for item in mon["Items"]:
                items[item] = True

        self.items_lut = EmbeddingLUT(
            ["none"] + sorted(list(items.keys())) + ["unknown_item"]
        )

        self._embedding_size = embedding_size

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        ally_items = [
            mon.item if mon.item else "unknown_item" for mon in gather_team(battle)
        ]
        opp_items = [
            mon.item if mon.item else "unknown_item"
            for mon in gather_opponent_team(battle)
        ]
        items_vec = self._embed_items(ally_items) + self._embed_items(opp_items)
        return items_vec

    def _embed_items(self, items: typing.List[str]) -> typing.List[int]:
        if len(items) < NUM_POKEMON:
            all_items = items + ["none" for _ in range(NUM_POKEMON - len(items))]
        elif len(items) > NUM_POKEMON:
            all_items = items[:NUM_POKEMON]
        else:
            all_items = items

        return [self.items_lut[item] for item in all_items]

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
                            len(self.items_lut)
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
                len(self.items_lut),
                self._embedding_size,
                self.seq_len * self.n_features,
            )
        }
