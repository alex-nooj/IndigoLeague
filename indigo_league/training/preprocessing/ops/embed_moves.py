import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import EmbeddingLUT
from indigo_league.utils.constants import NUM_MOVES
from indigo_league.utils.smogon_data import SmogonData


class EmbedMoves(Op):
    def __init__(self, embedding_size: int, seq_len: int):
        """Constructor for Op.

        Args:
            embedding_size: Size of the output of the embedding layer.  Rule of thumb is (size of codex)^(1/4).
        """
        super().__init__(seq_len=seq_len, n_features=8, key="move_ids")
        data = SmogonData()
        moves = {}
        for v in data.smogon_data["data"].values():
            for move in v["Moves"]:
                moves[move] = 1
        self.moves_lut = EmbeddingLUT(
            ["null"] + sorted(list(moves.keys())) + ["struggle"]
        )
        self._embedding_size = embedding_size

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        active_moves = [move.id for move in battle.available_moves]
        op_active_moves = [move for move in battle.opponent_active_pokemon.moves]
        ids = self._embed_moves(active_moves) + self._embed_moves(op_active_moves)
        return ids

    def _embed_moves(self, moves: typing.List[str]) -> typing.List[int]:
        if len(moves) > NUM_MOVES:
            all_moves = moves[:NUM_MOVES]
        elif len(moves) < NUM_MOVES:
            all_moves = moves + ["null" for _ in range(NUM_MOVES - len(moves))]
        else:
            all_moves = moves
        return [self.moves_lut[move] for move in all_moves]

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
                            len(self.moves_lut)
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
                len(self.moves_lut),
                self._embedding_size,
                self.seq_len * self.n_features,
            )
        }
