import pathlib
import typing
import numpy.typing as npt
from battling.environment.preprocessing.op import Op
from poke_env.environment import AbstractBattle
import numpy as np
import gym


class EmbedMoves(Op):
    def __init__(self, embedding_size: int):
        """Constructor for Op.

        Args:
            embedding_size: Size of the output of the embedding layer.  Rule of thumb is (size of codex)^(1/4).
        """
        pokemon_path = pathlib.Path(__file__).parent.parent.parent / "teams" / "gen8ou"
        moves = {}
        for mon in pokemon_path.iterdir():
            for moveset in mon.iterdir():
                with open(moveset, "r") as fp:
                    pokemon = fp.read()
                for move in (
                    pokemon.rsplit("Nature\n")[-1].replace("- ", "").rsplit("\n")[:-1]
                ):
                    moves[move] = 1

        self.moves_lut = {
            move.lower().replace(" ", "").replace("-", ""): ix
            for ix, move in enumerate(moves)
        }
        self.moves_lut["struggle"] = len(self.moves_lut)
        self.moves_lut["null"] = len(self.moves_lut)
        self._embedding_size = embedding_size

    def embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.Dict[str, npt.NDArray]:
        active_moves = [self.moves_lut[move.id] for move in battle.available_moves]
        if len(active_moves) > 4:
            active_moves = active_moves[:4]
        elif len(active_moves) < 4:
            active_moves += [
                self.moves_lut["null"] for _ in range(4 - len(active_moves))
            ]

        op_active_moves = [
            self.moves_lut[move] for move in battle.opponent_active_pokemon.moves
        ]
        if len(op_active_moves) > 4:
            op_active_moves = op_active_moves[:4]
        elif len(op_active_moves) < 4:
            op_active_moves += [
                self.moves_lut["null"] for _ in range(4 - len(op_active_moves))
            ]
        state["move_ids"] = np.asarray(active_moves + op_active_moves, dtype=np.int64)
        return state

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                "move_ids": gym.spaces.Box(
                    np.array([0 for _ in range(8)], dtype=np.int64),
                    np.array([len(self.moves_lut) for _ in range(8)]),
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
        return {"move_ids": (len(self.moves_lut), self._embedding_size, 8)}
