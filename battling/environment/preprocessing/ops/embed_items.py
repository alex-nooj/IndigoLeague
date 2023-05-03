import pathlib
import typing

import gym
import numpy.typing as npt
from poke_env.environment import AbstractBattle

from battling.environment.preprocessing.op import Op
from utils.gather_opponent_team import gather_opponent_team, gather_team
import numpy as np


class EmbedItems(Op):
    def __init__(self, embedding_size: int):
        """Constructor for Op.

        Args:
            embedding_size: Size of the output of the embedding layer.  Rule of thumb is (size of codex)^(1/4).
        """
        pokemon_path = pathlib.Path(__file__).parent.parent.parent / "teams" / "gen8ou"
        self.items_lut = {}
        self._embedding_size = embedding_size
        counter = 0
        for mon in pokemon_path.iterdir():
            for moveset in mon.iterdir():
                with open(moveset, "r") as fp:
                    pokemon = fp.read()
                item = (
                    pokemon.rsplit("@ ")[-1]
                    .rsplit("\n")[0]
                    .lower()
                    .replace("-", "")
                    .replace(" ", "")
                )
                if item not in self.items_lut:
                    self.items_lut[item] = counter
                    counter += 1
        self.items_lut["unknown_item"] = counter
        counter += 1
        self.items_lut["none"] = counter
        counter += 1

    def embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.Dict[str, npt.NDArray]:
        ally_items = [mon.item for mon in gather_team(battle)]
        if len(ally_items) < 6:
            ally_items += ["none" for _ in range(6 - len(ally_items))]
        opp_items = [mon.item for mon in gather_opponent_team(battle)]
        if len(opp_items) < 6:
            opp_items += ["none" for _ in range(6 - len(opp_items))]
        items = ally_items + opp_items
        items_vec = []
        for item in items:
            if item in self.items_lut:
                items_vec.append(self.items_lut[item])
            else:
                items_vec.append(self.items_lut["none"])
        state["item_ids"] = np.asarray(items_vec)
        return state

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                "item_ids": gym.spaces.Box(
                    np.array([0 for _ in range(12)], dtype=np.int64),
                    np.array([len(self.items_lut) for _ in range(12)], dtype=np.int64),
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
        return {"item_ids": (len(self.items_lut), self._embedding_size, 12)}
