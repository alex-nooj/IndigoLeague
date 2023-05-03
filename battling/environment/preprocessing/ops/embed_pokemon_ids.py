import pathlib
import typing

import gym
import poke_env
from poke_env.environment import AbstractBattle
import numpy.typing as npt
from battling.environment.preprocessing.op import Op
import numpy as np

from utils.gather_opponent_team import gather_team, gather_opponent_team


class EmbedPokemonIDs(Op):
    def __init__(self, embedding_size: int):
        """Constructor for Op.

        Args:
            embedding_size: Size of the output of the embedding layer.  Rule of thumb is (size of codex)^(1/4).
        """
        self.id_lut = {}
        self._embedding_size = embedding_size

        pokemon_path = pathlib.Path(__file__).parent.parent.parent / "teams" / "gen8ou"

        for species, info in poke_env.GEN8_POKEDEX.items():
            if info["num"] > 0:
                self.id_lut[species] = info["num"]

        self.poke_lut = {self.id_lut[pokemon.stem]: ix for ix, pokemon in enumerate(pokemon_path.iterdir())}
        self.poke_lut["fainted"] = len(self.poke_lut)

    def embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.Dict[str, npt.NDArray]:
        team_ids = [self.id_lut[mon.species] for mon in gather_team(battle)]
        if len(team_ids) > 6:
            team_ids = team_ids[:6]
        elif len(team_ids) < 6:
            team_ids += ["fainted" for _ in range(6 - len(team_ids))]

        opp_team_ids = [self.id_lut[mon.species] for mon in gather_opponent_team(battle)]
        if len(opp_team_ids) > 6:
            opp_team_ids = opp_team_ids[:6]
        elif len(opp_team_ids) < 6:
            opp_team_ids += ["fainted" for _ in range(6 - len(opp_team_ids))]

        ids = team_ids + opp_team_ids
        state["pokemon_ids"] = np.asarray([self.poke_lut[id] for id in ids], dtype=np.int64)
        return state

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                "pokemon_ids": gym.spaces.Box(
                    np.array([0 for _ in range(12)], dtype=np.int64),
                    np.array([len(self.poke_lut) for _ in range(12)]),
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
        return {"pokemon_ids": (len(self.poke_lut), self._embedding_size, 12)}
