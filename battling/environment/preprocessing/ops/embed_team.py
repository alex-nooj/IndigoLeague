import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle, Status

from battling.environment.preprocessing.op import Op
from utils.gather_opponent_team import gather_opponent_team, gather_team
from utils.normalize_stats import normalize_stats


class EmbedTeam(Op):
    def __init__(self):
        self._n_features = 10 * (8 + 7)

    def embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.Dict[str, npt.NDArray]:
        pokemon_list = []
        team = gather_team(battle)[1:]
        for pokemon in team:
            stats = normalize_stats(pokemon)

            status = [0.0 for _ in range(len(Status))]
            if pokemon.status is not None:
                status[int(pokemon.status.value - 1)] = 1.0

            pokemon_list += stats
            pokemon_list += status
        if len(team) < 5:
            for _ in range(5 - len(team)):
                pokemon_list += [-1.0 for _ in range(8 + 7)]
        opp_team = gather_opponent_team(battle)[1:]
        for pokemon in opp_team:
            stats = normalize_stats(pokemon)

            status = [0.0 for _ in range(len(Status))]
            if pokemon.status is not None:
                status[int(pokemon.status.value - 1)] = 1.0

            pokemon_list += stats
            pokemon_list += status
        if len(opp_team) < 5:
            for _ in range(5 - len(opp_team)):
                pokemon_list += [-1.0 for _ in range(8 + 7)]

        state["team_pokemon"] = np.asarray(pokemon_list)
        return state

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                "team_pokemon": gym.spaces.Box(
                    np.asarray([-1 for _ in range(self._n_features)]),
                    np.asarray([1 for _ in range(self._n_features)]),
                    dtype=np.float32,
                )
            }
        )
