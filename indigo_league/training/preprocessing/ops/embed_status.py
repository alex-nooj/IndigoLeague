import typing

import gym
import numpy as np
from numpy import typing as npt
from poke_env.environment import AbstractBattle
from poke_env.environment import Status

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import gather_opponent_team
from indigo_league.training.preprocessing.utils import gather_team
from indigo_league.utils.constants import NUM_POKEMON


class EmbedStatus(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len, n_features=2 * (NUM_POKEMON * len(Status) - 1), key=__name__
        )

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        active_status = [float(t == active.status) for t in Status if t != Status.FNT]
        opponent_status = [float(t == opponent.status) for t in Status if t != Status.FNT]

        team_status = []
        for mon in gather_team(battle)[1:]:
            team_status += [float(t == mon.status) for t in Status]
        if len(team_status) != len(Status) * (NUM_POKEMON - 1):
            team_status += [-1 for _ in range(len(Status) * (NUM_POKEMON - 1) - len(team_status))]
        opp_team_status = []
        for mon in gather_opponent_team(battle)[1:]:
            opp_team_status += [float(t == mon.status) for t in Status]
        if len(opp_team_status) != len(Status) * (NUM_POKEMON-1):
            opp_team_status += [-1 for _ in range(len(Status) * (NUM_POKEMON-1) - len(opp_team_status))]
        return active_status + opponent_status + team_status + opp_team_status

    def describe_embedding(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.array([-1.0 for _ in range(self.n_features * self.seq_len)]),
                    np.array([1.0 for _ in range(self.n_features * self.seq_len)]),
                    dtype=np.float32,
                )
            }
        )
