import typing

import numpy as np
from poke_env import teambuilder

from indigo_league.utils.smogon_data import SmogonData


class GeneticTeamBuilder(teambuilder.Teambuilder):
    def __init__(
        self,
        team_size: int = 6,
        mode: typing.Literal["random", "sample", "teammate"] = "teammate",
    ):
        self.data = SmogonData()
        self.mons = {}
        self.team_size = team_size
        self.generate_team(mode)

    def yield_team(self) -> str:
        if len(self.mons) == 0:
            self.generate_team()

        return self.join_team(self.parse_showdown_team(self._team_from_mons()))

    def generate_team(
        self, mode: typing.Literal["random", "sample", "teammate"] = "teammate"
    ):
        if mode == "random":
            mons = self.data.random_pokemon(size=self.team_size)
        elif mode == "sample":
            mons = self.data.sample_pokemon(size=self.team_size)
        elif mode == "teammate":
            mons = self.data.sample_pokemon()
            for _ in range(self.team_size - 1):
                mons += self.data.sample_teammates(mons[-1], size=1, team=mons)
        else:
            raise RuntimeError(
                f"Got unexpected mode: {mode} (Expected 'random', 'sample', or 'teammate'"
            )

        mons.sort()
        self.mons = {mon: self.data.build_pokemon(mon) for mon in mons}

    def mutate(self, team: typing.Dict[str, str], n_changes: int):
        survivors = np.random.choice(
            list(team.keys()), size=len(team.keys()) - n_changes
        )
        self.mons = {k: v for k, v in team.items() if k in survivors}
        while len(list(self.mons.keys())) < self.team_size:
            teammate = np.random.choice(list(team.keys()))
            new_mon = self.data.sample_teammates(
                teammate, size=1, team=list(self.mons.keys())
            )[0]
            self.mons[new_mon] = self.data.build_pokemon(new_mon)

    @property
    def team(self) -> str:
        return self._team_from_mons()

    @property
    def mons_dict(self) -> typing.Dict[str, str]:
        return self.mons

    def _team_from_mons(self) -> str:
        team = ""
        for mon in self.mons.values():
            team += mon
            team += "\n"
        return team
