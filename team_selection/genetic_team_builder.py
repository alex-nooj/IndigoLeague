from utils.smogon_data import SmogonData
from poke_env import teambuilder
import typing
import numpy as np


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

    def mutate(self, team: typing.Dict[str, str]):
        self.mons = team
        p = np.asarray([np.exp(self.team_size - x) for x in range(1, self.team_size)])
        n_mutations = np.random.choice(
            list(range(1, self.team_size)), p=p / np.sum(p), size=1
        )[0]

        team_mons = list(
            np.random.choice(
                [k for k in self.mons], size=self.team_size - n_mutations, replace=False
            )
        )
        while len(team_mons) != self.team_size:
            team_mons += np.random.choice(
                [
                    self.data.sample_pokemon(),
                    self.data.sample_teammates(team_mons[-1], size=1, team=team_mons),
                ]
            )

        team_mons.sort()
        new_team = {
            mon: self.mons[mon] if mon in self.mons else self.data.build_pokemon(mon)
            for mon in team_mons
        }

        self.mons = new_team

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
