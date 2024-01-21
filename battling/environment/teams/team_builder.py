import pathlib
import typing

import numpy as np
from poke_env.teambuilder.teambuilder import Teambuilder

from utils.smogon_data import SmogonData


def load_team_from_file(file_path: pathlib.Path) -> str:
    with open(str(file_path), "r") as fp:
        team = fp.read()
    return team


def save_team_to_file(file_path: pathlib.Path, team: str):
    with open(str(file_path / "team.txt"), "w") as fp:
        fp.write(team)


def generate_random_team(team_size: int) -> typing.List[str]:
    data = SmogonData()
    mons = data.sample_pokemon()
    for _ in range(team_size - 1):
        mons += data.sample_teammates(mons[-1], size=1, team=mons)
    mons.sort()

    team = []
    for mon in mons:
        team.append(data.build_pokemon(mon))
    return team


class AgentTeamBuilder(Teambuilder):
    def __init__(
        self, battle_format: str, team_size: int, randomize_team: bool = False
    ):
        self._team_size = team_size
        if not randomize_team:
            self._team = generate_random_team(6)
            print(f"\r{self._team[:team_size]}", end="")
        else:
            self._team = None

    def set_team_size(self, team_size: int):
        self._team_size = team_size

    def yield_team(self) -> str:
        if self._team:
            ixs = np.random.choice(
                list(range(len(self._team))), self._team_size, replace=False
            )
            team = [self._team[i] for i in ixs]
            return self.join_team(self.parse_showdown_team("\n".join(team)))
        else:
            team = generate_random_team(self._team_size)
            print(f"\r{team}", end="")
            return self.join_team(self.parse_showdown_team("\n".join(team)))

    @property
    def team_size(self) -> int:
        return self._team_size

    def save_team(self, save_dir: pathlib.Path):
        if not self._team:
            raise RuntimeError("Trying to save team before it is set!")
        with open(str(save_dir / "team.txt"), "w") as fp:
            fp.write("".join(self._team))
