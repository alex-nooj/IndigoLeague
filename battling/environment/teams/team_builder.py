import pathlib
import typing
import numpy as np
from poke_env.teambuilder.teambuilder import Teambuilder


def load_team_from_file(file_path: pathlib.Path) -> str:
    with open(str(file_path), "r") as fp:
        team = fp.read()
    return team


def save_team_to_file(file_path: pathlib.Path, team: str):
    with open(str(file_path / "team.txt"), "w") as fp:
        fp.write(team)


def generate_random_team(battle_format: str) -> str:
    team_dir = pathlib.Path(__file__).parent / battle_format
    mons = np.random.choice(list(team_dir.iterdir()), replace=False, size=6)
    movesets = [np.random.choice(list(mon.iterdir())) for mon in mons]
    team = ""
    for moveset in movesets:
        with open(moveset, "r") as fp:
            team += fp.read()
        team += "\n\n"
    return team


class AgentTeamBuilder(Teambuilder):
    def __init__(self, battle_format: str, team_path: typing.Optional[pathlib.Path] = None):
        if team_path is not None:
            if team_path.is_file():
                with open(str(team_path), "r") as fp:
                    team = fp.read()
                self._team = self.join_team(self.parse_showdown_team(team))
            else:
                raise RuntimeError(f"Team file '{team_path}' does not exist!")
        else:
            self._team = self.join_team(self.parse_showdown_team(generate_random_team(battle_format)))

    def yield_team(self) -> str:
        return self._team
