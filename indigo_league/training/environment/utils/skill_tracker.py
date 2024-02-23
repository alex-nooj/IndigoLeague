import pathlib
import typing

import trueskill
from memory_profiler import profile
from omegaconf import OmegaConf


class SkillTracker:
    def __init__(self, tag: str, league_path: pathlib.Path):
        self._agent_skills = {tag: trueskill.Rating()}
        self._tag = tag

        if (league_path / "trueskills.yaml").is_file():
            self._agent_skills = {}
            for tag, skill in OmegaConf.load(league_path / "trueskills.yaml").items():
                self._agent_skills[tag] = trueskill.Rating(
                    mu=skill["mu"], sigma=skill["sigma"]
                )

    def update(self, opponent: str, battle_won: bool):
        if battle_won:
            winner, loser = self._tag, opponent
        else:
            winner, loser = opponent, self._tag

        if winner not in self._agent_skills:
            self._agent_skills[winner] = trueskill.Rating()
        if loser not in self._agent_skills:
            self._agent_skills[loser] = trueskill.Rating()

        self._agent_skills[winner], self._agent_skills[loser] = trueskill.rate_1vs1(
            self._agent_skills[winner], self._agent_skills[loser]
        )

    @property
    def agent_skills(self) -> typing.Dict[str, trueskill.Rating]:
        return self._agent_skills
