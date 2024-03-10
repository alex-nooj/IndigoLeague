import pathlib
import typing

import trueskill
from poke_env.player import Player

from indigo_league.training.environment.utils.opponent_selector import OpponentSelector
from indigo_league.training.environment.utils.save_agent_skills import save_agent_skills
from indigo_league.training.environment.utils.skill_tracker import SkillTracker


class Matchmaker:
    def __init__(
        self, tag: str, league_path: pathlib.Path, battle_format: str, team_size: int
    ):
        self._skill_tracker = SkillTracker(tag=tag, league_path=league_path)
        self._selector = OpponentSelector(
            tag=tag,
            battle_format=battle_format,
            league_path=league_path,
            team_size=team_size,
        )

        self._tag = tag
        self._battle_format = battle_format
        self._league_path = league_path
        self.team_size = team_size

    def update_and_choose(
        self, opponent: str, battle_won: bool
    ) -> typing.Tuple[str, Player]:
        self._skill_tracker.update(opponent, battle_won)
        return self._selector.choose(self._skill_tracker.agent_skills)

    def save(self):
        save_agent_skills(self._league_path, self.agent_skills)

    def set_team_size(self, team_size: int):
        self.team_size = team_size
        self._selector = OpponentSelector(
            tag=self._tag,
            battle_format=self._battle_format,
            league_path=self._league_path,
            team_size=team_size,
        )

    @property
    def agent_skills(self) -> typing.Dict[str, trueskill.Rating]:
        return self._skill_tracker.agent_skills
