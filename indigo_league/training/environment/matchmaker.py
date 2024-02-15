import pathlib
import typing

import numpy as np
import trueskill
from omegaconf import OmegaConf
from poke_env.player import Player

from indigo_league.utils.load_player import load_player


class Matchmaker:
    def __init__(
        self, tag: str, league_path: pathlib.Path, battle_format: str, team_size: int
    ):
        self.agent_skills = {tag: trueskill.Rating()}
        self._tag = tag
        self._battle_format = battle_format
        self._league_path = league_path
        self.team_size = team_size

        self._load_league_skills()
        self._players = {
            t: load_player(
                tag=t, league_path=league_path, battle_format=battle_format, team_size=team_size
            )
            for t in self.agent_skills if t not in [tag, "FixedHeuristics"]
        }

    def choose(self) -> typing.Tuple[str, Player]:
        opponent_tag = self._choose_trueskill()
        player = load_player(
            tag=opponent_tag,
            league_path=self._league_path,
            battle_format=self._battle_format,
            team_size=self.team_size,
        )
        return opponent_tag, player

    def update(self, opponent: str, battle_won: bool):
        if battle_won:
            self._update(self._tag, opponent)
        else:
            self._update(opponent, self._tag)

    def update_and_choose(
        self, opponent: str, battle_won: bool
    ) -> typing.Tuple[str, Player]:
        self.update(opponent, battle_won)

        return self.choose()

    def save(self):
        dict_skills = {}
        for agent in self._league_path.iterdir():
            if agent.stem in self.agent_skills:
                dict_skills[agent.stem] = {
                    "mu": self.agent_skills[agent.stem].mu,
                    "sigma": self.agent_skills[agent.stem].sigma,
                }

        for agent in ["FixedHeuristics"]:
            dict_skills[agent] = {
                "mu": self.agent_skills[agent].mu,
                "sigma": self.agent_skills[agent].sigma,
            }
        OmegaConf.save(config=dict_skills, f=(self._league_path / "trueskills.yaml"))

    def set_team_size(self, team_size: int):
        self.team_size = team_size

    def _update(self, winner: str, loser: str):
        if winner not in self.agent_skills:
            self.agent_skills[winner] = trueskill.Rating()
        if loser not in self.agent_skills:
            self.agent_skills[loser] = trueskill.Rating()

        self.agent_skills[winner], self.agent_skills[loser] = trueskill.rate_1vs1(
            self.agent_skills[winner], self.agent_skills[loser]
        )

    def _choose_trueskill(self) -> str:
        # Grab the probability of a tie between our agent and the league agents
        match_qualities = []
        tags = []
        for tag, rating in self.agent_skills.items():
            if tag == self._tag:
                continue
            match_qualities.append(
                3
                if trueskill.quality_1vs1(
                    self.agent_skills[tag], self.agent_skills[self._tag]
                )
                > 0.5
                else 1
            )
            tags.append(tag)

        # Occasionally choose an opponent at random
        if np.random.uniform() > 0.9:
            return str(np.random.choice(tags))
        else:
            # Normalize the probabilities of ties
            match_qualities = np.asarray(match_qualities) / np.sum(match_qualities)

            # Use the normalized match qualities to select an opponent. Bias will be toward agents we're likely to tie
            return str(np.random.choice(tags, p=match_qualities))

    def _load_league_skills(self):
        if (self._league_path / "trueskills.yaml").is_file():
            self.agent_skills = {}
            for tag, skill in OmegaConf.load(
                self._league_path / "trueskills.yaml"
            ).items():
                self.agent_skills[tag] = trueskill.Rating(
                    mu=skill["mu"], sigma=skill["sigma"]
                )
