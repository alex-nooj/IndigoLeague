import csv
import pathlib
import typing

import numpy as np
import poke_env
import torch
import trueskill
from poke_env.player import Player
from stable_baselines3 import PPO

from battling.environment.opponent_player import OpponentPlayer
from battling.environment.teams.team_builder import AgentTeamBuilder


class Matchmaker:
    def __init__(self, tag: str, league_path: pathlib.Path, battle_format: str):
        self.agent_skills = {tag: trueskill.Rating()}
        self._tag = tag
        self._battle_format = battle_format
        self._league_path = league_path
        self._load_league_skills()
        self._pre_training = True
        self._curriculum = ["RandomPlayer", "MaxBasePowerPlay", "SimpleHeuristics"]

    def update_and_choose(
        self, opponent: str, battle_won: bool
    ) -> typing.Tuple[str, poke_env.player.Player]:
        if battle_won:
            self._update(self._tag, opponent)
        else:
            self._update(opponent, self._tag)
        return self.choose()

    def save(self):
        with open(self._league_path / "trueskills.csv", "w") as fp:
            writer = csv.DictWriter(fp, fieldnames=["tag", "mu", "sigma"])
            writer.writeheader()
            for agent in self._league_path.iterdir():
                if agent.stem in self.agent_skills:
                    writer.writerow(
                        {
                            "tag": agent.stem,
                            "mu": self.agent_skills[agent.stem].mu,
                            "sigma": self.agent_skills[agent.stem].sigma,
                        }
                    )
            for agent in ["RandomPlayer", "MaxBasePowerPlay", "SimpleHeuristics"]:
                writer.writerow(
                    {
                        "tag": agent,
                        "mu": self.agent_skills[agent].mu,
                        "sigma": self.agent_skills[agent].sigma,
                    }
                )

    def _update(self, winner: str, loser: str):
        if winner not in self.agent_skills:
            self.agent_skills[winner] = trueskill.Rating()
        if loser not in self.agent_skills:
            self.agent_skills[loser] = trueskill.Rating()

        self.agent_skills[winner], self.agent_skills[loser] = trueskill.rate_1vs1(
            self.agent_skills[winner], self.agent_skills[loser]
        )

    def choose(self) -> typing.Tuple[str, poke_env.player.Player]:
        # Grab the probability of a tie between our agent and the league agents
        match_qualities = []
        tags = []
        for tag, rating in self.agent_skills.items():
            if tag == self._tag:
                continue
            match_qualities.append(
                3 if trueskill.quality_1vs1(
                    self.agent_skills[tag], self.agent_skills[self._tag]
                ) > 0.5
                else 1
            )
            tags.append(tag)

        # Normalize the probabilities of ties
        match_qualities = np.asarray(match_qualities) / np.sum(match_qualities)

        # Use the normalized match qualities to select an opponent. Bias will be toward agents we're likely to tie
        opponent_tag = np.random.choice(tags, p=match_qualities)
        player = self._load_player(opponent_tag)
        return opponent_tag, player

    def _load_league_skills(self):
        if (self._league_path / "trueskills.csv").is_file():
            with open(self._league_path / "trueskills.csv", mode="r") as fp:
                csv_file = csv.DictReader(fp)
                for line in csv_file:
                    self.agent_skills[line["tag"]] = trueskill.Rating(
                        mu=float(line["mu"]), sigma=float(line["sigma"])
                    )

    def _load_player(self, opponent_tag: str) -> Player:
        if opponent_tag == "RandomPlayer":
            return poke_env.player.RandomPlayer(
                battle_format=self._battle_format,
                team=AgentTeamBuilder(
                    battle_format=self._battle_format,
                ),
            )
        elif opponent_tag == "MaxBasePowerPlay":
            return poke_env.player.MaxBasePowerPlayer(
                battle_format=self._battle_format,
                team=AgentTeamBuilder(battle_format=self._battle_format),
            )
        elif opponent_tag == "SimpleHeuristics":
            return poke_env.player.SimpleHeuristicsPlayer(
                battle_format=self._battle_format,
                team=AgentTeamBuilder(battle_format=self._battle_format),
            )
        else:
            agent_path = self._league_path / opponent_tag
            model = PPO.load(agent_path / "network.zip")
            peripherals = torch.load(agent_path / "team.pth")
            return OpponentPlayer(
                model=model,
                team=peripherals["team"],
                preprocessor=peripherals["preprocessor"],
                tag=opponent_tag,
                battle_format=self._battle_format,
            )

