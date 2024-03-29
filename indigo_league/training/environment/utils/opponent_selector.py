import pathlib
import typing

import numpy as np
import trueskill
from numpy import typing as npt
from poke_env.player import Player

from indigo_league.training.environment.utils.load_player import load_player


def qualities_to_probabilities(
    tag: str, agent_skills: typing.Dict[str, trueskill.Rating]
) -> npt.NDArray:
    qualities = np.asarray(
        [
            trueskill.quality_1vs1(agent_skills[opp], agent_skills[tag])
            for opp in agent_skills.keys()
            if opp != tag
        ]
    )

    sine_sq = np.sin(qualities) ** 2

    return sine_sq / np.sum(sine_sq)


class OpponentSelector:
    def __init__(
        self,
        tag: str,
        battle_format: str,
        league_path: pathlib.Path,
        team_size: int,
    ):
        self._tag = tag
        self._battle_format = battle_format
        self.league_path = league_path
        self.team_size = team_size

    def choose(
        self,
        agent_skills: typing.Dict[str, trueskill.Rating],
    ) -> typing.Tuple[str, Player]:
        tags = [t for t in agent_skills.keys() if t != self._tag]
        if np.random.uniform() > 0.9:
            probs = [1.0 / len(tags) for _ in tags]
        else:
            probs = qualities_to_probabilities(tag=self._tag, agent_skills=agent_skills)

        opponent_tag = str(np.random.choice(tags, p=probs))
        opp = load_player(
            tag=opponent_tag,
            league_path=self.league_path,
            battle_format=self._battle_format,
            team_size=self.team_size,
        )
        return opponent_tag, opp
