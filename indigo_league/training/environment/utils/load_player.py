import pathlib

import poke_env
import torch
from sb3_contrib import MaskablePPO

from indigo_league.teams.team_builder import AgentTeamBuilder
from indigo_league.training.environment.opponent_player import OpponentPlayer
from indigo_league.utils.fixed_heuristics_player import FixedHeuristicsPlayer


def load_player(
    tag: str, league_path: pathlib.Path, battle_format: str, team_size: int
) -> poke_env.player.Player:
    if tag == "RandomPlayer":
        return poke_env.player.RandomPlayer(
            battle_format=battle_format,
            team=AgentTeamBuilder(
                battle_format=battle_format,
                team_size=team_size,
                randomize_team=True,
            ),
        )
    elif tag == "MaxBasePowerPlay":
        return poke_env.player.MaxBasePowerPlayer(
            battle_format=battle_format,
            team=AgentTeamBuilder(
                battle_format=battle_format,
                team_size=team_size,
                randomize_team=True,
            ),
        )
    elif tag == "FixedHeuristics":
        return FixedHeuristicsPlayer(
            battle_format=battle_format,
            team=AgentTeamBuilder(
                battle_format=battle_format,
                team_size=team_size,
                randomize_team=True,
            ),
        )
    else:
        agent_path = league_path / tag
        model = MaskablePPO.load(agent_path / "network.zip")
        peripherals = torch.load(agent_path / "team.pth")
        return OpponentPlayer(
            model=model,
            team=peripherals["team"],
            preprocessor=peripherals["preprocessor"],
            tag=tag,
            battle_format=battle_format,
            team_size=team_size,
        )
