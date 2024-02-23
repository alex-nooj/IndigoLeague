import pathlib
import typing

import torch
from sb3_contrib import MaskablePPO

from indigo_league.teams import load_team_from_file
from indigo_league.training.environment import build_env
from indigo_league.training.environment import Gen8Env
from indigo_league.utils.directory_helper import PokePath


def resume_training(
    resume_path: pathlib.Path,
    battle_format: str,
    rewards: typing.Dict[str, float],
) -> typing.Tuple[PokePath, MaskablePPO, Gen8Env, int]:

    tag = resume_path.parent.stem
    poke_path = PokePath(tag=tag)
    print(resume_path)
    team_file = resume_path.parent / "team.pth"
    print(team_file)
    team_info = torch.load(team_file)
    team = team_info["team"]
    team.set_team(load_team_from_file(str(resume_path.parent / "team.txt")))
    preprocessor = team_info["preprocessor"]

    env = build_env(
        ops=preprocessor,
        seq_len=1,
        poke_path=poke_path,
        **rewards,
        battle_format=battle_format,
        team_size=team.team_size,
        team=team,
        change_opponent=False,
        starting_opponent="FixedHeuristics",
    )

    model = MaskablePPO.load(
        resume_path,
        env=env,
        verbose=1,
        tensorboard_log=str(poke_path.agent_dir),
    )
    return poke_path, model, env, team.team_size
