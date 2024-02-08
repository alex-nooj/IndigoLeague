import pathlib
import typing

import torch
from sb3_contrib import MaskablePPO

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
    preprocessor = team_info["preprocessor"]

    env = Gen8Env(
        preprocessor,
        **rewards,
        poke_path=poke_path,
        battle_format=battle_format,
        start_challenging=True,
        team=team,
        team_size=team.team_size,
        change_opponent=False,
        starting_opponent="SimpleHeuristics",
        seq_len=1,
    )

    model = MaskablePPO.load(
        resume_path,
        env=env,
        verbose=1,
        tensorboard_log=str(poke_path.agent_dir),
    )
    return poke_path, model, env, team.team_size
