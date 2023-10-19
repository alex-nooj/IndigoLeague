import pathlib
import typing

import numpy.typing as npt
import stable_baselines3.common.callbacks as sb3_callbacks
import torch
from poke_env.player import RandomPlayer
from sb3_contrib import MaskablePPO
from stable_baselines3.common.base_class import BaseAlgorithm

import utils
from battling.agent.transformer_feature_extractor import TransformerFeatureExtractor
from battling.callbacks.curriculum_callback import CurriculumCallback
from battling.callbacks.save_peripherals_callback import SavePeripheralsCallback
from battling.callbacks.success_callback import SuccessCallback
from battling.environment.gen8env import Gen8Env
from battling.environment.teams.team_builder import AgentTeamBuilder


def resume_training(
    resume_path: pathlib.Path,
    battle_format: str,
    rewards: typing.Dict[str, float],
) -> typing.Tuple[str, int, utils.PokePath, MaskablePPO, Gen8Env, int]:
    seq_len = 1
    tag, n_steps, _ = resume_path.stem.rsplit("_")
    poke_path = utils.PokePath(tag=tag)
    print(resume_path)
    team_file = resume_path.parent / "team.pth"
    print(team_file)
    team_info = torch.load(team_file)
    team = team_info["team"]
    preprocessor = team_info["preprocessor"]
    opponent = RandomPlayer(
        battle_format=battle_format, team=AgentTeamBuilder(battle_format=battle_format, team_size=team.team_size)
    )
    env = Gen8Env(
        preprocessor,
        **rewards,
        seq_len=seq_len,
        tag=tag,
        league_path=poke_path.league_dir,
        battle_format=battle_format,
        opponent=opponent,
        start_challenging=True,
        team=team,
        team_size=team.team_size
    )

    model = MaskablePPO.load(
        resume_path,
        env=env,
        verbose=1,
        tensorboard_log=str(poke_path.agent_dir),
    )
    return tag, int(n_steps), poke_path, model, env, team.team_size


def create_env(
    ops: typing.Dict[str, typing.Dict[str, typing.Any]],
    rewards: typing.Dict[str, float],
    seq_len: int,
    battle_format: str,
    team_size: int,
    tag: typing.Optional[str] = None,
) -> Gen8Env:
    poke_path = utils.PokePath(tag=tag)
    if tag is None:
        tag = poke_path.tag
    opponent = RandomPlayer(
        battle_format=battle_format,
        team=AgentTeamBuilder(battle_format=battle_format, team_size=team_size),
    )

    return Gen8Env(
        ops,
        **rewards,
        seq_len=seq_len,
        tag=tag,
        league_path=poke_path.league_dir,
        battle_format=battle_format,
        opponent=opponent,
        start_challenging=True,
        team_size=team_size,
    )


def new_training(
    env: Gen8Env,
    seq_len: int,
    shared: typing.List[int],
    pi: typing.List[int],
    vf: typing.List[int],
    tag: typing.Optional[str] = None,
) -> typing.Tuple[str, utils.PokePath, BaseAlgorithm]:
    poke_path = utils.PokePath(tag=tag)
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=str(poke_path.agent_dir),
        policy_kwargs=dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(
                embedding_infos=env.preprocessor.embedding_infos(),
                seq_len=seq_len,
                n_linear_layers=1,
                n_encoders=3,
                shared=shared,
                n_heads=8,
                d_feedforward=1024,
                dropout=0.0,
            ),
            net_arch=dict(pi=pi, vf=vf),
            # activation_fn=torch.nn.LeakyReLU,
        ),
    )

    return tag, poke_path, model


def main(
    ops: typing.Dict[str, typing.Dict[str, typing.Any]],
    rewards: typing.Dict[str, float],
    battle_format: str,
    total_timesteps: int,
    save_freq: int,
    shared: typing.List[int],
    pi: typing.List[int],
    vf: typing.List[int],
    starting_team_size=1,
    final_team_size=6,
    tag: typing.Optional[str] = None,
    resume: typing.Optional[str] = None,
):
    if resume is not None and pathlib.Path(resume).is_file():
        tag, n_steps, poke_path, model, env, starting_team_size = resume_training(
            pathlib.Path(resume), battle_format, rewards
        )
        total_timesteps -= n_steps
    else:
        seq_len = 1
        env = create_env(
            ops=ops,
            rewards=rewards,
            seq_len=seq_len,
            battle_format=battle_format,
            team_size=starting_team_size,
            tag=tag,
        )
        tag, poke_path, model = new_training(env, seq_len, shared, pi, vf, tag)

    checkpoint_callback = sb3_callbacks.CheckpointCallback(
        save_freq,
        save_path=str(poke_path.agent_dir),
        name_prefix=poke_path.tag,
    )

    for team_size in range(starting_team_size, final_team_size):
        print(f"Team Size: {team_size}")
        env.set_team_size(team_size)
        model.set_env(env)

        model.learn(
            total_timesteps=total_timesteps,
            callback=sb3_callbacks.CallbackList(
                [
                    checkpoint_callback,
                    SavePeripheralsCallback(poke_path=poke_path, save_freq=save_freq),
                    CurriculumCallback(0.7),
                ]
            ),
            reset_num_timesteps=False,
        )

    print(f"Team Size: {final_team_size}")
    env.set_team_size(final_team_size)
    model.set_env(env)

    model.learn(
        total_timesteps=total_timesteps,
        callback=sb3_callbacks.CallbackList(
            [
                checkpoint_callback,
                SavePeripheralsCallback(poke_path=poke_path, save_freq=save_freq),
                SuccessCallback(poke_path.agent_dir, poke_path.league_dir, tag=tag),
            ]
        ),
        reset_num_timesteps=False,
    )


if __name__ == "__main__":
    cfg_file = pathlib.Path(__file__).parent / "main.yaml"
    cfg = utils.load_config(cfg_file)
    main(**cfg)
