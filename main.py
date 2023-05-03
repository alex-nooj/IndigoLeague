import pathlib
import typing

import stable_baselines3.common.callbacks as sb3_callbacks
import torch
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO

import utils
from battling.agent.pokemon_feature_extractor import PokemonFeatureExtractor
from battling.callbacks.save_peripherals_callback import SavePeripheralsCallback
from battling.callbacks.success_callback import SuccessCallback
from battling.environment.gen8env import Gen8Env
from battling.environment.teams.team_builder import AgentTeamBuilder


def resume_training(
    resume_path: pathlib.Path,
    battle_format: str,
    rewards: typing.Dict[str, float],
) -> typing.Tuple[str, int, utils.PokePath, PPO]:
    tag, n_steps, _ = resume_path.stem.rsplit("_")
    poke_path = utils.PokePath(tag=tag)
    team_file = resume_path.parent / "team.pth"
    team_info = torch.load(team_file)
    team = team_info["team"]
    preprocessor = team_info["preprocessor"]
    opponent = RandomPlayer(
        battle_format=battle_format, team=AgentTeamBuilder(battle_format=battle_format)
    )
    env = Gen8Env(
        preprocessor,
        **rewards,
        tag=tag,
        league_path=poke_path.league_dir,
        battle_format=battle_format,
        opponent=opponent,
        start_challenging=True,
        team=team,
    )

    model = PPO.load(
        resume_path,
        env=env,
        verbose=1,
        tensorboard_log=str(poke_path.agent_dir),
    )
    return tag, int(n_steps), poke_path, model


def new_training(
        ops: typing.Dict[str, typing.Dict[str, typing.Any]],
        rewards: typing.Dict[str, float],
        battle_format: str,
        shared: typing.List[int],
        pi: typing.List[int],
        vf: typing.List[int],
        tag: typing.Optional[str] = None,
        team_path: typing.Optional[str]=None,
) -> typing.Tuple[str, utils.PokePath, PPO]:
    poke_path = utils.PokePath(tag=tag)
    if tag is None:
        tag = poke_path.tag
    opponent = RandomPlayer(
        battle_format=battle_format, team=AgentTeamBuilder(battle_format=battle_format)
    )
    env = Gen8Env(
        ops,
        **rewards,
        tag=tag,
        league_path=poke_path.league_dir,
        battle_format=battle_format,
        opponent=opponent,
        start_challenging=True,
        team_path=team_path,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=str(poke_path.agent_dir),
        policy_kwargs=dict(
            features_extractor_class=PokemonFeatureExtractor,
            features_extractor_kwargs=dict(
                embedding_infos=env.preprocessor.embedding_infos(),
                n_linear_layers=0,
                shared=shared,
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
    tag: typing.Optional[str] = None,
    team_path: typing.Optional[str] = None,
    resume: typing.Optional[str] = None,
):
    if resume is not None and pathlib.Path(resume).is_file():
        tag, n_steps, poke_path, model = resume_training(pathlib.Path(resume), battle_format, rewards)
        total_timesteps -= n_steps
    else:
        tag, poke_path, model = new_training(ops, rewards, battle_format, shared, pi, vf, tag=tag, team_path=team_path)

    checkpoint_callback = sb3_callbacks.CheckpointCallback(
        save_freq,
        save_path=str(poke_path.agent_dir),
        name_prefix=poke_path.tag,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=sb3_callbacks.CallbackList(
            [
                checkpoint_callback,
                SavePeripheralsCallback(poke_path=poke_path, save_freq=save_freq),
                SuccessCallback(league_dir=poke_path.league_dir, tag=poke_path.tag),
            ]
        ),
        reset_num_timesteps=False,
    )


if __name__ == "__main__":
    cfg_file = pathlib.Path(__file__).parent / "main.yaml"
    cfg = utils.load_config(cfg_file)
    main(**cfg)
