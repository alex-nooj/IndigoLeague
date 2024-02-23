import asyncio
import pathlib
import typing

import stable_baselines3.common.callbacks as sb3_callbacks
import torch
from memory_profiler import profile
from sb3_contrib import MaskablePPO

from indigo_league import training
from indigo_league.teams.load_team import load_team_from_file
from indigo_league.teams.run_genetic_algo import genetic_team_search
from indigo_league.teams.team_builder import AgentTeamBuilder
from indigo_league.training import callbacks
from indigo_league.training.environment import build_env
from indigo_league.training.network import PokemonFeatureExtractor
from indigo_league.utils import load_config
from indigo_league.utils.directory_helper import PokePath


def setup(
    ops: typing.Dict[str, typing.Dict[str, typing.Any]],
    rewards: typing.Dict[str, float],
    battle_format: str,
    seq_len: int,
    ensemble_size: int,
    shared: typing.List[int],
    pi: typing.List[int],
    vf: typing.List[int],
    starting_team_size: int,
    poke_path: PokePath,
    teambuilder: typing.Optional[AgentTeamBuilder],
):
    if teambuilder is None:
        teambuilder = asyncio.get_event_loop().run_until_complete(
            genetic_team_search(20, 0, battle_format, 1)
        )
    teambuilder.save_team(poke_path.agent_dir)
    env = build_env(
        ops=ops,
        seq_len=seq_len,
        poke_path=poke_path,
        **rewards,
        battle_format=battle_format,
        team_size=starting_team_size,
        change_opponent=False,
        starting_opponent="FixedHeuristics",
        team=teambuilder,
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=str(poke_path.agent_dir),
        policy_kwargs=dict(
            features_extractor_class=PokemonFeatureExtractor,
            features_extractor_kwargs=dict(
                embedding_infos=env.preprocessor.embedding_infos(),
                seq_len=seq_len,
                n_linear_layers=1,
                n_encoders=3,
                shared=shared,
                n_heads=8,
                d_feedforward=1024,
                dropout=0.0,
                ensemble_size=ensemble_size,
            ),
            net_arch=dict(pi=pi, vf=vf),
            activation_fn=torch.nn.LeakyReLU,
        ),
    )

    return env, model


def main(
    ops: typing.Dict[str, typing.Dict[str, typing.Any]],
    rewards: typing.Dict[str, float],
    battle_format: str,
    total_timesteps: int,
    save_freq: int,
    seq_len: int,
    ensemble_size: int,
    shared: typing.List[int],
    pi: typing.List[int],
    vf: typing.List[int],
    starting_team_size=1,
    final_team_size=6,
    tag: typing.Optional[str] = None,
    resume: typing.Optional[str] = None,
    teambuilder: AgentTeamBuilder = None,
):
    if resume is not None and pathlib.Path(resume).is_file():
        poke_path, model, env, starting_team_size = training.resume_training(
            pathlib.Path(resume), battle_format, rewards
        )
    else:
        poke_path = PokePath(tag=tag)

        env, model = setup(
            ops=ops,
            rewards=rewards,
            battle_format=battle_format,
            seq_len=seq_len,
            ensemble_size=ensemble_size,
            shared=shared,
            pi=pi,
            vf=vf,
            starting_team_size=starting_team_size,
            poke_path=poke_path,
            teambuilder=teambuilder,
        )

    checkpoint_callback = sb3_callbacks.CheckpointCallback(
        save_freq,
        save_path=str(poke_path.agent_dir),
        name_prefix=poke_path.tag,
    )
    callback_list = [
        checkpoint_callback,
        callbacks.SavePeripheralsCallback(poke_path=poke_path, save_freq=save_freq),
    ]

    if starting_team_size != final_team_size:
        training.curriculum(
            env=env,
            model=model,
            starting_team_size=starting_team_size,
            final_team_size=final_team_size,
            total_timesteps=total_timesteps,
            poke_path=poke_path,
            callback_list=sb3_callbacks.CallbackList(
                callback_list
                + [
                    callbacks.CurriculumCallback(),
                ]
            ),
        )
    env.set_team_size(final_team_size)
    env.change_opponent = True

    training.train(
        env=env,
        model=model,
        total_timesteps=total_timesteps,
        poke_path=poke_path,
        callback_list=sb3_callbacks.CallbackList(
            callback_list
            + [
                callbacks.SuccessCallback(
                    poke_path.agent_dir, poke_path.league_dir, poke_path.tag
                )
            ]
        ),
    )


if __name__ == "__main__":
    cfg_file = pathlib.Path(__file__).parent / "main.yaml"
    cfg = load_config(cfg_file)
    if "team" in cfg:
        try:
            team_list = load_team_from_file(cfg["team"])
            team = AgentTeamBuilder(
                cfg["battle_format"], cfg["starting_team_size"], False
            )
            team.set_team(team_list)
            cfg["teambuilder"] = team
        except RuntimeError as e:
            print("Could not open team file!")
        del cfg["team"]
    main(**cfg)
