import asyncio
import pathlib
import typing

import stable_baselines3.common.callbacks as sb3_callbacks
import torch
from sb3_contrib import MaskablePPO

import utils
from battling.agent.transformer_feature_extractor import TransformerFeatureExtractor
from battling.callbacks.curriculum_callback import CurriculumCallback
from battling.callbacks.save_peripherals_callback import SavePeripheralsCallback
from battling.callbacks.success_callback import SuccessCallback
from battling.environment.gen8env import Gen8Env
from battling.environment.teams.load_team import load_team_from_file
from battling.environment.teams.team_builder import AgentTeamBuilder
from team_selection.run_genetic_algo import genetic_team_search


def resume_training(
    resume_path: pathlib.Path,
    battle_format: str,
    rewards: typing.Dict[str, float],
) -> typing.Tuple[utils.PokePath, MaskablePPO, Gen8Env, int]:
    try:
        tag, _, _ = resume_path.stem.rsplit("_")
        if tag == "runtime":
            tag = resume_path.parent.stem
    except ValueError:
        tag = resume_path.parent.stem

    poke_path = utils.PokePath(tag=tag)
    print(resume_path)
    team_file = resume_path.parent / "team.pth"
    print(team_file)
    team_info = torch.load(team_file)
    team = team_info["team"]
    preprocessor = team_info["preprocessor"]

    env = Gen8Env(
        preprocessor,
        **rewards,
        tag=tag,
        league_path=poke_path.league_dir,
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


def setup(
    ops: typing.Dict[str, typing.Dict[str, typing.Any]],
    rewards: typing.Dict[str, float],
    battle_format: str,
    seq_len: int,
    shared: typing.List[int],
    pi: typing.List[int],
    vf: typing.List[int],
    starting_team_size: int,
    poke_path: utils.PokePath,
    teambuilder: typing.Optional[AgentTeamBuilder],
    tag: str,
):
    if tag is None:
        tag = poke_path.tag
    if teambuilder is None:
        teambuilder = asyncio.get_event_loop().run_until_complete(
            genetic_team_search(100, 1, battle_format, 1)
        )
    teambuilder.save_team(poke_path.agent_dir)

    env = Gen8Env(
        ops,
        **rewards,
        seq_len=seq_len,
        tag=tag,
        league_path=poke_path.league_dir,
        battle_format=battle_format,
        start_challenging=True,
        team_size=starting_team_size,
        change_opponent=False,
        starting_opponent="SimpleHeuristics",
        team=teambuilder,
    )

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
            activation_fn=torch.nn.LeakyReLU,
        ),
    )

    return env, model


def train(
    env: Gen8Env,
    model: MaskablePPO,
    starting_team_size: int,
    final_team_size: int,
    total_timesteps: int,
    save_freq: int,
    poke_path: utils.PokePath,
    tag: str,
    callbacks: sb3_callbacks.CallbackList = None,
):
    checkpoint_callback = sb3_callbacks.CheckpointCallback(
        save_freq,
        save_path=str(poke_path.agent_dir),
        name_prefix=poke_path.tag,
    )

    try:
        for team_size in range(starting_team_size, final_team_size):
            print(f"Team Size: {team_size}")
            env.set_team_size(team_size)
            model.set_env(env)

            model.learn(
                total_timesteps=total_timesteps,
                callback=sb3_callbacks.CallbackList(
                    [
                        checkpoint_callback,
                        SavePeripheralsCallback(
                            poke_path=poke_path, save_freq=save_freq
                        ),
                        CurriculumCallback(0.7),
                    ]
                ),
                reset_num_timesteps=False,
            )
            model.save(str(poke_path.agent_dir / f"team_size_{team_size}.zip"))

        print(f"Team Size: {final_team_size}")
        env.set_team_size(final_team_size)
        env.change_opponent = True
        if callbacks is None:
            callbacks = [
                checkpoint_callback,
                SavePeripheralsCallback(poke_path=poke_path, save_freq=save_freq),
                SuccessCallback(poke_path.agent_dir, poke_path.league_dir, tag=tag),
            ]
        model.set_env(env)
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt as e:
        model.save(
            str(
                poke_path.agent_dir
                / f"keyboard_interrupt_{checkpoint_callback.num_timesteps}.zip"
            )
        )
        raise e
    except RuntimeError as e:
        model.save(
            str(
                poke_path.agent_dir
                / f"runtime_error_{checkpoint_callback.num_timesteps}.zip"
            )
        )
        raise e


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
    callbacks: sb3_callbacks.CallbackList = None,
    teambuilder: AgentTeamBuilder = None,
):
    if resume is not None and pathlib.Path(resume).is_file():
        poke_path, model, env, starting_team_size = resume_training(
            pathlib.Path(resume), battle_format, rewards
        )
    else:
        poke_path = utils.PokePath(tag=tag)

        env, model = setup(
            ops=ops,
            rewards=rewards,
            battle_format=battle_format,
            seq_len=1,
            shared=shared,
            pi=pi,
            vf=vf,
            starting_team_size=starting_team_size,
            poke_path=poke_path,
            teambuilder=teambuilder,
            tag=tag,
        )

    train(
        env,
        model,
        starting_team_size,
        final_team_size,
        total_timesteps,
        save_freq,
        poke_path,
        poke_path.tag,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    cfg_file = pathlib.Path(__file__).parent / "main.yaml"
    cfg = utils.load_config(cfg_file)
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
