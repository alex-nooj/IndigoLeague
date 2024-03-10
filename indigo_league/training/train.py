import gc

from sb3_contrib import MaskablePPO
from stable_baselines3.common import callbacks as sb3_callbacks

from indigo_league.training.environment import Gen8Env
from indigo_league.utils.directory_helper import PokePath


def train(
    env: Gen8Env,
    model: MaskablePPO,
    total_timesteps: int,
    poke_path: PokePath,
    callback_list: sb3_callbacks.CallbackList,
    starting_step: int = 0,
) -> int:
    try:
        model.set_env(env)
        model.learn(
            total_timesteps=total_timesteps - starting_step,
            callback=callback_list,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt as e:
        model.save(str(poke_path.agent_dir / f"keyboard_interrupt.zip"))
        raise e
    except RuntimeError as e:
        model.save(str(poke_path.agent_dir / f"runtime_error.zip"))
        raise e

    return model.num_timesteps


def curriculum(
    env: Gen8Env,
    model: MaskablePPO,
    starting_team_size: int,
    final_team_size: int,
    total_timesteps: int,
    poke_path: PokePath,
    callback_list: sb3_callbacks.CallbackList,
) -> int:

    starting_step = 0
    for team_size in range(starting_team_size, final_team_size + 1):
        print(f"Team Size: {team_size}")
        env.set_team_size(team_size)
        train(
            env=env,
            model=model,
            total_timesteps=total_timesteps,
            poke_path=poke_path,
            callback_list=callback_list,
            starting_step=starting_step,
        )
        env.reset_battles()
        gc.collect()  # Poke Env's JSON decoders don't go away on their own

    return starting_step
