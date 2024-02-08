import stable_baselines3.common.callbacks as sb3_callbacks
from sb3_contrib import MaskablePPO

from indigo_league.training.environment import Gen8Env
from indigo_league.utils.directory_helper import PokePath


def league(
    env: Gen8Env,
    model: MaskablePPO,
    final_team_size: int,
    total_timesteps: int,
    poke_path: PokePath,
    callback_list: sb3_callbacks.CallbackList,
):
    try:
        env.set_team_size(final_team_size)
        env.change_opponent = True
        model.set_env(env)
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt as e:
        model.save(str(poke_path.agent_dir / f"keyboard_interrupt.zip"))
        raise e
    except RuntimeError as e:
        model.save(str(poke_path.agent_dir / f"runtime_error.zip"))
        raise e
    return model
