import typing

import stable_baselines3.common.callbacks as sb3_callbacks
import torch

from utils import PokePath


class SavePeripheralsCallback(sb3_callbacks.BaseCallback):
    def __init__(self, save_freq: int, poke_path: PokePath, verbose: typing.Optional[int] = 1):
        super().__init__(verbose=verbose)
        self.save_freq = save_freq
        self.poke_path = poke_path

    def _on_training_start(self):
        torch.save(
            {
                "team": self.training_env.envs[0].env.agent._team,
                "preprocessor": self.training_env.envs[0].env.preprocessor,
            },
            self.poke_path.agent_dir / "team.pth"
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.training_env.envs[0].env.matchmaker.save()
        return True
