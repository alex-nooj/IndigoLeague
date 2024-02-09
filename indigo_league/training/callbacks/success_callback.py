import pathlib

import stable_baselines3.common.callbacks as sb3_callbacks
import torch
from omegaconf import OmegaConf


class SuccessCallback(sb3_callbacks.BaseCallback):
    def __init__(
        self,
        agent_dir: pathlib.Path,
        league_dir: pathlib.Path,
        tag: str,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self._agent_dir = agent_dir
        self._league_dir = league_dir
        self._tag = tag
        # Add 3 for the number of scripted agents
        self._n_league_agents = len([d for d in league_dir.iterdir() if d.is_dir()]) + 1
        self._next_step = False

    def _on_step(self) -> bool:
        if self._next_step:
            self._next_step = False
            league_agents_beat = []
            for agent, win_rate in self.training_env.envs[0].env.win_rates.items():
                total_wins = sum(win_rate)
                self.logger.record(f"win_rates/{agent}", total_wins)
                league_agents_beat.append(total_wins / float(win_rate.maxlen) > 0.6)
            OmegaConf.save(
                config=OmegaConf.create(
                    {
                        k: sum(v)
                        for k, v in self.training_env.envs[0].env.win_rates.items()
                    }
                ),
                f=(self._agent_dir / "win_rates.yaml"),
            )
            self.logger.record(
                f"trueskill/mu",
                self.training_env.envs[0]
                .env.matchmaker.agent_skills[self.training_env.envs[0].env.tag]
                .mu,
            )
            self.logger.record(
                f"trueskill/sigma",
                self.training_env.envs[0]
                .env.matchmaker.agent_skills[self.training_env.envs[0].env.tag]
                .sigma,
            )
            if sum(league_agents_beat) >= 0.7 * self._n_league_agents:
                (self._league_dir / self._tag).mkdir(parents=True, exist_ok=True)
                self.model.save(self._league_dir / self._tag / "network.zip")
                torch.save(
                    {
                        "team": self.training_env.envs[0].env.agent._team,
                        "preprocessor": self.training_env.envs[0].env.preprocessor,
                    },
                    self._league_dir / self._tag / "team.pth",
                )
                self.training_env.envs[0].env.matchmaker.save()
                self.training_env.envs[0].env.close()
                return False
        self._next_step = self.locals["dones"][0]
        return True
