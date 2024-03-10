import collections
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
        self.win_rates = {"FixedHeuristics": collections.deque(maxlen=100)}
        for d in league_dir.iterdir():
            if d.is_dir():
                self.win_rates[d.stem] = collections.deque(maxlen=100)
        print("== League Agent ==")
        for k in self.win_rates:
            print(k)

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        wins_updated = False
        for info in infos:
            if "win" in info:
                wins_updated = True
                self.win_rates[info["win"]["opp"]].append(int(info["win"]["result"]))
                for k, v in self.win_rates.items():
                    self.logger.record(f"win_rates/{k}", sum(v))

        if wins_updated:
            win_totals = sum(
                [sum(v) >= (0.6 * v.maxlen) for v in self.win_rates.values()]
            )
            self.logger.record("league/success_rate", win_totals / len(self.win_rates))
            if win_totals >= 0.7 * len(self.win_rates):
                (self._league_dir / self._tag).mkdir(parents=True, exist_ok=True)
                self.model.save(self._league_dir / self._tag / "network.zip")

                if hasattr(self.model.env, "envs"):
                    team = self.model.env.envs[0].env.team
                    preprocessor = self.model.env.envs[0].env.preprocessor
                else:
                    team = None
                    preprocessor = None
                torch.save(
                    {
                        "team": team,
                        "preprocessor": preprocessor,
                    },
                    self._league_dir / self._tag / "team.pth",
                )
                return False
        return True

    def _on_rollout_end(self) -> None:
        OmegaConf.save(
            config=OmegaConf.create({k: sum(v) for k, v in self.win_rates.items()}),
            f=(self._agent_dir / "win_rates.yaml"),
        )

    def _on_training_end(self) -> None:
        self._on_rollout_end()
