import collections

import stable_baselines3.common.callbacks as sb3_callbacks


class CurriculumCallback(sb3_callbacks.BaseCallback):
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.win_rates = {}
        self.queue_len = 100

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "win" in info:
                opp_tag = info["win"]["opp"]
                if opp_tag not in self.win_rates:
                    self.win_rates[opp_tag] = collections.deque(maxlen=self.queue_len)
                self.win_rates[opp_tag].append(int(info["win"]["result"]))

        if (
            "FixedHeuristics" in self.win_rates
            and sum(self.win_rates["FixedHeuristics"]) > 0.5 * self.queue_len
        ):
            self.win_rates = {}
            return False
        return True

    def _on_rollout_end(self) -> None:
        for k, v in self.win_rates.items():
            self.logger.record(f"win_rates/{k}", sum(v))
        self.logger.record("team_size", self.locals["infos"][0]["team_size"])

    def _on_training_end(self) -> None:
        self.win_rates = {}
