import stable_baselines3.common.callbacks as sb3_callbacks


class CurriculumCallback(sb3_callbacks.BaseCallback):
    def __init__(self, threshold: float, verbose: int = 1):
        super().__init__(verbose)
        self.threshold = threshold
        self.next_step = False

    def _on_step(self) -> bool:
        if self.next_step:
            self._next_step = False
            league_agents_beat = []
            self.logger.record(f"train/team_size", self.training_env.envs[0].env.matchmaker.team_size)
            win_rates = self.training_env.envs[0].env.win_rates
            for agent, win_rate in win_rates.items():
                total_wins = sum(win_rate)
                self.logger.record(f"win_rates/{agent}", total_wins)
                league_agents_beat.append(int(total_wins > (win_rate.maxlen / 2)))
            if sum(league_agents_beat) == 3:
                self.training_env.envs[0].env.matchmaker.reset()
                return False
        self.next_step = self.locals["dones"][0]
        return True
