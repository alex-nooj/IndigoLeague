import typing

import stable_baselines3.common.callbacks as sb3_callbacks


class RunnerCheck:
    def __init__(self):
        self.continue_running = True


class ControllerCallback(sb3_callbacks.BaseCallback):
    def __init__(
        self, continue_running: RunnerCheck, verbose: typing.Optional[int] = 1
    ):
        super().__init__(verbose=verbose)
        self.continue_running = continue_running

    def _on_step(self) -> bool:
        return self.continue_running.continue_running
