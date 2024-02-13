import typing


class RewardScheduler:
    """Handles the scaling of a single reward value.

    Note that this class does not handle the actual scheduling, just the reduction of the value.
    Providing a scale <1 will result in the value going to 0, >1 will cause the reward to blow up,
    and ==1 causes the reward to remain the same.
    """

    def __init__(self, value: float, scale: float):
        """Initializes the instance for a single reward.

        Args:
            value: The starting weight for this reward
            scale: Number to multiply value by when update() is called.
        """
        self._value, self._scale = value, scale

    def update(self):
        self._value *= self._scale

    @property
    def value(self) -> float:
        return self._value


class RewardHelper:
    def __init__(
        self,
        schedule: int,
        faint: typing.Dict[str, float],
        hp: typing.Dict[str, float],
        status: typing.Dict[str, float],
        victory: typing.Dict[str, float],
    ):
        self.schedule = schedule
        self._reward_values = {
            "fainted_value": RewardScheduler(**faint),
            "hp_value": RewardScheduler(**hp),
            "status_value": RewardScheduler(**status),
            "victory_value": RewardScheduler(**victory),
        }
        self._n_steps = 0

    @property
    def reward_values(self) -> typing.Dict[str, float]:
        self._n_steps = (self._n_steps + 1) % self.schedule
        if self._n_steps == 0:
            # Apply the scaling
            [v.update() for v in self._reward_values.values()]

        return {k: v.value for k, v in self._reward_values.items()}
