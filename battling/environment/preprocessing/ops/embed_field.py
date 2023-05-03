import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle, Weather, SideCondition, Field

from battling.environment.preprocessing.op import Op


class EmbedField(Op):
    def embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.Dict[str, npt.NDArray]:
        weather_vec = [0.0 for _ in Weather]
        for weather in battle.weather:
            weather_vec[weather.value - 1] = 1.0

        side_conditions = [0.0 for _ in SideCondition]
        for side_condition in battle.side_conditions:
            side_conditions[side_condition.value - 1] = 1.0

        opp_side_conditions = [0.0 for _ in SideCondition]
        for side_condition in battle.opponent_side_conditions:
            opp_side_conditions[side_condition.value - 1] = 1.0

        field = [0.0 for _ in Field]
        for field_effect in battle.fields:
            field[field_effect.value - 1] = 1.0

        state["field"] = np.asarray(
            weather_vec + side_conditions + opp_side_conditions + field
        )
        return state

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        n_features = len(Weather) + 2 * len(SideCondition) + len(Field)
        return gym.spaces.Dict(
            {
                "field": gym.spaces.Box(
                    np.asarray([0.0 for _ in range(n_features)]),
                    np.asarray([1.0 for _ in range(n_features)]),
                    dtype=np.float32,
                )
            }
        )
