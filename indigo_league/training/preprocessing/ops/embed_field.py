import typing

import gym
import numpy as np
import numpy.typing as npt
from memory_profiler import profile
from poke_env.environment import AbstractBattle
from poke_env.environment import Field
from poke_env.environment import SideCondition
from pympler import asizeof

from indigo_league.training.preprocessing.op import Op

MEANINGFUL_SIDE_CONDITIONS = [
    SideCondition.AURORA_VEIL,
    SideCondition.LIGHT_SCREEN,
    SideCondition.REFLECT,
    SideCondition.SAFEGUARD,
    SideCondition.SPIKES,
    SideCondition.STEALTH_ROCK,
    SideCondition.STICKY_WEB,
    SideCondition.TAILWIND,
    SideCondition.TOXIC_SPIKES,
]


MEANINGFUL_FIELD = [
    Field.ELECTRIC_TERRAIN,
    Field.GRASSY_TERRAIN,
    Field.MAGIC_ROOM,
    Field.MISTY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
    Field.TRICK_ROOM,
    Field.WONDER_ROOM,
]


class EmbedField(Op):
    def __init__(self, seq_len: int):
        n_features = 2 * len(MEANINGFUL_SIDE_CONDITIONS) + len(MEANINGFUL_FIELD)
        super().__init__(seq_len=seq_len, n_features=n_features, key="field")
        print(__name__, asizeof.asizeof(self) / 1e9)

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        side_conditions = [
            float(s in battle.side_conditions) for s in MEANINGFUL_SIDE_CONDITIONS
        ]
        opp_side_conditions = [
            float(s in battle.opponent_side_conditions)
            for s in MEANINGFUL_SIDE_CONDITIONS
        ]
        field = [float(f in battle.fields) for f in MEANINGFUL_FIELD]

        return side_conditions + opp_side_conditions + field

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.asarray([0.0 for _ in range(self.seq_len * self.n_features)]),
                    np.asarray([1.0 for _ in range(self.seq_len * self.n_features)]),
                    dtype=np.float32,
                )
            }
        )
