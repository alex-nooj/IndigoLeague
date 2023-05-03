import typing

import gym
import numpy.typing as npt
from poke_env.environment import AbstractBattle

from utils.dynamic_import import dynamic_import


class Preprocessor:
    def __init__(self, ops: typing.Dict[str, typing.Dict[str, typing.Any]]):
        self._ops = []
        self._obs_space = {}
        self._embedding_infos = {}
        for op_path, op_args in ops.items():
            op = dynamic_import(op_path)(**op_args)
            self._ops.append(op)
            self._obs_space.update(op.describe_embedding())
            self._embedding_infos.update(op.embedding_infos())
        self._obs_space = gym.spaces.Dict(self._obs_space)

    def embed_battle(self, battle: AbstractBattle) -> typing.Dict[str, npt.NDArray]:
        state = {}
        for op in self._ops:
            state = op.embed_battle(battle, state)
        return state

    def reset(self):
        for op in self._ops:
            op.reset()

    def describe_embedding(self) -> gym.spaces.Space:
        return self._obs_space

    def embedding_infos(self) -> typing.Dict[str, typing.Tuple[int, int]]:
        return self._embedding_infos
