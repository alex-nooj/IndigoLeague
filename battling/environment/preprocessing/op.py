import abc

import gym
import numpy.typing as npt
from poke_env.environment import AbstractBattle
import typing


class Op(abc.ABC):
    @abc.abstractmethod
    def embed_battle(self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]) -> typing.Dict[str, npt.NDArray]:
        ...

    @abc.abstractmethod
    def describe_embedding(self) -> gym.spaces.Dict:
        ...

    def reset(self):
        pass

    def embedding_infos(self) -> typing.Dict[str, typing.Tuple[int, int]]:
        return {}
