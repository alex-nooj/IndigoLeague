import abc
import typing
from collections import deque

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle


class Op(abc.ABC):
    def __init__(self, seq_len: int, n_features: int, key: str):
        self.seq_len = seq_len
        self.n_features = n_features
        self.key = key.rsplit(".")[-1]
        self.frames = deque(maxlen=seq_len)
        self.reset()

    @abc.abstractmethod
    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        ...

    def embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.Dict[str, npt.NDArray]:
        self.frames.append(self._embed_battle(battle, state))
        ret_list = []
        for frame in self.frames:
            ret_list += frame
        ret_val = {self.key: np.asarray(ret_list)}
        return ret_val

    @abc.abstractmethod
    def describe_embedding(self) -> gym.spaces.Dict:
        ...

    def _reset(self):
        pass

    def reset(self):
        self._reset()
        self.frames.clear()
        for _ in range(self.seq_len):
            self.frames.append([0.0 for _ in range(self.n_features)])

    def embedding_infos(self) -> typing.Dict[str, typing.Tuple[int, int]]:
        return {}
