import importlib
import typing

import gym
import numpy.typing as npt
from poke_env.environment import AbstractBattle


def dynamic_import(target: str) -> typing.Callable:
    target_path = ".".join(target.split(".")[:-1])
    target_module = importlib.import_module(target_path)
    module = getattr(target_module, target.split(".")[-1])
    return module


class Preprocessor:
    def __init__(
        self, ops: typing.Dict[str, typing.Dict[str, typing.Any]], seq_len: int
    ):
        self._ops = []
        self._obs_space = {}
        self._embedding_infos = {}
        for op_path, op_args in ops.items():
            op = dynamic_import(op_path)(seq_len=seq_len, **op_args)
            print(f"{op_path.rsplit('.')[-1]}: {op.n_features}")
            self._ops.append(op)

            self._obs_space.update(op.describe_embedding())
            self._embedding_infos.update(op.embedding_infos())
        self._obs_space = gym.spaces.Dict(self._obs_space)

    def embed_battle(self, battle: AbstractBattle) -> typing.Dict[str, npt.NDArray]:
        state = {}
        for op in self._ops:
            state.update(op.embed_battle(battle, state))
        return state

    def reset(self):
        for op in self._ops:
            op.reset()

    def describe_embedding(self) -> gym.spaces.Dict:
        return self._obs_space

    def embedding_infos(self) -> typing.Dict[str, typing.Tuple[int, int]]:
        return self._embedding_infos
