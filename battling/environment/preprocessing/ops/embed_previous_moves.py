import typing

import gym
import numpy.typing as npt
from poke_env.environment import AbstractBattle

from battling.environment.preprocessing.op import Op
import numpy as np


def find_prev_move(
    prev_mon: str, curr_mon: str, prev_pps: typing.List[int], curr_pps: typing.List[int]
) -> typing.List[float]:
    prev_move = [0.0 for _ in range(5)]

    if prev_mon != curr_mon:
        prev_move[-1] = 1.0
    else:
        for ix, (prev_pp, curr_pp) in enumerate(zip(prev_pps, curr_pps)):
            if prev_pp != curr_pp:
                prev_move[ix] = 1.0
                break
    return prev_move


class EmbedPreviousMoves(Op):
    def __init__(self, seq_len: int):
        super().__init__(seq_len=seq_len, n_features=2 * 5, key="prev_move")
        self.prev_pokemon = ""
        self.prev_move_pp = [0 for _ in range(4)]
        self.prev_opp_pokemon = ""
        self.prev_opp_move_pp = [0 for _ in range(4)]

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        if self.prev_pokemon != "":
            own_prev_move = find_prev_move(
                self.prev_pokemon,
                battle.active_pokemon.species,
                [move.current_pp for move in list(battle.active_pokemon.moves.values())[:4]],
                self.prev_move_pp,
            )
            opp_prev_move = find_prev_move(
                self.prev_opp_pokemon,
                battle.opponent_active_pokemon.species,
                [move.current_pp for move in list(battle.opponent_active_pokemon.moves.values())[:4]],
                self.prev_opp_move_pp,
            )
        else:
            own_prev_move = [0.0 for _ in range(5)]
            opp_prev_move = [0.0 for _ in range(5)]
        self.prev_pokemon = battle.active_pokemon.species
        self.prev_move_pp = [move.current_pp for move in list(battle.active_pokemon.moves.values())[:4]]
        self.opp_prev_pokemon = battle.opponent_active_pokemon.species
        self.opp_prev_move_pp = [
            move.current_pp for move in list(battle.opponent_active_pokemon.moves.values())[:4]
        ]
        return own_prev_move + opp_prev_move

    def _reset(self):
        self.prev_pokemon = ""
        self.prev_move_pp = [0 for _ in range(4)]
        self.prev_opp_pokemon = ""
        self.prev_opp_move_pp = [0 for _ in range(4)]

    def describe_embedding(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.zeros(self.seq_len * self.n_features, dtype=float),
                    np.ones(self.seq_len * self.n_features, dtype=float),
                    dtype=float,
                )
            }
        )
