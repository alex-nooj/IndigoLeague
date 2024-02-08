import typing

import gym
import numpy as np
from numpy import typing as npt
from poke_env.environment import AbstractBattle
from poke_env.environment import Pokemon

from indigo_league.training.preprocessing.op import Op
from indigo_league.utils.constants import NUM_POKEMON


class EmbedSwitchOptions(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len,
            n_features=2 + (NUM_POKEMON * 4),
            key=__name__,
        )

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        active_matchup = self._estimate_matchup(battle.active_pokemon, battle.opponent_active_pokemon)
        switch_options = [-1.0 for _ in range((NUM_POKEMON - 1) * 4)]
        for ix, switches in enumerate(battle.available_switches):
            switch_options[
                ix * 4 : (ix + 1) * 4
            ] = self._estimate_matchup(switches, battle.opponent_active_pokemon)
        n_remaining_mons = len(
            [m for m in battle.team.values() if m.fainted is False]
        ) / float(NUM_POKEMON)
        n_opp_remaining_mons = (
            NUM_POKEMON - len([m for m in battle.team.values() if m.fainted is True])
        ) / float(NUM_POKEMON)
        return (
            active_matchup + switch_options
            + [n_remaining_mons, n_opp_remaining_mons]
        )

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon) -> typing.List[float]:
        type_adv = max(
            [opponent.damage_multiplier(t) for t in mon.types if t is not None]
        )
        type_dis = max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        speed_tier = 1.0 if mon.base_stats["spe"] > opponent.base_stats["spe"] else -1.0

        return [type_adv, type_dis, speed_tier, mon.current_hp_fraction]

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """

        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.array(
                        [-1.0 for _ in range(self.n_features * self.seq_len)],
                        dtype=np.float32,
                    ),
                    np.array(
                        [-1.0 for _ in range(self.n_features * self.seq_len)],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                )
            }
        )
