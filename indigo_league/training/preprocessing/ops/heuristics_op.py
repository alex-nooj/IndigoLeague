import typing

import gym
import numpy as np
from numpy import typing as npt
from poke_env.environment import AbstractBattle
from poke_env.environment import Pokemon
from poke_env.environment import SideCondition

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import calc_move_damage
from indigo_league.utils.constants import NUM_MOVES
from indigo_league.utils.constants import NUM_POKEMON

ENTRY_HAZARDS = {
    "spikes": SideCondition.SPIKES,
    "stealhrock": SideCondition.STEALTH_ROCK,
    "stickyweb": SideCondition.STICKY_WEB,
    "toxicspikes": SideCondition.TOXIC_SPIKES,
}
ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}


class HeuristicsOp(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len,
            n_features=2
            + (NUM_POKEMON * NUM_MOVES)
            + 1
            + 1
            + 1
            + NUM_MOVES
            + NUM_MOVES
            + NUM_MOVES
            + NUM_MOVES,
            key="HeuristicsOp",
        )

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        should_switch = self._should_switch_out(battle)
        switch_options = [-1.0 for _ in range((NUM_POKEMON - 1) * NUM_MOVES)]
        for ix, switches in enumerate(battle.available_switches):
            switch_options[
                ix * NUM_MOVES : (ix + 1) * NUM_MOVES
            ] = self._estimate_matchup(switches, battle.opponent_active_pokemon)
        n_remaining_mons = len(
            [m for m in battle.team.values() if m.fainted is False]
        ) / float(NUM_POKEMON)
        n_opp_remaining_mons = (
            NUM_POKEMON - len([m for m in battle.team.values() if m.fainted is True])
        ) / float(NUM_POKEMON)
        hp_fraction = battle.opponent_active_pokemon.current_hp_fraction
        setup_moves = [-1.0 for _ in range(NUM_MOVES)]
        removal_moves = [-1.0 for _ in range(NUM_MOVES)]
        boost_moves = [-1.0 for _ in range(NUM_MOVES)]
        dmg_moves = [-1.0 for _ in range(NUM_MOVES)]
        for ix, move in enumerate(battle.available_moves):
            if ix >= len(setup_moves):
                break
            setup_moves[ix] = float(
                move.id in ENTRY_HAZARDS
                and ENTRY_HAZARDS[move.id] not in battle.opponent_side_conditions
            )
            if move.id in ANTI_HAZARDS_MOVES and battle.side_conditions:
                removal_moves[ix] = 1.0
            else:
                removal_moves[ix] = 0.0

            if (
                move.boosts
                and sum(move.boosts.values()) >= 2
                and move.target == "self"
                and min(
                    [
                        battle.active_pokemon.boosts[s]
                        for s, v in move.boosts.items()
                        if v > 0
                    ]
                )
                < 6
            ):
                boost_moves[ix] = 1.0
            else:
                boost_moves[ix] = 0.0

            dmg_moves[ix] = calc_move_damage(
                move=move,
                usr=battle.active_pokemon,
                tgt=battle.opponent_active_pokemon,
                weather=list(battle.weather.keys())[0]
                if len(battle.weather) > 0
                else None,
                side_conditions=list(battle.opponent_side_conditions.keys()),
            )
        return (
            should_switch
            + switch_options
            + [n_remaining_mons, n_opp_remaining_mons, hp_fraction]
            + setup_moves
            + removal_moves
            + boost_moves
            + dmg_moves
        )

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon) -> typing.List[float]:
        type_adv = max(
            [opponent.damage_multiplier(t) for t in mon.types if t is not None]
        )
        type_dis = max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        speed_tier = 0.0
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            speed_tier = 1.0
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            speed_tier = -1.0

        return [type_adv, type_dis, speed_tier, mon.current_hp_fraction]

    def _should_switch_out(self, battle: AbstractBattle) -> typing.List[float]:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        def_bad = 0.0
        atk_bad = 0.0

        if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
            def_bad = 1.0
        if active.boosts["atk"] <= -3 and active.stats["atk"] >= active.stats["spa"]:
            atk_bad = 1.0
        if active.boosts["spa"] <= -3 and active.stats["atk"] <= active.stats["spa"]:
            atk_bad = 1.0

        return [def_bad, atk_bad] + self._estimate_matchup(active, opponent)

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
