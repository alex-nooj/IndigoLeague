import typing

import gym
import numpy as np
from numpy import typing as npt
from poke_env.environment import AbstractBattle
from poke_env.environment import Pokemon

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import calc_move_damage
from indigo_league.training.preprocessing.utils import check_boosts
from indigo_league.training.preprocessing.utils import estimate_matchup
from indigo_league.training.preprocessing.utils import move_helpers
from indigo_league.utils.constants import NUM_MOVES
from indigo_league.utils.constants import NUM_POKEMON


def embed_mon(
    mon: typing.Optional[Pokemon], opponent: typing.Optional[Pokemon]
) -> typing.List[float]:
    return [
        estimate_matchup.determine_type_advantage(mon, opponent)
        if mon is not None and opponent is not None
        else -1.0,
        estimate_matchup.determine_type_advantage(opponent, mon)
        if mon is not None and opponent is not None
        else -1.0,
        estimate_matchup.determine_speed_tier(mon, opponent)
        if mon is not None and opponent is not None
        else -1.0,
        mon.current_hp_fraction if mon is not None else -1.0,
    ]


def embed_active_moves(battle: AbstractBattle) -> typing.List[float]:
    embedding = []
    for ix in range(NUM_MOVES):
        if ix < len(battle.available_moves) and battle.available_moves[ix] is not None:
            embedding += [
                move_helpers.check_hazard_move(
                    battle.available_moves[ix], battle.opponent_side_conditions
                ),
                move_helpers.check_setup_move(
                    battle.available_moves[ix], battle.side_conditions
                ),
                move_helpers.check_removal_move(
                    battle.available_moves[ix], battle.side_conditions
                ),
                move_helpers.check_boost_move(
                    battle.available_moves[ix], battle.active_pokemon.boosts
                ),
                calc_move_damage(
                    move=battle.available_moves[ix],
                    usr=battle.active_pokemon,
                    tgt=battle.opponent_active_pokemon,
                    weather=list(battle.weather.keys())[0]
                    if len(battle.weather) > 0
                    else None,
                    side_conditions=list(battle.opponent_side_conditions.keys()),
                ),
                move_helpers.check_status_move(
                    battle.available_moves[ix],
                    battle.active_pokemon,
                    battle.opponent_active_pokemon,
                    list(battle.opponent_team.values()),
                    battle.fields,
                    battle.weather,
                ),
                battle.available_moves[ix].accuracy,
            ]
        else:
            embedding += [-1.0 for _ in range(7)]

    return embedding


class HeuristicsOp(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len,
            n_features=5 + NUM_MOVES * 7 + NUM_POKEMON * 4,
            key="HeuristicsOp",
        )

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        embedding = [
            estimate_matchup.determine_remaining_mons(battle.team),
            estimate_matchup.determine_remaining_mons(battle.opponent_team),
            check_boosts.check_defense(battle.active_pokemon),
            check_boosts.check_attack(battle.active_pokemon),
            battle.opponent_active_pokemon.current_hp_fraction,
        ]
        embedding += embed_active_moves(battle)

        embedding += embed_mon(battle.active_pokemon, battle.opponent_active_pokemon)
        for ix in range(NUM_POKEMON - 1):
            embedding += embed_mon(
                mon=battle.available_switches[ix]
                if ix < len(battle.available_switches)
                else None,
                opponent=battle.opponent_active_pokemon,
            )
        return embedding

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
