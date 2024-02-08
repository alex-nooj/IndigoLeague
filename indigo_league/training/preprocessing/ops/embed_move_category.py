import typing

from numpy import typing as npt
from poke_env.environment import AbstractBattle
from poke_env.environment import SideCondition
import gym
import numpy as np

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import calc_move_damage
from indigo_league.utils.constants import NUM_MOVES

ENTRY_HAZARDS = {
    "spikes": SideCondition.SPIKES,
    "stealhrock": SideCondition.STEALTH_ROCK,
    "stickyweb": SideCondition.STICKY_WEB,
    "toxicspikes": SideCondition.TOXIC_SPIKES,
}
ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}


class EmbedMoveCategory(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len,
            n_features=4 * NUM_MOVES,
            key=__name__,
        )

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
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
                and min([battle.active_pokemon.boosts[s] for s, v in move.boosts.items() if v > 0])
                < 6
            ):
                boost_moves[ix] = 1.0
            else:
                boost_moves[ix] = 0.0

            dmg_moves[ix] = calc_move_damage(
                move=move,
                usr=battle.active_pokemon,
                tgt=battle.opponent_active_pokemon,
                weather=list(battle.weather.keys())[0] if len(battle.weather) > 0 else None,
                side_conditions=list(battle.opponent_side_conditions.keys()),
            )

        return setup_moves + removal_moves + boost_moves + dmg_moves

    def describe_embedding(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.array([-1.0 for _ in range(self.n_features * self.seq_len)]),
                    np.array([1.0 for _ in range(self.n_features * self.seq_len)]),
                    dtype=np.float32,
                )
            }
        )
