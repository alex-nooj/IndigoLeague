import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import embed_moves
from indigo_league.utils.constants import NUM_MOVES


class EmbedActiveMoves(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len,
            n_features=2 * NUM_MOVES * (2 + 1 + 1 + 1),
            key="active_moves",
        )

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        """Embeds active and opponent active pokemon in the current state.

        Args:
            battle: Current state of the battle (functional)
            state: Current state of the battle (observation)

        Returns:
            Dict[str, NDArray]: The updated observational state.
        """
        moves = embed_moves(
            moves=battle.active_pokemon.moves.values(),
            usr=battle.active_pokemon,
            tgt=battle.opponent_active_pokemon,
            weather=list(battle.weather.keys())[0] if len(battle.weather) > 0 else None,
            side_conditions=list(battle.opponent_side_conditions.keys()),
        )
        while len(moves) < self.n_features / 2:
            moves.append(-1.0)

        opp_moves = embed_moves(
            moves=list(battle.opponent_active_pokemon.moves.values()),
            usr=battle.opponent_active_pokemon,
            tgt=battle.active_pokemon,
            weather=list(battle.weather.keys())[0] if len(battle.weather) > 0 else None,
            side_conditions=list(battle.side_conditions.keys()),
        )
        while len(opp_moves) < self.n_features / 2:
            opp_moves.append(-1.0)

        return moves + opp_moves

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.asarray([-1 for _ in range(self.n_features * self.seq_len)]),
                    np.asarray([1 for _ in range(self.n_features * self.seq_len)]),
                    dtype=np.float32,
                )
            }
        )
