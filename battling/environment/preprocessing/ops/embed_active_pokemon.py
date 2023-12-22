import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle
from poke_env.environment import Status

from battling.environment.preprocessing.op import Op
from utils import damage_helpers
from utils.normalize_stats import normalize_stats


class EmbedActivePokemon(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len,
            n_features=2 * (2 + 8 + len(Status)),
            key="active_pokemon",
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

        # Rather than encode the pokemon's types as a 1-hot, we'll instead measure the damage multiplier of the
        # opponent's types against ours (i.e., opponent attacks us)
        types = [
            damage_helpers.type_multiplier("", poke_type, battle.active_pokemon) / 4.0
            if poke_type
            else -1
            for poke_type in battle.opponent_active_pokemon.types
        ]

        stats = normalize_stats(battle.active_pokemon)
        status = [float(t == battle.active_pokemon.status) for t in Status]

        # Now we measure the damage multiplier of our own types against the opponent
        opp_types = [
            damage_helpers.type_multiplier(
                "", poke_type, battle.opponent_active_pokemon
            )
            / 4.0
            if poke_type
            else -1
            for poke_type in battle.active_pokemon.types
        ]

        opp_stats = normalize_stats(battle.opponent_active_pokemon)
        opp_status = [float(t == battle.opponent_active_pokemon.status) for t in Status]

        return types + stats + status + opp_types + opp_stats + opp_status

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
