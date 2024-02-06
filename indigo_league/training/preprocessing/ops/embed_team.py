import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle
from poke_env.environment import Status

from indigo_league.training.preprocessing.op import Op
from indigo_league.training.preprocessing.utils import calc_move_damage
from indigo_league.training.preprocessing.utils import gather_team
from indigo_league.training.preprocessing.utils import normalize_stats
from indigo_league.training.preprocessing.utils import type_multiplier


class EmbedTeam(Op):
    def __init__(self, seq_len: int):
        super().__init__(
            seq_len=seq_len,
            n_features=5 * (2 + 8 + len(Status) + 4 * 2),
            key="team_pokemon",
        )

    def _embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.List[float]:
        pokemon_list = []
        team = gather_team(battle)[1:]
        for pokemon in team:
            # Rather than encode the pokemon's types as a 1-hot, we'll instead measure the damage multiplier of the
            # opponent's types against ours (i.e., opponent attacks us)
            types = [
                type_multiplier("", poke_type, battle.active_pokemon) / 4.0
                if poke_type
                else -1
                for poke_type in battle.opponent_active_pokemon.types
            ]
            stats = normalize_stats(pokemon)
            status = [float(t == pokemon.status) for t in Status]
            # Current move power
            moves = list(pokemon.moves.values())
            if len(moves) > 4:
                moves = moves[:4]
            move_dmg = [
                calc_move_damage(
                    move=move,
                    usr=pokemon,
                    tgt=battle.opponent_active_pokemon,
                    weather=list(battle.weather.keys())[0]
                    if len(battle.weather) > 0
                    else None,
                    side_conditions=list(battle.opponent_side_conditions.keys()),
                )
                for move in moves
            ]
            while len(move_dmg) != 4:
                move_dmg.append(0.0)

            opp_moves = list(battle.opponent_active_pokemon.moves.values())

            if len(opp_moves) > 4:
                opp_moves = opp_moves[:4]
            opp_move_dmg = []
            for move in opp_moves:
                opp_move_dmg.append(
                    calc_move_damage(
                        move=move,
                        usr=battle.opponent_active_pokemon,
                        tgt=pokemon,
                        weather=list(battle.weather.keys())[0]
                        if len(battle.weather) > 0
                        else None,
                        side_conditions=list(battle.side_conditions.keys()),
                    )
                )
            while len(opp_move_dmg) != 4:
                opp_move_dmg.append(0.0)
            pokemon_list += types + stats + status + move_dmg + opp_move_dmg
        while len(pokemon_list) < self.n_features:
            pokemon_list.append(-1.0)
        return pokemon_list

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                self.key: gym.spaces.Box(
                    np.asarray([-1 for _ in range(self.seq_len * self.n_features)]),
                    np.asarray([1 for _ in range(self.seq_len * self.n_features)]),
                    dtype=np.float32,
                )
            }
        )
