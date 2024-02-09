from collections import Counter

import numpy as np
import numpy.typing as npt
from poke_env import PlayerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.environment import Effect
from poke_env.environment import Status
from poke_env.player import BattleOrder
from poke_env.player import Player
from sb3_contrib import MaskablePPO

from indigo_league.teams.team_builder import AgentTeamBuilder
from indigo_league.training.preprocessing.preprocessor import Preprocessor
from indigo_league.utils.constants import NUM_MOVES

_PLAYER_COUNTER = Counter()


class OpponentPlayer(Player):
    def __init__(
        self,
        model: MaskablePPO,
        preprocessor: Preprocessor,
        team: AgentTeamBuilder,
        tag: str,
        team_size: int,
        *args,
        **kwargs,
    ):
        _PLAYER_COUNTER.update([tag])
        team.set_team_size(team_size)
        super().__init__(
            team=team,
            player_configuration=PlayerConfiguration(
                f"{tag} {_PLAYER_COUNTER[tag]}", None
            ),
            *args,
            **kwargs,
        )
        self._model = model
        self._preprocessor = preprocessor

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        obs = self._preprocessor.embed_battle(battle=battle)
        action, _ = self._model.predict(obs, action_masks=self.action_masks(battle))
        return self.action_to_move(int(action), battle)

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        if (
            action < NUM_MOVES
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 4])
        else:
            return self.choose_random_move(battle)

    def action_masks(self, battle: AbstractBattle) -> npt.NDArray:
        moves = np.zeros(4)
        if battle.active_pokemon is not None:
            for ix, move in enumerate(battle.active_pokemon.moves.values()):
                if move.id in [m.id for m in battle.available_moves]:
                    if (
                        move.id.lower() == "substitute"
                        and Effect.SUBSTITUTE in battle.active_pokemon.effects
                    ):
                        moves[ix] = 0
                    else:
                        moves[ix] = 1 if move.current_pp != 0 else 0
        team = np.zeros(6)
        team_mon_names = list(battle.team.keys())

        for ix, mon in enumerate(team_mon_names):
            team[ix] = int(
                battle.team[mon].status != Status.FNT and not battle.team[mon].active
            )
        return np.concatenate([moves, team])
