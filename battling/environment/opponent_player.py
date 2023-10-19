from collections import Counter

from poke_env import PlayerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.player import BattleOrder
from poke_env.player import Player
from stable_baselines3 import PPO

from battling.environment.preprocessing.preprocessor import Preprocessor
from battling.environment.teams.team_builder import AgentTeamBuilder

_PLAYER_COUNTER = Counter()


class OpponentPlayer(Player):
    def __init__(
        self,
        model: PPO,
        preprocessor: Preprocessor,
        team: AgentTeamBuilder,
        tag: str,
        team_size: int,
        *args,
        **kwargs,
    ):
        _PLAYER_COUNTER.update([tag])
        team.set_team_size(team_size)
        super().__init__(team=team, player_configuration=PlayerConfiguration(f"{tag} {_PLAYER_COUNTER[tag]}", None), *args, **kwargs)
        self._model = model
        self._preprocessor = preprocessor

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        obs = self._preprocessor.embed_battle(battle=battle)
        action, _ = self._model.predict(obs)
        return self.action_to_move(action, battle)

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        if (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 4])
        else:
            return self.choose_random_move(battle)
