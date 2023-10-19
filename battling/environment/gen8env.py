import collections
import pathlib
import typing
import numpy as np

import gym
import numpy.typing as npt
import poke_env
from poke_env import PlayerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.environment import Battle
from poke_env.environment import Status
from poke_env.player import BattleOrder
from poke_env.player.openai_api import ActionType, ObservationType

from battling.environment.matchmaking.matchmaker import Matchmaker
from battling.environment.preprocessing.preprocessor import Preprocessor
from battling.environment.teams.team_builder import AgentTeamBuilder


class Gen8Env(poke_env.player.Gen8EnvSinglePlayer):
    _ACTION_SPACE = list(range(4 + 6))

    def __init__(
        self,
        ops: typing.Union[typing.Dict[str, typing.Dict[str, typing.Any]], Preprocessor],
        seq_len: int,
        tag: str,
        league_path: pathlib.Path,
        fainted_value: float,
        hp_value: float,
        status_value: float,
        victory_value: float,
        battle_format: str,
        team_size: int,
        *args,
        team: typing.Optional[AgentTeamBuilder] = None,
        **kwargs
    ):
        if isinstance(ops, Preprocessor):
            self.preprocessor = ops
        else:
            self.preprocessor = Preprocessor(ops, seq_len=seq_len)

        if team is None:
            team = AgentTeamBuilder(battle_format=battle_format, team_size=team_size)
        super().__init__(
            battle_format=battle_format,
            team=team,
            player_configuration=PlayerConfiguration(tag, None),
            *args,
            **kwargs
        )
        self._tag = tag
        self._reward_values = {
            "fainted_value": fainted_value,
            "hp_value": hp_value,
            "status_value": status_value,
            "victory_value": victory_value,
        }
        self.matchmaker = Matchmaker(
            tag=tag.rsplit(" ")[0],
            league_path=league_path,
            battle_format=battle_format,
            team_size=team_size,
        )
        self.win_rates = {}
        self._opp_tag = "RandomPlayer"
        self._next_tag = "RandomPlayer"
        self.tag = tag.rsplit(" ")[0]

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(current_battle, **self._reward_values)

    def embed_battle(self, battle: AbstractBattle) -> typing.Dict[str, npt.NDArray]:
        return self.preprocessor.embed_battle(battle)

    def describe_embedding(self) -> gym.spaces.Space:
        return self.preprocessor.describe_embedding()

    def step(
        self, action: ActionType
    ) -> typing.Union[
        typing.Tuple[ObservationType, float, bool, bool, dict],
        typing.Tuple[ObservationType, float, bool, dict],
    ]:
        obs, reward, done, info = super().step(action=action)
        if done:
            # Track the rolling win/loss rate against each opponent
            if self._opp_tag not in self.win_rates:
                self.win_rates[self._opp_tag] = collections.deque(maxlen=100)
            self.win_rates[self._opp_tag].append(int(self.current_battle.won))

            # Update the skill ratings and choose our next opponent
            next_tag, player = self.matchmaker.update_and_choose(
                self._opp_tag, self.current_battle.won
            )
            self._opp_tag = self._next_tag
            self._next_tag = next_tag
            self.set_opponent(player)
        return obs, reward, done, info

    def reset(self, *args, **kwargs) -> typing.Dict[str, npt.NDArray]:
        # Reset the preprocessor
        self.preprocessor.reset()
        return super().reset(*args, **kwargs)

    def action_masks(self, *args, **kwargs) -> npt.NDArray:
        battle = self.current_battle
        moves = np.zeros(4)
        if battle.active_pokemon is not None:
            for ix, move in enumerate(battle.active_pokemon.moves.values()):
                if move.id in [m.id for m in battle.available_moves]:
                    moves[ix] = 1 if move.current_pp != 0 else 0
        team = np.zeros(6)
        team_mon_names = list(battle.team.keys())
        team_mon_names.sort()

        for ix, mon in enumerate(team_mon_names):
            team[ix] = int(battle.team[mon].status != Status.FNT and not battle.team[mon].active)
        return np.concatenate([moves, team])

    def action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        action_mask = self.action_masks()
        if action_mask[action]:
            if action < 4:
                return self.agent.create_order(
                    list(battle.active_pokemon.moves.values())[action]
                )
            else:
                return self.agent.create_order(list(battle.team.values())[action - 4])
        else:
            return self.agent.choose_random_move(battle)

    def set_team_size(self, team_size: int):
        self.agent._team.set_team_size(team_size)
        self.matchmaker.set_team_size(team_size)
        self.win_rates = {}
