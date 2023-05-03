import collections
import pathlib
import typing

import gym
import numpy.typing as npt
import poke_env
from poke_env import PlayerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.player import BattleOrder
from poke_env.player.openai_api import ActionType, ObservationType

from battling.environment.matchmaking.matchmaker import Matchmaker
from battling.environment.preprocessing.preprocessor import Preprocessor
from battling.environment.teams.team_builder import AgentTeamBuilder


class Gen8Env(poke_env.player.Gen8EnvSinglePlayer):
    _ACTION_SPACE = list(range(4 + 5))

    def __init__(
        self,
        ops: typing.Union[typing.Dict[str, typing.Dict[str, typing.Any]], Preprocessor],
        tag: str,
        league_path: pathlib.Path,
        fainted_value: float,
        hp_value: float,
        status_value: float,
        victory_value: float,
        battle_format: str,
        *args,
        team_path: typing.Optional[str] = None,
        team: typing.Optional[AgentTeamBuilder] = None,
        **kwargs
    ):
        if isinstance(ops, Preprocessor):
            self.preprocessor = ops
        else:
            self.preprocessor = Preprocessor(ops)

        if team is None:
            team = AgentTeamBuilder(battle_format=battle_format, team_path=team_path)
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
            tag=tag.rsplit(" ")[0], league_path=league_path, battle_format=battle_format
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

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        if (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.agent.create_order(battle.available_moves[action])
        elif 0 <= action - 4 < len(battle.available_switches):
            return self.agent.create_order(battle.available_switches[action - 4])
        else:
            return self.agent.choose_random_move(battle)
