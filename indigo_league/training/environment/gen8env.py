import collections
import logging
import typing
from logging.handlers import RotatingFileHandler

import gym
import numpy.typing as npt
import poke_env
from poke_env import PlayerConfiguration
from poke_env.environment import AbstractBattle
from poke_env.environment import Battle
from poke_env.player import BattleOrder
from poke_env.player.openai_api import ActionType
from poke_env.player.openai_api import ObservationType

from indigo_league.teams.team_builder import AgentTeamBuilder
from indigo_league.training.environment.matchmaker import Matchmaker
from indigo_league.training.environment.utils.action_masking import action_masks
from indigo_league.training.preprocessing.preprocessor import Preprocessor
from indigo_league.utils.constants import NUM_MOVES
from indigo_league.utils.constants import NUM_POKEMON
from indigo_league.utils.directory_helper import PokePath
from indigo_league.utils.load_player import load_player


class Gen8Env(poke_env.player.Gen8EnvSinglePlayer):
    _ACTION_SPACE = list(range(NUM_MOVES + NUM_POKEMON))

    def __init__(
        self,
        ops: typing.Union[typing.Dict[str, typing.Dict[str, typing.Any]], Preprocessor],
        seq_len: int,
        poke_path: PokePath,
        fainted_value: float,
        hp_value: float,
        status_value: float,
        victory_value: float,
        battle_format: str,
        team_size: int,
        *args,
        team: typing.Optional[AgentTeamBuilder] = None,
        change_opponent: bool = False,
        starting_opponent: str = "RandomPlayer",
        **kwargs,
    ):
        if isinstance(ops, Preprocessor):
            self.preprocessor = ops
        else:
            self.preprocessor = Preprocessor(ops, seq_len=seq_len)

        if team is None:
            team = AgentTeamBuilder(battle_format=battle_format, team_size=team_size)
        self.matchmaker = Matchmaker(
            tag=poke_path.tag.rsplit(" ")[0],
            league_path=poke_path.league_dir,
            battle_format=battle_format,
            team_size=team_size,
        )
        self.league_path = poke_path.league_dir
        self.team_size = team_size
        self.win_rates = {}
        self.change_opponent = change_opponent
        self._opp_tag = starting_opponent
        self._next_tag = starting_opponent

        super().__init__(
            battle_format=battle_format,
            team=team,
            player_configuration=PlayerConfiguration(poke_path.tag, None),
            opponent=load_player(
                tag=starting_opponent,
                league_path=poke_path.league_dir,
                battle_format=battle_format,
                team_size=team_size,
            ),
            *args,
            **kwargs,
        )
        self._tag = poke_path.tag
        self._reward_values = {
            "fainted_value": fainted_value,
            "hp_value": hp_value,
            "status_value": status_value,
            "victory_value": victory_value,
        }

        self.tag = poke_path.tag.rsplit(" ")[0]

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

        handler = RotatingFileHandler(
            filename=poke_path.agent_dir / f"{poke_path.tag.lower()}.log",
            maxBytes=1024 * 1024 * 5,
            backupCount=3,
        )
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

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
        self._logger.debug(f"Action: {action}")
        obs, reward, done, info = super().step(action=action)
        self._logger.debug(f"Obs: {obs}")
        self._logger.debug(f"Reward: {reward}")
        self._logger.debug(f"Done: {done}")

        if done:
            self._choose_next_player()

        return obs, reward, done, info

    def reset(self, *args, **kwargs) -> typing.Dict[str, npt.NDArray]:
        # Reset the preprocessor
        self.preprocessor.reset()
        return super().reset(*args, **kwargs)

    def action_masks(self, *args, **kwargs) -> npt.NDArray:
        mask = action_masks(self.current_battle)
        self._logger.debug(f"Mask: {mask}")
        return mask

    def action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        action_mask = self.action_masks()
        if action_mask[action]:
            if action < NUM_MOVES:
                self._logger.debug(
                    f"Action {action} interpreted as a move ({list(battle.active_pokemon.moves.keys())[action]}"
                )
                return self.agent.create_order(
                    list(battle.active_pokemon.moves.values())[action]
                )
            else:
                self._logger.debug(
                    f"Action {action} interpreted as a switch ({list(battle.team.values())[action - NUM_MOVES]}"
                )
                return self.agent.create_order(
                    list(battle.team.values())[action - NUM_MOVES]
                )
        else:
            self._logger.debug(f"Had to choose random action (given {action})")
            return self.agent.choose_random_move(battle)

    def set_team_size(self, team_size: int):
        self.team_size = team_size
        self.agent._team.set_team_size(team_size)
        self.matchmaker.set_team_size(team_size)
        self._opponent._team.set_team_size(team_size)
        self.win_rates = {}

    def _choose_next_player(self):
        # Track the rolling win/loss rate against each opponent
        if self._opp_tag not in self.win_rates:
            self.win_rates[self._opp_tag] = collections.deque(maxlen=100)
        self.win_rates[self._opp_tag].append(int(self.current_battle.won))

        # Update the skill ratings and choose our next opponent
        next_tag, player = self.matchmaker.update_and_choose(
            self._opp_tag, self.current_battle.won
        )

        if self.change_opponent:
            self._opp_tag = self._next_tag
            if next_tag != self._next_tag:
                self._next_tag = next_tag
                self.set_opponent(player)
            self._next_tag = next_tag
