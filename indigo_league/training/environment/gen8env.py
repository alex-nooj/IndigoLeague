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
from pympler import asizeof

from indigo_league.teams.team_builder import AgentTeamBuilder
from indigo_league.training.environment.matchmaker import Matchmaker
from indigo_league.training.environment.utils.action_masking import action_masks
from indigo_league.training.environment.utils.load_player import load_player
from indigo_league.training.environment.utils.reward_scheduler import RewardHelper
from indigo_league.training.preprocessing.preprocessor import Preprocessor
from indigo_league.utils.constants import NUM_MOVES
from indigo_league.utils.constants import NUM_POKEMON
from indigo_league.utils.directory_helper import PokePath


def build_env(
    ops: typing.Union[typing.Dict[str, typing.Dict[str, typing.Any]], Preprocessor],
    seq_len: int,
    poke_path: PokePath,
    schedule: int,
    fainted_value: typing.Dict[str, float],
    hp_value: typing.Dict[str, float],
    status_value: typing.Dict[str, float],
    victory_value: typing.Dict[str, float],
    battle_format: str,
    team_size: int,
    team: typing.Optional[AgentTeamBuilder] = None,
    change_opponent: bool = False,
    starting_opponent: str = "FixedHeuristics",
):
    if isinstance(ops, Preprocessor):
        preprocessor = ops
    else:
        preprocessor = Preprocessor(ops, seq_len=seq_len)
    reward_helper = RewardHelper(
        schedule=schedule,
        faint=fainted_value,
        hp=hp_value,
        status=status_value,
        victory=victory_value,
    )
    matchmaker = Matchmaker(
        tag=poke_path.tag.split(" ")[0],
        league_path=poke_path.league_dir,
        battle_format=battle_format,
        team_size=team_size,
    )

    if team is None:
        team = AgentTeamBuilder(battle_format=battle_format, team_size=team_size)
    else:
        team.set_team_size(team_size)

    return Gen8Env(
        poke_path=poke_path,
        preprocessor=preprocessor,
        reward_helper=reward_helper,
        matchmaker=matchmaker,
        battle_format=battle_format,
        team=team,
        change_opponent=change_opponent,
        start_challenging=starting_opponent,
    )


class Gen8Env(poke_env.player.Gen8EnvSinglePlayer):
    _ACTION_SPACE = list(range(NUM_MOVES + NUM_POKEMON))

    def __init__(
        self,
        poke_path: PokePath,
        preprocessor: Preprocessor,
        reward_helper: RewardHelper,
        matchmaker: Matchmaker,
        battle_format: str,
        team: AgentTeamBuilder,
        *args,
        change_opponent: bool = False,
        starting_opponent: str = "FixedHeuristics",
        **kwargs,
    ):
        self.preprocessor = preprocessor
        self.matchmaker = matchmaker
        self.league_path = poke_path.league_dir
        self.team_size = team.team_size
        self.win_rates = {}
        self.change_opponent = change_opponent
        self._opp_tag = starting_opponent
        self._next_tag = starting_opponent
        self.team = team

        super().__init__(
            battle_format=battle_format,
            team=team,
            player_configuration=PlayerConfiguration(poke_path.tag, None),
            opponent=load_player(
                tag=starting_opponent,
                league_path=poke_path.league_dir,
                battle_format=battle_format,
                team_size=team.team_size,
            ),
            *args,
            **kwargs,
        )
        self._tag = poke_path.tag
        self._reward_helper = reward_helper

        self.tag = poke_path.tag.split(" ")[0]

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
        print(__name__, asizeof.asizeof(self) / 1e9)
        self._logger.debug(asizeof.asizeof(self))

    def calc_reward(self, last_battle, current_battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle=current_battle,
            number_of_pokemons=self.team_size,
            **self._reward_helper.reward_values,
        )

    def embed_battle(self, battle: AbstractBattle) -> typing.Dict[str, npt.NDArray]:
        return self.preprocessor.embed_battle(battle)

    def describe_embedding(self) -> gym.spaces.Space:
        return self.preprocessor.describe_embedding()

    def step(
        self, action: ActionType
    ) -> typing.Tuple[ObservationType, float, bool, dict]:
        # self._logger.debug(f"Action: {action}")
        obs, reward, done, info = super().step(action=action)
        # self._logger.debug(f"Obs: {obs}")
        # self._logger.debug(f"Reward: {reward}")
        # self._logger.debug(f"Done: {done}")
        if done:
            info["win"] = {
                "opp": self._opp_tag,
                "result": self.current_battle.won,
            }
            self.update_win_rates()
            self.switch_opponent()
        info["team_size"] = self.team_size
        return obs, reward, done, info

    def reset(self, *args, **kwargs) -> typing.Dict[str, npt.NDArray]:
        # Reset the preprocessor
        self.preprocessor.reset()
        self.reset_battles()
        self._logger.debug(asizeof.asizeof(self))
        return super().reset(*args, **kwargs)

    def action_masks(self, *args, **kwargs) -> npt.NDArray:
        mask = action_masks(self.current_battle)
        # self._logger.debug(f"Mask: {mask}")
        return mask

    def action_to_move(self, action: int, battle: Battle) -> BattleOrder:
        action_mask = self.action_masks()
        if action_mask[action]:
            if action < NUM_MOVES:
                # self._logger.debug(
                #     f"Action {action} interpreted as a move "
                #     + f"({list(battle.active_pokemon.moves.keys())[action]}"
                # )
                return self.agent.create_order(
                    list(battle.active_pokemon.moves.values())[action]
                )
            else:
                # self._logger.debug(
                #     f"Action {action} interpreted as a switch "
                #     + f"({list(battle.team.values())[action - NUM_MOVES]}"
                # )
                return self.agent.create_order(
                    list(battle.team.values())[action - NUM_MOVES]
                )
        else:
            # self._logger.debug(f"Had to choose random action (given {action})")
            return self.agent.choose_random_move(battle)

    def set_team_size(self, team_size: int):
        self.team_size = team_size
        self.agent._team.set_team_size(team_size)
        self.matchmaker.set_team_size(team_size)
        self._opponent._team.set_team_size(team_size)
        self.win_rates = {}

    def update_win_rates(self):
        # Track the rolling win/loss rate against each opponent
        if self._opp_tag not in self.win_rates:
            self.win_rates[self._opp_tag] = collections.deque(maxlen=100)
        self.win_rates[self._opp_tag].append(int(self.current_battle.won))

    def switch_opponent(self):
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

    def reset_battles(self):
        self.agent._battles = {
            k: v for k, v in self.agent._battles.items() if not v.finished
        }
