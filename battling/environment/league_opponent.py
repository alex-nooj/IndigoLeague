import collections
import pathlib

import numpy as np
import numpy.typing as npt
import poke_env
import torch
from poke_env.environment import AbstractBattle
from poke_env.environment import Effect
from poke_env.environment import Status
from poke_env.player import BattleOrder
from poke_env.teambuilder import Teambuilder
from sb3_contrib import MaskablePPO

from battling.environment.teams.team_builder import AgentTeamBuilder


class OpponentTeamBuilder(Teambuilder):
    def __init__(self, change_players: bool):
        self._team_queue = collections.deque()
        self.change_players = change_players

    def push_back(self, team: AgentTeamBuilder):
        self._team_queue.append(team)

    def yield_team(self) -> str:
        if len(self._team_queue) == 0:
            raise RuntimeError("Team queue is empty! Need to call 'push_back()' first!")

        team = self._team_queue[0].yield_team()
        if not self.change_players:
            self._team_queue.popleft()

        return team


class LeaguePolicy:
    def __init__(self, league_path: pathlib.Path, tag: str):
        agent_path = league_path / tag
        self.model = MaskablePPO.load(agent_path / "network.zip")
        peripherals = torch.load(agent_path / "team.pth")
        self.team = peripherals["team"]
        self.preprocessor = peripherals["preprocessor"]

    def choose_move(self, battle: AbstractBattle) -> int:
        obs = self.preprocessor.embed_battle(battle=battle)
        action, _ = self.model.predict(obs, action_masks=self.action_masks(battle))
        return int(action)

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


class LeagueOpponent(poke_env.player.SimpleHeuristicsPlayer):
    def __init__(
        self,
        starting_opponent: str,
        league_path: pathlib.Path,
        battle_format: str,
        team_size: int,
        change_players: bool,
        *args,
        **kwargs,
    ):
        self.league_path = league_path
        self.battle_format = battle_format
        self._team_size = team_size

        self.opponent_queue = collections.deque()
        self.tag_queue = collections.deque()
        self.team_builder = OpponentTeamBuilder(change_players)

        self.policy = None

        self.load_player(starting_opponent)
        self.next_player()
        super().__init__(
            battle_format=battle_format, team=self.team_builder, *args, **kwargs
        )

    def load_player(self, tag: str):
        self.tag_queue.append(tag)
        if tag == "SimpleHeuristics":
            self.opponent_queue.append(None)
            self.team_builder.push_back(
                AgentTeamBuilder(
                    battle_format=self.battle_format,
                    team_size=self._team_size,
                    randomize_team=True,
                )
            )
        else:
            player = LeaguePolicy(league_path=self.league_path, tag=tag)
            self.opponent_queue.append(player)
            self.team_builder.push_back(player.team)

    def next_player(self):
        print(f"Now playing against {self.tag_queue.popleft()}\n")
        self.policy = self.opponent_queue.popleft()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if self.policy is not None:
            return self.action_to_move(self.policy.choose_move(battle), battle)
        else:
            return super().choose_move(battle)

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

    def set_team_size(self, team_size: int):
        self._team_size = team_size
