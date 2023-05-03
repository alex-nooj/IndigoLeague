import asyncio
import pathlib
import typing

import numpy as np
from poke_env import teambuilder
from poke_env.player import SimpleHeuristicsPlayer
from tqdm import tqdm

from battling.environment.teams.team_builder import AgentTeamBuilder


class TeamChanger(SimpleHeuristicsPlayer):
    def change_team(self, team: typing.Union[str, AgentTeamBuilder]):
        if isinstance(team, str):
            self._team = teambuilder.ConstantTeambuilder(team)
        else:
            self._team = team


class RandomTeamBuilder(teambuilder.Teambuilder):
    def __init__(self, choices: typing.List[pathlib.Path]):
        self.choices = choices

    def yield_team(self) -> str:
        return self.join_team(
            self.parse_showdown_team(
                list_of_path_to_team(generate_random_team(self.choices))
            )
        )


def choose_new_pokemon(
    team: typing.List[pathlib.Path], choices: typing.List[pathlib.Path]
) -> pathlib.Path:
    mon_species = [mon.parent.stem for mon in team]
    valid_choices = [mon for mon in choices if mon.parent.stem not in mon_species]
    return np.random.choice(valid_choices)


def generate_random_team(choices: typing.List[pathlib.Path]) -> typing.List[pathlib.Path]:
    team = []
    for _ in range(6):
        team.append(choose_new_pokemon(team, choices))
    return team


def load_choices() -> typing.List[pathlib.Path]:
    team_dir = (
        pathlib.Path(__file__).parent.parent / "battling" / "environment" / "teams" / "gen8ou"
    )
    choices = []
    for mon in team_dir.iterdir():
        choices += list(mon.iterdir())
    return choices


def mutate_team(
    team: typing.List[pathlib.Path], choices: typing.List[pathlib.Path]
) -> typing.List[pathlib.Path]:
    n_mutations = int(np.random.randint(low=1, high=4, size=1))
    new_team = list(np.random.choice(team, size=6 - n_mutations, replace=False))
    for _ in range(n_mutations):
        new_team.append(choose_new_pokemon(new_team, choices))
    return new_team


def list_of_path_to_team(team: typing.List[pathlib.Path]) -> str:
    str_team = ""
    for mon in team:
        with open(mon, "r") as fp:
            str_team += fp.read()
        str_team += "\n\n"
    return str_team


async def evaluate_team(
    player1: TeamChanger, player2: TeamChanger, teams: typing.List[typing.List[pathlib.Path]]
) -> typing.List[int]:
    team_scores = [0 for _ in teams]
    for i, team_1 in enumerate(tqdm(teams)):
        player1.change_team(list_of_path_to_team(team_1))
        await player1.battle_against(player2, 100)
        team_scores[i] += player1.n_won_battles / 100
        player1.reset_battles()
        player2.reset_battles()
    return team_scores


async def main(population_size: int, battle_format: str, desired_win_rate: float):
    # Step 1: Generate N random teams
    choices = load_choices()
    teams = [generate_random_team(choices) for _ in range(population_size)]

    highest_win_rate = 0
    generation = 0
    best_teams = []
    player1 = TeamChanger(
        battle_format=battle_format,
        max_concurrent_battles=100,
    )
    player2 = TeamChanger(
        battle_format=battle_format,
        max_concurrent_battles=100,
        team=RandomTeamBuilder(choices),
    )
    while generation < desired_win_rate:
        # Step 2: Have those teams compete against each other
        player_scores = await evaluate_team(player1, player2, teams)

        # Step 3: Take the M highest-scoring teams and generate N-M new teams by changing between 1 and 3 pokemon on the
        # team
        best_teams = [x for _, x in sorted(zip(player_scores, teams), key=lambda pair: pair[0])][
            -population_size // 10 :
        ]
        highest_win_rate = sorted(player_scores)[-1]
        print(f"Generation {generation}: Highest score: {highest_win_rate:0.2f}")
        new_teams = []
        while (len(best_teams) + len(new_teams)) < population_size:
            for team in best_teams:
                new_teams.append(mutate_team(team, choices))
                if len(best_teams) + len(new_teams) >= population_size:
                    break
        teams = best_teams + new_teams
        generation += 1
    print("=== Best Team ===")
    print(list_of_path_to_team(best_teams[-1]))


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main(100, "gen8ou", 10))
