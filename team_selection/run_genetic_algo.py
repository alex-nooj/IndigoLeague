import asyncio
import typing

import numpy as np
from poke_env.player import SimpleHeuristicsPlayer
import json
from team_selection.extensive_team_builder import ExtensiveTeamBuilder
from team_selection.extensive_team_builder import generate_all_pokemon
from team_selection.genetic_team_builder import GeneticTeamBuilder


class TeamChanger(SimpleHeuristicsPlayer):
    def change_team(self, team: ExtensiveTeamBuilder):
        self._team = team


async def evaluate_team(
    player1: TeamChanger, player2: TeamChanger, teams: typing.List[GeneticTeamBuilder], n_battles: int = 20
) -> typing.List[int]:
    team_scores = [0 for _ in teams]
    for ix_1, team_1 in enumerate(teams[:-1]):
        player1.change_team(team_1)
        for ix_2, team_2 in enumerate(teams[ix_1+1:]):
            player2.change_team(team_2)
            await player1.battle_against(player2, n_battles)
            team_scores[ix_1] += player1.n_won_battles
            team_scores[ix_2] += 20 - player1.n_won_battles
            player1.reset_battles()
            player2.reset_battles()
    return team_scores


async def main(population_size: int, n_mutations: int, battle_format: str, n_gens: int):
    import pathlib
    # Step 1: Generate N random teams
    teams = [GeneticTeamBuilder(mode=np.random.choice(["random", "sample", "teammate"])) for _ in range(population_size)]

    all_mons = generate_all_pokemon()
    mon_names = [m for m in all_mons]
    file1 = pathlib.Path(__file__).parent / "pokemon_list_1.json"
    file2 = pathlib.Path(__file__).parent / "pokemon_list_2.json"

    mons_1 = {k: all_mons[k] for k in mon_names[:len(mon_names)//2]}

    mons_2 = {k: all_mons[k] for k in mon_names[len(mon_names)//2:]}

    teambuilder_1 = ExtensiveTeamBuilder(all_mons, file1)
    teambuilder_2 = ExtensiveTeamBuilder(all_mons, file2)
    player1 = TeamChanger(
        battle_format=battle_format,
        max_concurrent_battles=10,
    )
    player2 = TeamChanger(
        battle_format=battle_format,
        max_concurrent_battles=10,
    )

    player1.change_team(teambuilder_1)
    player2.change_team(teambuilder_2)

    for _ in range(1000):
        await player1.battle_against(player2, 1)
    # for generation in range(n_gens):
    #     # Step 2: Have each team battle each other
    #     team_scores = await evaluate_team(player1, player2, teams)
    #
    #     # Step 3: Take the M highest-scoring teams and generate N-M new teams by changing between 1 and 3 pokemon on the
    #     # team
    #     teams = [x for _, x in sorted(zip(team_scores, teams), key=lambda pair: pair[0])]
    #     for team in teams[:n_mutations]:
    #         team.mutate(np.random.choice(teams[n_mutations:]).mons_dict)
    #
    #     highest_win_rate = sorted(team_scores)[-1]
    #     print(f"Generation {generation}: Highest score: {highest_win_rate:0.2f}")
    # print("=== Best Team ===")
    # print(teams[-1].team)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main(20, 10, "gen8ou", 5))
