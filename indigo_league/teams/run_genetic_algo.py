import numpy as np
from poke_env.player import cross_evaluate

from indigo_league.teams.genetic_team_builder import GeneticTeamBuilder
from indigo_league.teams.team_builder import AgentTeamBuilder
from indigo_league.utils.constants import NUM_POKEMON
from indigo_league.utils.fixed_heuristics_player import FixedHeuristicsPlayer


async def genetic_team_search(
    population_size: int, n_mutations: int, battle_format: str, n_gens: int
) -> AgentTeamBuilder:
    # Step 1: Generate N random teams
    teams = [
        GeneticTeamBuilder(mode=np.random.choice(["random", "sample", "teammate"]))
        for _ in range(population_size)
    ]

    for generation in range(n_gens):
        players = [
            FixedHeuristicsPlayer(
                battle_format=battle_format, max_concurrent_battles=10, team=team
            )
            for team in teams
        ]

        # Step 2: Have each team battle each other
        cross_evaluation = await cross_evaluate(players, n_challenges=10)

        team_scores = []
        for p_1, results in cross_evaluation.items():
            team_scores.append(
                sum([v for v in results.values() if v is not None])
                / (population_size - 1)
            )

        # Step 3: Take the M highest-scoring teams and generate N-M new teams by changing between 1 and 3 pokemon on the
        # team
        teams = [
            x for _, x in sorted(zip(team_scores, teams), key=lambda pair: pair[0])
        ]
        for team in teams[:n_mutations]:
            team.mutate(np.random.choice(teams[n_mutations:]).mons_dict, 2)

        highest_win_rate = sorted(team_scores)[-1]
        print(f"Generation {generation}: Highest score: {highest_win_rate:0.2f}")
    print("=== Best Team ===")
    print(teams[-1].team)
    team = AgentTeamBuilder(battle_format, NUM_POKEMON)
    team.set_team(list(teams[-1].mons_dict.values()))
    return team
