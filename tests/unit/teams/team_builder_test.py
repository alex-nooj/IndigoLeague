import pytest

from indigo_league.teams import AgentTeamBuilder
from indigo_league.teams.team_builder import generate_random_team
from indigo_league.utils.constants import NUM_POKEMON


class TestGenerateRandomTeam:
    @pytest.mark.parametrize("team_size", list(range(1, NUM_POKEMON + 1)))
    def test_team_length(self, team_size: int):
        assert len(generate_random_team(team_size)) == team_size

    def test_unique_mons(self):
        team = generate_random_team(NUM_POKEMON)
        mons = [s.split("\n")[0] for s in team]
        assert len(set(mons)) == len(mons)


class TestTeamBuilder:
    def test_set_team(self):
        team = ["A", "B", "C", "D"]
        team_builder = AgentTeamBuilder(
            battle_format="gen8ou", team_size=NUM_POKEMON, randomize_team=False
        )
        team_builder.set_team(team)
        assert all(i == j for i, j in zip(team_builder._team, team))
        assert all(i == j for i, j in zip(team_builder.team, team))

    @pytest.mark.parametrize("team_size", list(range(1, NUM_POKEMON + 1)))
    def test_team_size(self, team_size: int):
        team_builder = AgentTeamBuilder(
            battle_format="gen8ou", team_size=team_size, randomize_team=False
        )
        assert team_builder.team_size == team_size

    @pytest.mark.parametrize("team_size", list(range(1, NUM_POKEMON + 1)))
    def test_set_team_size(self, team_size: int):
        team_builder = AgentTeamBuilder(
            battle_format="gen8ou", team_size=NUM_POKEMON, randomize_team=False
        )
        team_builder.set_team_size(team_size=team_size)
        assert team_builder.team_size == team_size

    def test_randomize_team(self):
        team_builder = AgentTeamBuilder(
            battle_format="gen8ou", team_size=NUM_POKEMON, randomize_team=True
        )
        assert team_builder.team is None
        assert team_builder.yield_team() != team_builder.yield_team()

    def test_save_team(self, tmp_path):
        team_builder = AgentTeamBuilder(
            battle_format="gen8ou", team_size=NUM_POKEMON, randomize_team=False
        )

        team_builder.save_team(tmp_path)

        assert (tmp_path / "team.txt").is_file()

        with open(tmp_path / "team.txt", "r") as f:
            text = f.read()

        assert "".join(team_builder.team) == text

    def test_save_team_error(self, tmp_path):
        team_builder = AgentTeamBuilder(
            battle_format="gen8ou", team_size=NUM_POKEMON, randomize_team=True
        )

        try:
            team_builder.save_team(tmp_path)
        except RuntimeError:
            assert True
            return
        assert False
