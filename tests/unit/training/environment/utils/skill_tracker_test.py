from unittest.mock import patch

import pytest
import trueskill

from indigo_league.training.environment.utils.skill_tracker import SkillTracker


# Fixture to simulate loading existing skills from a YAML file
@pytest.fixture
def mock_skill_data():
    return {
        "Agent1": {"mu": 25.0, "sigma": 8.333},
        "Agent2": {"mu": 30.0, "sigma": 7.5},
    }


# Fixture for a temporary league path
@pytest.fixture
def league_path(tmp_path):
    return tmp_path


# Test initialization with existing skill data
def test_init_with_existing_skills(league_path, mock_skill_data):
    config_path = league_path / "trueskills.yaml"
    with patch("pathlib.Path.is_file", return_value=True), patch(
        "omegaconf.OmegaConf.load", return_value=mock_skill_data
    ):
        tracker = SkillTracker("Agent1", league_path)
        assert "Agent1" in tracker.agent_skills
        assert tracker.agent_skills["Agent1"].mu == 25.0


# Test initialization without existing skill data
def test_init_without_existing_skills(league_path):
    with patch("pathlib.Path.is_file", return_value=False):
        tracker = SkillTracker("NewAgent", league_path)
        assert "NewAgent" in tracker.agent_skills
        assert (
            tracker.agent_skills["NewAgent"].mu == trueskill.Rating().mu
        )  # Default mu


# Test the update method
def test_update(league_path):
    with patch("pathlib.Path.is_file", return_value=False), patch(
        "trueskill.rate_1vs1",
        return_value=(
            trueskill.Rating(mu=26.0, sigma=7.8),
            trueskill.Rating(mu=24.0, sigma=8.2),
        ),
    ):
        tracker = SkillTracker("Agent1", league_path)
        tracker.update("Agent2", battle_won=True)
        assert tracker.agent_skills["Agent1"].mu == pytest.approx(26.0)
        assert tracker.agent_skills["Agent2"].mu == pytest.approx(24.0)


# Test the agent_skills property
def test_agent_skills_property(league_path):
    with patch("pathlib.Path.is_file", return_value=False):
        tracker = SkillTracker("Agent1", league_path)
        assert isinstance(tracker.agent_skills, dict)
        assert "Agent1" in tracker.agent_skills
