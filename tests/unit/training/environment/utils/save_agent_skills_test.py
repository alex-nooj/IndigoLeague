from unittest.mock import patch

import pytest
import trueskill

from indigo_league.training.environment.utils.save_agent_skills import save_agent_skills

# Mocking a trueskill.Rating object
mock_rating = trueskill.Rating(mu=25.0, sigma=8.333)

# Example agent skills data
agent_skills = {
    "Agent1": trueskill.Rating(mu=25.0, sigma=8.333),
    "Agent2": trueskill.Rating(mu=30.0, sigma=7.5),
    "SimpleHeuristics": trueskill.Rating(mu=20.0, sigma=6.666),
}


@pytest.fixture
def mock_league_path(tmp_path):
    # Create a temporary directory and files to simulate the league directory
    (tmp_path / "Agent1.txt").write_text("Agent 1 data")
    (tmp_path / "Agent2.txt").write_text("Agent 2 data")
    # Return the path to this temporary directory
    return tmp_path


def test_save_agent_skills(mock_league_path):
    # Patch 'OmegaConf.save' to prevent actual file writing and to track calls to it.
    with patch("omegaconf.OmegaConf.save", autospec=True) as mock_save:
        save_agent_skills(mock_league_path, agent_skills)

        # Ensure OmegaConf.save was called
        assert (
            mock_save.call_count == 1
        ), "OmegaConf.save was not called exactly once as expected."

        # If OmegaConf.save was called with keyword arguments, adjust retrieval method
        if mock_save.call_args[1]:  # Checking if there are keyword arguments
            saved_config = mock_save.call_args[1].get("config")
        else:
            saved_config = mock_save.call_args[0][0]  # The first positional argument

        # Assuming the function call was corrected and now 'saved_config' can be accessed,
        # proceed with the assertions to validate the saved configuration
        assert "Agent1" in saved_config, "Expected 'Agent1' in saved config."
        assert saved_config["Agent1"]["mu"] == 25.0, "Incorrect 'mu' for 'Agent1'."
        assert (
            saved_config["Agent1"]["sigma"] == 8.333
        ), "Incorrect 'sigma' for 'Agent1'."

        assert "Agent2" in saved_config, "Expected 'Agent2' in saved config."
        assert saved_config["Agent2"]["mu"] == 30.0
        assert saved_config["Agent2"]["sigma"] == 7.5

        assert (
            "SimpleHeuristics" in saved_config
        ), "Expected 'SimpleHeuristics' in saved config."
        assert saved_config["SimpleHeuristics"]["mu"] == 20.0
        assert saved_config["SimpleHeuristics"]["sigma"] == 6.666


def test_incorrect_simple_heuristics_handling(mock_league_path):
    # This test is to illustrate how you might test for incorrect handling
    # of the 'SimpleHeuristics' agent. Given the provided code directly assigns
    # the Rating object, this test is expected to fail without code correction.
    with patch("omegaconf.OmegaConf.save") as mock_save:
        save_agent_skills(mock_league_path, agent_skills)

        # Ensure OmegaConf.save was called
        assert (
            mock_save.call_count == 1
        ), "OmegaConf.save was not called exactly once as expected."

        # If OmegaConf.save was called with keyword arguments, adjust retrieval method
        if mock_save.call_args[1]:  # Checking if there are keyword arguments
            saved_config = mock_save.call_args[1].get("config")
        else:
            saved_config = mock_save.call_args[0][0]  # The first positional argument

        # This assertion will fail with the current implementation
        assert isinstance(saved_config["SimpleHeuristics"]["mu"], float)
        assert isinstance(saved_config["SimpleHeuristics"]["sigma"], float)
