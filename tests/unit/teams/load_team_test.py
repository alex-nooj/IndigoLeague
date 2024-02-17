import pathlib
from unittest.mock import MagicMock
from unittest.mock import mock_open

import pytest

from indigo_league.teams import load_team_from_file


# Test for non-existing file
def test_load_team_non_existing_file(monkeypatch):
    # Mock pathlib.Path.is_file to return False
    monkeypatch.setattr(pathlib.Path, "is_file", MagicMock(return_value=False))
    with pytest.raises(RuntimeError) as excinfo:
        load_team_from_file("non_existing_file.txt")
    assert "does not exist" in str(excinfo.value)


# Test for correctly parsing team members from a file
def test_load_team_correct_parsing(monkeypatch):
    # Mock file content
    team_content = "Member1\n\nMember2\n"
    monkeypatch.setattr("builtins.open", mock_open(read_data=team_content))
    # Assuming pathlib.Path.is_file would return True
    monkeypatch.setattr(pathlib.Path, "is_file", MagicMock(return_value=True))

    team = load_team_from_file("team.txt")
    assert team == ["Member1\n", "Member2\n"], "Team members were not parsed correctly"


# Test for an empty file
def test_load_team_empty_file(monkeypatch):
    # Mock an empty file
    monkeypatch.setattr("builtins.open", mock_open(read_data=""))
    monkeypatch.setattr(pathlib.Path, "is_file", MagicMock(return_value=True))

    team = load_team_from_file("empty_team.txt")
    assert team == [], "Function should return an empty list for an empty file"
