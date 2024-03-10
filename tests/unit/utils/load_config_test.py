from unittest.mock import mock_open
from unittest.mock import patch

import omegaconf
import pytest

from indigo_league.utils import load_config

# Sample configurations for testing
default_config = """
app:
  name: MyApp
  version: 1.0
"""

cli_config = """
app:
  version: 2.0
other:
  enabled: true
"""

file_config = """
app:
  name: UpdatedApp
other:
  feature: new
"""


# Test loading configuration from a file
def test_load_config_from_file():
    with patch("builtins.open", mock_open(read_data=default_config)), patch(
        "pathlib.Path.is_file", return_value=True
    ), patch(
        "omegaconf.OmegaConf.load",
        return_value=omegaconf.OmegaConf.create(default_config),
    ):
        config = load_config("path/to/default/config.yaml", False)
        assert config["app"]["name"] == "MyApp"
        assert config["app"]["version"] == 1.0


# Test loading configuration with CLI arguments
def test_load_config_with_cli():
    with patch(
        "omegaconf.OmegaConf.from_cli",
        return_value=omegaconf.OmegaConf.create(cli_config),
    ), patch("builtins.open", mock_open(read_data=default_config)), patch(
        "pathlib.Path.is_file", return_value=True
    ), patch(
        "omegaconf.OmegaConf.load",
        side_effect=[
            omegaconf.OmegaConf.create(default_config),
            omegaconf.OmegaConf.create(file_config),
        ],
    ):
        config = load_config("path/to/default/config.yaml", True)
        assert config["app"]["version"] == 2.0
        assert config["other"]["enabled"]


# Test handling of non-existent CLI-specified configuration file
def test_nonexistent_cli_config_file():
    with patch(
        "omegaconf.OmegaConf.from_cli",
        return_value=omegaconf.OmegaConf.create({"config": "nonexistent.yaml"}),
    ), patch("pathlib.Path.is_file", return_value=False), patch(
        "builtins.open", mock_open(read_data=default_config)
    ), patch(
        "omegaconf.OmegaConf.load",
        return_value=omegaconf.OmegaConf.create(default_config),
    ), pytest.warns(
        UserWarning
    ):
        config = load_config("path/to/default/config.yaml", True)
        assert config["app"]["name"] == "MyApp"  # Falls back to default config
