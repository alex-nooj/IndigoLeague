from poke_env import PlayerConfiguration
from poke_env.environment import Pokemon
from poke_env.player import SimpleHeuristicsPlayer

from indigo_league.utils.fixed_heuristics_player import FixedHeuristicsPlayer


def test_none_mon():
    agent = FixedHeuristicsPlayer(
        PlayerConfiguration(username="Test Player", password="test")
    )
    for stat in ["atk", "def", "spa", "spd", "spe"]:
        assert agent._stat_estimation(None, stat) == 1.0


def test_stat_estimation():
    agent = FixedHeuristicsPlayer(
        PlayerConfiguration(username="Test Player 1", password="test")
    )
    agent2 = SimpleHeuristicsPlayer(
        PlayerConfiguration(username="Test Player 2", password="test")
    )
    mon = Pokemon(species="charizard")
    for stat in ["atk", "def", "spa", "spd", "spe"]:
        assert agent._stat_estimation(mon, stat) == agent2._stat_estimation(mon, stat)
