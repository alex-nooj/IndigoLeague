import typing

import pytest
from poke_env.environment import Pokemon

from indigo_league.training.preprocessing.ops import heuristics_op


class TestEmbedMon:
    @pytest.mark.parametrize(
        "mon,opponent,expected",
        [
            (None, Pokemon(species="charizard"), -1.0),
            (Pokemon(species="charizard"), None, -1.0),
            (None, None, -1.0),
            (Pokemon(species="jolteon"), Pokemon(species="golem"), 0.0),
            (Pokemon(species="chikorita"), Pokemon(species="charizard"), (0.25 / 4.0)),
            (Pokemon(species="chikorita"), Pokemon(species="metagross"), (0.5 / 4.0)),
            (Pokemon(species="chikorita"), Pokemon(species="slaking"), (1.0 / 4.0)),
            (Pokemon(species="chikorita"), Pokemon(species="totodile"), (2.0 / 4.0)),
            (Pokemon(species="chikorita"), Pokemon(species="quagsire"), (4.0 / 4.0)),
        ],
    )
    def test_mons_type_advantage(
        self,
        mon: typing.Optional[Pokemon],
        opponent: typing.Optional[Pokemon],
        expected: float,
    ):
        result = heuristics_op.embed_mon(mon, opponent)

        assert result[0] == expected

    @pytest.mark.parametrize(
        "mon,opponent,expected",
        [
            (None, Pokemon(species="charizard"), -1.0),
            (Pokemon(species="charizard"), None, -1.0),
            (None, None, -1.0),
            (Pokemon(species="jolteon"), Pokemon(species="golem"), (0.0 / 4.0)),
            (Pokemon(species="chikorita"), Pokemon(species="charizard"), (0.25 / 4.0)),
            (Pokemon(species="chikorita"), Pokemon(species="metagross"), (0.5 / 4.0)),
            (Pokemon(species="chikorita"), Pokemon(species="slaking"), (1.0 / 4.0)),
            (Pokemon(species="chikorita"), Pokemon(species="totodile"), (2.0 / 4.0)),
            (Pokemon(species="chikorita"), Pokemon(species="quagsire"), (4.0 / 4.0)),
        ],
    )
    def test_mons_type_disadvantage(
        self,
        mon: typing.Optional[Pokemon],
        opponent: typing.Optional[Pokemon],
        expected: float,
    ):
        result = heuristics_op.embed_mon(opponent, mon)

        assert result[1] == expected
