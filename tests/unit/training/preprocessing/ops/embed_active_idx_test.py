from unittest.mock import MagicMock

import pytest
from poke_env.environment import Battle

from indigo_league.training.preprocessing.ops import EmbedActiveIdx


class TestEmbedActiveIdx:
    @pytest.mark.parametrize("seq_len", [1, 5])
    def test_construction(self, seq_len: int):
        op = EmbedActiveIdx(seq_len=seq_len)
        assert op.describe_embedding()[op.key].shape[0] == op.n_features * seq_len

    def test_embed_battle(self):
        logger = MagicMock()
        op = EmbedActiveIdx(seq_len=1)

        battle = Battle("tag", "username", logger)
        battle.get_pokemon("p1: azumarill", force_self_team=True)
        battle.get_pokemon("p1: blastoise", force_self_team=True)
        battle.get_pokemon("p1: carnivine", force_self_team=True)
        battle.get_pokemon("p1: diancie", force_self_team=True)
        battle.get_pokemon("p1: marill", force_self_team=True)
        battle.get_pokemon("p1: zapdos", force_self_team=True)
        battle.get_pokemon("p1: zapdos")._active = True
        idx = op._embed_battle(battle, {})
        assert all(i == j for i, j in zip(idx, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
