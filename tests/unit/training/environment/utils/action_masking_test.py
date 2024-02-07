import numpy as np
from poke_env.environment import Move
from poke_env.environment import Pokemon

from indigo_league.training.environment.utils.action_masking import moves_mask
from indigo_league.utils.constants import NUM_MOVES


class TestMovesMask:
    def test_no_active_mon(self):
        # Arrange
        active_pokemon = None
        available_moves = [Move("flamethrower")]
        expected = np.zeros(NUM_MOVES)

        # Act
        mask = moves_mask(active_pokemon, available_moves)

        # Assert
        assert np.prod(np.equal(mask, expected))

    def test_no_pp(self):
        # Arrange
        active_pokemon = Pokemon(species="Charizard")

        move1 = Move("flamethrower")
        move1._current_pp = 0
        move2 = Move("aerialace")
        move2._current_pp = 5
        move3 = Move("blastburn")
        move3._current_pp = 0
        move4 = Move("fly")
        move4._current_pp = 5

        available_moves = [move1, move2, move3, move4]
        expected = np.asarray([0.0, 1.0, 0.0, 1.0])

        active_pokemon._moves = {m.id: m for m in available_moves}

        # Act
        mask = moves_mask(active_pokemon, available_moves)

        # Assert
        assert np.prod(np.equal(mask, expected))


def test_switch_mask():
    pass


def test_action_masks():
    pass
