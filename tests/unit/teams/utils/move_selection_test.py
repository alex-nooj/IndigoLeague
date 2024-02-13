import pytest
from poke_env.environment import Move
from poke_env.environment import MoveCategory

from indigo_league.teams.utils import move_selection


@pytest.mark.parametrize("category", [MoveCategory.STATUS, MoveCategory.PHYSICAL, MoveCategory.SPECIAL])
def test_remove_move_category(category: MoveCategory):
    moves = {
        "flamethrower": 0,
        "machpunch": 1,
        "lightscreen": 2,
    }

    results = move_selection.remove_move_category(moves, category)

    for k, v in results.items():
        assert Move(k).category != category
        assert v == moves[k]
