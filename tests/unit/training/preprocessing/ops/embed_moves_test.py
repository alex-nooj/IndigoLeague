import typing

from indigo_league.training.preprocessing.ops import EmbedMoves
from indigo_league.teams.smogon_data import SmogonData


def setup() -> typing.Tuple[EmbedMoves, typing.List[str]]:
    data = SmogonData()
    moves = {}
    for v in data.smogon_data["data"].values():
        for move in v["Moves"]:
            moves[move] = 1
    move_list = ["null"] + sorted(list(moves.keys())) + ["struggle"]
    return EmbedMoves(8, 4), move_list


def test_embed_moves():
    # Arrange
    op, moves = setup()

    # Act
    embedded_moves = []
    for i in range(0, len(moves), 4):
        embedded_moves += op._embed_moves(moves[i : i + 4])

    # Assert
    for ix, embedded_moves in enumerate(embedded_moves[: len(moves)]):
        assert embedded_moves == ix


def test_less_than_six():
    # Arrange
    op, moves = setup()

    # Trim off "none"
    moves = moves[1:]

    # Act
    for i in range(5):
        embedded_moves = op._embed_moves(moves[:i])

        # Assert
        assert len(embedded_moves) == 4
        for ix, move in enumerate(embedded_moves[:i]):
            assert ix + 1 == move
        for move in embedded_moves[i:]:
            assert move == 0
