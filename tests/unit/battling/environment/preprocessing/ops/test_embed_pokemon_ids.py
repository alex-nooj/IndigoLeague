import typing

from battling.environment.preprocessing.ops.embed_pokemon_ids import EmbedPokemonIDs
from utils.smogon_data import SmogonData


def setup() -> typing.Tuple[EmbedPokemonIDs, typing.List[str]]:
    data = SmogonData()
    mons = ["fainted"] + list(data.smogon_data["data"].keys())
    return EmbedPokemonIDs(8, 4), mons


def test_embed_mons():
    # Arrange
    op, mons = setup()

    # Act
    for i in range(0, len(mons), 6):
        _ = op._embed_pokemon_ids(mons[i:i+6])

    # Assert
    assert True


def test_less_than_six():
    # Arrange
    op, mons = setup()

    # Trim off "none"
    mons = mons[1:]

    # Act
    for i in range(7):
        embedded_mons = op._embed_pokemon_ids(mons[:i])

        # Assert
        assert len(embedded_mons) == 6
        for ix, mon in enumerate(embedded_mons[:i]):
            assert mon != 0
        for mon in embedded_mons[i:]:
            assert mon == 0
