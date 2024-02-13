import typing

from indigo_league.training.preprocessing.ops.embed_abilities import EmbedAbilities
from indigo_league.utils.smogon_data import SmogonData


def setup() -> typing.Tuple[EmbedAbilities, typing.List[str]]:
    data = SmogonData()
    abilities = {}
    for mon in data.smogon_data["data"].values():
        for ability in mon["Abilities"]:
            abilities[ability] = 1
    ability_list = ["none"] + sorted(list(abilities.keys()))
    return EmbedAbilities(8, 4), ability_list


def test_embed_abilities():
    # Arrange
    op, abilities = setup()

    # Act
    embedded_abilities = []
    for i in range(0, len(abilities), 6):
        embedded_abilities += op._embed_abilities(abilities[i : i + 6])

    # Assert
    for ix, embedded_abilities in enumerate(embedded_abilities[: len(abilities)]):
        assert embedded_abilities == ix


def test_less_than_six():
    # Arrange
    op, abilities = setup()

    # Trim off "none"
    abilities = abilities[1:]

    # Act
    for i in range(7):
        embedded_abilities = op._embed_abilities(abilities[:i])

        # Assert
        assert len(embedded_abilities) == 6
        for ix, ability in enumerate(embedded_abilities[:i]):
            assert ix + 1 == ability
        for ability in embedded_abilities[i:]:
            assert ability == 0
