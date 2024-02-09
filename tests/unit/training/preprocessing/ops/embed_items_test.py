import typing

from indigo_league.training.preprocessing.ops import EmbedItems
from indigo_league.utils.smogon_data import SmogonData


def setup() -> typing.Tuple[EmbedItems, typing.List[str]]:
    data = SmogonData()
    items = {}
    for mon in data.smogon_data["data"].values():
        for item in mon["Items"]:
            items[item] = 1
    ability_list = ["none"] + sorted(list(items.keys())) + ["unknown_item"]
    return EmbedItems(8, 4), ability_list


def test_embed_abilities():
    # Arrange
    op, items = setup()

    # Act
    embedded_items = []
    for i in range(0, len(items), 6):
        embedded_items += op._embed_items(items[i : i + 6])

    # Assert
    for ix, embedded_item in enumerate(embedded_items[: len(items)]):
        assert embedded_item == ix


def test_less_than_six():
    # Arrange
    op, items = setup()

    # Trim off "none"
    items = items[1:]

    # Act
    for i in range(7):
        embedded_items = op._embed_items(items[:i])

        # Assert
        assert len(embedded_items) == 6
        for ix, item in enumerate(embedded_items[:i]):
            assert ix + 1 == item
        for item in embedded_items[i:]:
            assert item == 0
