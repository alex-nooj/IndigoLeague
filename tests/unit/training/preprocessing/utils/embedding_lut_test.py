from indigo_league.training.preprocessing.utils import EmbeddingLUT


class TestEmbeddingLUT:
    def test_constructor(self):
        keys = ["a", "b", "c", "d"]
        values = ["e", "f", "g", "h"]
        lut = EmbeddingLUT(keys=keys, values=values)
        assert lut.keys() == keys

        assert lut.values() == values

    def test_none_values(self):
        keys = ["a", "b", "c", "d"]
        values = None
        lut = EmbeddingLUT(keys=keys, values=values)
        assert lut.keys() == keys

        assert lut.values() == list(range(len(keys)))

    def test_get_item(self):
        keys = ["a", "b", "c", "d"]
        values = ["e", "f", "g", "h"]
        lut = EmbeddingLUT(keys=keys, values=values)
        for key, value in zip(keys, values):
            assert lut[key] == value

    def test_len(self):
        keys = ["a", "b", "c", "d"]
        values = ["e", "f", "g", "h"]
        lut = EmbeddingLUT(keys=keys, values=values)
        assert len(lut) == len(keys)
