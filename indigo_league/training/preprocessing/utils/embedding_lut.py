import typing

from indigo_league.utils.str_helpers import format_str


class EmbeddingLUT:
    def __init__(
        self,
        keys: typing.Union[typing.List[str], typing.List[int]],
        values: typing.Optional[typing.Any] = None,
    ):
        self.lut = {}
        counter = 0
        for ix, k in enumerate(keys):
            if isinstance(k, int):
                key = str(k)
            else:
                key = format_str(k)

            if values:
                self.lut[key] = values[ix]
            elif key not in self.lut:
                self.lut[key] = counter
                counter += 1

    def __getitem__(self, key: typing.Union[str, int]) -> int:
        if isinstance(key, int):
            return self.lut[str(key)]
        else:
            return self.lut[format_str(key)]

    def __len__(self) -> int:
        return len(self.lut)

    def keys(self) -> typing.List[str]:
        return list(self.lut.keys())

    def values(self) -> typing.List[int]:
        return list(self.lut.values())
