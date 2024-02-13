import typing

import numpy as np


def choose_from_dict(
    freq_dict: typing.Dict[str, float], size: int = 1
) -> typing.List[str]:
    keys = []
    values = []
    for k, v in freq_dict.items():
        if len(k) != 0:
            keys.append(k)
            values.append(v)

    values = np.asarray(values)
    return list(
        np.random.choice(keys, size=size, replace=False, p=values / np.sum(values))
    )
