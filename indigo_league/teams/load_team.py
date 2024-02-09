import pathlib
import typing


def load_team_from_file(filename: str) -> typing.List[str]:
    if not pathlib.Path(filename).is_file():
        raise RuntimeError(f"Team file {filename} does not exist!")

    with open(filename, "r") as fp:
        contents = fp.read()

    return [
        p + "\n" if p[-1] != "\n" else p for p in contents.split("\n\n") if p != "\n"
    ]
