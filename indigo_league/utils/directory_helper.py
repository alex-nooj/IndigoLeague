import pathlib
import typing

import indigo_league.utils.trainers


class PokePath:
    def __init__(self, tag: typing.Optional[str] = None):
        league_path = pathlib.Path(__file__)
        while league_path.stem != "IndigoLeague":
            league_path = league_path.parent
        self._league_path = league_path.parent / "pokemon_league"
        if tag is None:
            self._tag = self._get_tag()
        else:
            self._tag = tag

        # Make all the various paths
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.league_dir.mkdir(parents=True, exist_ok=True)
        self.challenger_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_dir(self) -> pathlib.Path:
        return self._league_path

    @property
    def challenger_dir(self) -> pathlib.Path:
        return self._league_path / "challengers"

    @property
    def agent_dir(self) -> pathlib.Path:
        (self.challenger_dir / self._tag).mkdir(parents=True, exist_ok=True)
        return self.challenger_dir / self._tag

    @property
    def league_dir(self) -> pathlib.Path:
        return self._league_path / "league"

    @property
    def tag(self) -> str:
        return self._tag

    def _get_tag(self) -> str:
        for tag in indigo_league.utils.trainers.TRAINERS:
            if not (self._league_path / "challengers" / tag).is_dir():
                return tag
        raise RuntimeError("Out of trainer names!")
