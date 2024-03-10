import typing
from dataclasses import dataclass

from indigo_league.teams.utils.create_pokemon_str import create_pokemon_str


@dataclass
class PokeInfo:
    species: str
    item: str
    ability: str
    evs: typing.Dict[str, int]
    nature: str
    moves: typing.List[str]

    def __post_init__(self):
        total_evs = sum(self.evs.values())
        if total_evs > 510:
            raise ValueError("Total EVs cannot exceed 510.")

        for stat, value in self.evs.items():
            if value < 0 or value > 252:
                raise ValueError(
                    f"EVs for {stat} must be between 0 and 252 (got {value})"
                )

    def showdown_str(self) -> str:
        return create_pokemon_str(
            name=self.species.lower(),
            item=self.item.lower(),
            ability=self.ability.lower(),
            evs=[str(v) for v in self.evs.values() if v > 0],
            nature=self.nature.lower(),
            moves=self.moves,
        )
