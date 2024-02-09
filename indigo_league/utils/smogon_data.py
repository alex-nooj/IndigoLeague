import json
import pathlib
import typing

import numpy as np

from indigo_league.utils.constants import NUM_MOVES
from indigo_league.utils.str_helpers import format_str


class SmogonData:
    def __init__(self):
        filepath = pathlib.Path(__file__)
        while not (filepath / "data" / "gen8ou-0.json").is_file():
            filepath = filepath.parent
        filepath = filepath / "data"
        self.smogon_data = {}

        with open(filepath / "gen8ou-0.json", "r") as f:
            data = json.load(f)
            self.smogon_data["info"] = data["info"]
            self.smogon_data["data"] = {
                k.lower(): v
                for k, v in data["data"].items()
                if v["Raw count"] / data["info"]["number of battles"] > 0.03
            }
        with open(filepath / "transfer_only_moves.json", "r") as f:
            transfer_only_moves = json.load(f)
            for mon, moves in transfer_only_moves.items():
                mon_name = mon.lower()
                if mon_name in self.smogon_data["data"]:
                    self.smogon_data["data"][mon_name]["Moves"] = {
                        m.lower(): v
                        for m, v in self.smogon_data["data"][mon_name]["Moves"].items()
                        if m not in moves
                    }

    def random_pokemon(self, size=1) -> typing.List[str]:
        return self._safe_sample_pokemon(
            {k: 1 for k in self.smogon_data["data"]}, size=size
        )

    def sample_pokemon(self, size: int = 1) -> typing.List[str]:
        return self._safe_sample_pokemon(
            {
                k: self.smogon_data["data"][k]["Raw count"]
                for k in self.smogon_data["data"]
            },
            size=size,
        )

    def sample_teammates(
        self,
        pokemon_name: str,
        size: int = 1,
        team: typing.Optional[typing.List[str]] = None,
    ) -> typing.List[str]:
        if team is None:
            team = []

        freq_dict = {}
        for mon, v in self.smogon_data["data"][pokemon_name.lower()][
            "Teammates"
        ].items():
            if mon.lower() not in self.smogon_data["data"]:
                continue
            for team_mon in team:
                if team_mon.lower().rsplit("-")[0] == mon.lower().rsplit("-")[0]:
                    break
            else:
                freq_dict[mon] = v
        return self._safe_sample_pokemon(
            freq_dict,
            size=size,
        )

    def build_pokemon(self, pokemon_name: str) -> str:
        if pokemon_name.lower() not in self.smogon_data["data"]:
            raise RuntimeError("Pokemon not found!")

        pokemon_data = self.smogon_data["data"][pokemon_name.lower()]

        # Pick item
        item = choose_from_dict(pokemon_data["Items"])[0]

        # Pick ability
        ability = choose_from_dict(pokemon_data["Abilities"])[0]

        # Nature/EV Spread
        evs_nature = choose_from_dict(pokemon_data["Spreads"])[0]
        nature = evs_nature.rsplit(":")[0]
        evs = evs_nature.rsplit(":")[-1].rsplit("/")

        # Pick moves
        moves = safe_sample_moves(pokemon_name.lower(), pokemon_data["Moves"])

        return create_pokemon_str(
            pokemon_name.lower(), item, ability, evs, nature, moves
        )

    def _safe_sample_pokemon(
        self, freq_dict: typing.Dict[str, float], size: int
    ) -> typing.List[str]:
        keys = [k for k in freq_dict]
        values = np.asarray([v for v in freq_dict.values()])
        selection = []
        for _ in range(size):
            selection.append(
                str(np.random.choice(keys, p=values / np.sum(values), size=1)[0])
            )

            # We have to be careful because of pokemon like Rotom and Rotom-Wash
            keys = [
                k
                for k in keys
                if selection[-1].rsplit("-")[0].lower() != k.rsplit("-")[0].lower()
            ]
            values = np.asarray([freq_dict[k] for k in keys])
        return selection


def safe_sample_moves(
    pokemon_name: str, moves: typing.Dict[str, float]
) -> typing.List[str]:
    moves = {k: v for k, v in moves.items() if k not in ["", "teleport", "zapcannon"]}

    if pokemon_name.lower() == "ditto":
        return ["transform"]
    else:
        return choose_from_dict(moves, NUM_MOVES)


def remove_incompatible_moves(
    moves: typing.Dict[str, float], incompatible_moves: typing.List[str]
) -> typing.List[str]:
    keys = [k for k in moves if len(k) != 0 and format_str(k) not in incompatible_moves]
    values = np.asarray([v for k, v in moves.items() if k in keys])
    return list(np.random.choice(keys, p=values / np.sum(values), size=NUM_MOVES))


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


def create_pokemon_str(
    name: str,
    item: str,
    ability: str,
    evs: typing.List[str],
    nature: str,
    moves: typing.List[str],
) -> str:
    mon_name = "-".join([s.capitalize() for s in name.rsplit("-")])
    if item == "nothing":
        pokemon_str = f"{mon_name}\nAbility: {ability}\n"
    else:
        pokemon_str = f"{mon_name} @ {item}\nAbility: {ability}\n"
    if any([k != "0" for k in evs]):
        ev_str = "EVs:"
        for ev, stat in zip(evs, ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]):
            if ev != "0":
                ev_str += f" {ev} {stat} /"
        # Trim the final backslash
        ev_str = ev_str[:-1] + "\n"
    else:
        ev_str = "EVs:"
        for stat in ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]:
            ev_str += f" 1 {stat} /"
        # Trim the final backslash
        ev_str = ev_str[:-1] + "\n"
    nature_str = f"{nature} Nature\n"
    move_str = ""
    for move in moves:
        move_str += f"- {move}\n"

    return pokemon_str + ev_str + nature_str + move_str
