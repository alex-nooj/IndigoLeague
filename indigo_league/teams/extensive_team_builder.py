import json
import pathlib
import typing

import tqdm
from poke_env import teambuilder

from indigo_league.utils.constants import NUM_POKEMON
from indigo_league.utils.smogon_data import create_pokemon_str
from indigo_league.utils.smogon_data import SmogonData
from indigo_league.utils.str_helpers import format_str


def generate_all_pokemon() -> typing.Dict[str, typing.List[str]]:
    data = SmogonData()
    pokemon = {}
    filename = (
        pathlib.Path(__file__).parent.parent / "utils/data/transfer_only_moves.json"
    )
    with open(filename, "r") as f:
        transfer_only_moves = json.load(f)

    mons = [m for m in data.smogon_data["data"]]
    mons.sort()
    for mon in tqdm.tqdm(mons):
        if mon.lower() != "slowbro":
            continue
        pokemon[mon] = []
        moves = [
            m
            for m in data.smogon_data["data"][mon]["Moves"]
            if m not in ["", "zapcannon"]
        ]
        if mon.lower() in transfer_only_moves:
            moves = [
                m
                for m in moves
                if format_str(m) not in transfer_only_moves[format_str(mon)]
            ]
        for i in range(0, len(moves) - 3):
            for j in range(i + 1, len(moves) - 2):
                for k in range(j + 1, len(moves) - 1):
                    for l in range(k + 1, len(moves)):
                        move_set = [moves[i], moves[j], moves[k], moves[l]]
                        for ability in data.smogon_data["data"][mon]["Abilities"]:
                            item = list(data.smogon_data["data"][mon]["Items"].keys())[
                                0
                            ]
                            spread = list(
                                data.smogon_data["data"][mon]["Spreads"].keys()
                            )[0]
                            nature = spread.rsplit(":")[0]
                            evs = spread.rsplit(":")[-1].rsplit("/")
                            pokemon[mon].append(
                                create_pokemon_str(
                                    mon, item, ability, evs, nature, move_set
                                )
                            )
    return pokemon


class ExtensiveTeamBuilder(teambuilder.Teambuilder):
    def __init__(
        self, mons_list: typing.Dict[str, typing.List[str]], filename: pathlib.Path
    ):
        self.mons = mons_list
        self.filename = filename

    def yield_team(self) -> str:
        mons = [mon for mon in self.mons]
        team = ""
        team_mons = []
        for mon in mons:
            if mon.rsplit("-")[0] in team_mons:
                continue
            else:
                team_mons.append(mon.rsplit("-")[0])
            team += self.mons[mon].pop()
            team += "\n"
            if len(self.mons[mon]) == 0:
                del self.mons[mon]
            if len(team_mons) == NUM_POKEMON:
                break
        print(team)
        return self.join_team(self.parse_showdown_team(team))
