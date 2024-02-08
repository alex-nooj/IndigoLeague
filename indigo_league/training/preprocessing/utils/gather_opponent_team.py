import typing

from poke_env.environment import AbstractBattle
from poke_env.environment import Pokemon
from poke_env.environment import Status


def gather_opponent_team(battle: AbstractBattle) -> typing.List[Pokemon]:
    opponent_team = [mon for mon in battle.opponent_team.values() if not mon.active]

    def species(mon: Pokemon) -> str:
        return mon.species

    for mon in battle._teampreview_opponent_team:
        if (
            mon._POKEDEX_DICT[mon.species]["num"]
            not in [m._POKEDEX_DICT[m.species]["num"] for m in opponent_team]
            and mon._POKEDEX_DICT[mon.species]["num"]
            != battle.opponent_active_pokemon._POKEDEX_DICT[battle.opponent_active_pokemon.species][
                "num"
            ]
        ):
            opponent_team.append(mon)

    sorted(opponent_team, key=species)
    return [battle.opponent_active_pokemon] + [mon for mon in opponent_team]


def gather_team(battle: AbstractBattle) -> typing.List[Pokemon]:
    return [battle.active_pokemon] + battle.available_switches
