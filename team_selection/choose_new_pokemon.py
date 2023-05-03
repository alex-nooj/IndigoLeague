import json
import numpy as np
import typing

from poke_env import GEN8_POKEDEX


def load_pokemon() -> typing.List[str]:
    valid_pokemon = []
    for species, pokemon in GEN8_POKEDEX.items():
        if "forme" in pokemon and ("mega" in pokemon["forme"].lower() or pokemon["forme"].lower() == "gmax"):
            continue
        elif "evos" in pokemon:
            continue
        valid_pokemon.append(species)
    return valid_pokemon


def generate_random_moveset(pokemon: str, learnset: typing.Dict[str, typing.Dict[str, typing.Any]]) -> typing.List[str]:
    move_pool = learnset[pokemon]["learnset"]
    return list(np.random.choice(move_pool.keys(), size=min([len(move_pool), 4])))


def choose_nature(pokemon: str) -> str:
    pass


if __name__ == "__main__":
    pokemon = load_pokemon()
    print()