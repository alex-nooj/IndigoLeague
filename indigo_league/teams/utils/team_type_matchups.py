import typing

from poke_env.environment import Pokemon
from poke_env.environment import PokemonType


def defense_matchups(team: typing.List[str]) -> typing.Dict[PokemonType, int]:
    """Checks the type match-ups when a pokemon attacks a pokemon on this team

    If a move used against a pokemon would be considered super-effective (x2 or
    x4) then we increase the count by 1. If a type is considered not very
    effective or would have no effect, then we decrease the count by 1.

    Args:
        team: List of pokemon species on the team

    Returns:
        Dict mapping types to the number of pokemon it's super-effective against
    """
    type_matchup = {t: 0 for t in PokemonType}

    for species in team:
        mon = Pokemon(species=species)
        for attack_type in PokemonType:
            effectiveness = mon.damage_multiplier(attack_type)
            if effectiveness > 1:
                type_matchup[attack_type] += 1
            elif effectiveness < 1:
                type_matchup[attack_type] -= 1
    return type_matchup
