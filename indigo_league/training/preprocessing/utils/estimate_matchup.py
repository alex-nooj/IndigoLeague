import typing

from poke_env.environment import Pokemon

from indigo_league.training.preprocessing.utils import stat_estimation
from indigo_league.utils.constants import NUM_POKEMON

POSSIBLE_ADVANTAGES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
ADV_MAP = {
    adv: round(2 * (ix / (len(POSSIBLE_ADVANTAGES) - 1)) - 1, 2)
    for ix, adv in enumerate(POSSIBLE_ADVANTAGES)
}


def determine_type_advantage(mon: Pokemon, opponent: Pokemon) -> float:
    return (
        max([opponent.damage_multiplier(t) for t in mon.types if t is not None]) / 4.0
    )


def determine_speed_tier(mon: Pokemon, opponent: Pokemon) -> float:
    mon_spe = stat_estimation(mon, "spe")
    opp_spe = stat_estimation(opponent, "spe")
    if mon_spe > opp_spe:
        return 1.0
    elif opp_spe > mon_spe:
        return -1.0

    return 0.0


def determine_remaining_mons(team: typing.Dict[str, Pokemon]) -> float:
    return float(len([m for m in team.values() if m.fainted is False])) / float(
        NUM_POKEMON
    )
