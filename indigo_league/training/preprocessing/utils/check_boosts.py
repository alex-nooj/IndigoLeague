from poke_env.environment import Pokemon


def check_defense(active: Pokemon) -> float:
    return 1.0 if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3 else -1.0


def check_attack(active: Pokemon) -> float:
    if active.boosts["atk"] <= -3 and active.stats["atk"] >= active.stats["spa"]:
        return 1.0
    elif active.boosts["spa"] <= -3 and active.stats["atk"] <= active.stats["spa"]:
        return 1.0

    return -1.0
