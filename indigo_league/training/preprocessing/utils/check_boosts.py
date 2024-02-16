from poke_env.environment import Pokemon


def check_defense(active: Pokemon) -> float:
    """Determines if defense stats have been lowered too much

    Args:
        active: The current active pokemon

    Returns:
        -1.0 if def or spd has a -3 boost or lower, 1.0 otherwise
    """
    return -1.0 if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3 else 1.0


def check_attack(active: Pokemon) -> float:
    """Determines if attack stats have been lowered too much

    Args:
        active: The current active pokemon

    Returns:
        -1.0 if the pokemon's best attack stat has a -3.0 boost or lower, 1.0 otherwise
    """
    if active.boosts["atk"] <= -3 and active.stats["atk"] >= active.stats["spa"]:
        return -1.0
    elif active.boosts["spa"] <= -3 and active.stats["atk"] <= active.stats["spa"]:
        return -1.0

    return 1.0
