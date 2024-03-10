import typing


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
