import asyncio
import pathlib

import utils

from indigo_league.training.environment.utils.load_player import load_player
from indigo_league.utils.directory_helper import PokePath


async def battle_human(
    player_name: str, battle_format: str, team_size: int, opponent_tag: str
):
    poke_path = PokePath(tag=player_name)
    opponent = load_player(
        tag=opponent_tag,
        league_path=poke_path.league_dir,
        battle_format=battle_format,
        team_size=team_size,
    )
    await opponent.send_challenges(player_name, 1)


if __name__ == "__main__":
    cfg_file = pathlib.Path(__file__).parent / "battle_human.yaml"
    asyncio.get_event_loop().run_until_complete(
        battle_human(**utils.load_config(cfg_file))
    )
