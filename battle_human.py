import asyncio
import pathlib

import utils
from battling.environment.matchmaking.matchmaker import Matchmaker
from utils import PokePath


async def battle_human(
    player_name: str, battle_format: str, team_size: int, opponent_tag: str
):
    poke_path = PokePath(tag=player_name)
    matchmaker = Matchmaker(player_name, poke_path.league_dir, battle_format, team_size)
    opponent = matchmaker.load_player(opponent_tag)
    await opponent.send_challenges(player_name, 1)


if __name__ == "__main__":
    cfg_file = pathlib.Path(__file__).parent / "battle_human.yaml"
    asyncio.get_event_loop().run_until_complete(
        battle_human(**utils.load_config(cfg_file))
    )
