import asyncio
import typing

import poke_env
import trueskill
from omegaconf import OmegaConf
from tabulate import tabulate

from utils import PokePath
from utils.load_player import load_player


def convert_username(username: str) -> str:
    return username.rsplit(" ")[0]


def battle_and_rate(
    results: typing.Dict[str, typing.Dict[str, typing.Optional[float]]],
    n_challenges: int,
) -> typing.Dict[str, trueskill.Rating]:
    trueskills = {convert_username(p): trueskill.Rating() for p in results.keys()}

    for p_1, stats in results.items():
        for p_2, win_rate in stats.items():
            if win_rate is None:
                continue
            for _ in range(int(n_challenges * win_rate)):
                (
                    trueskills[convert_username(p_1)],
                    trueskills[convert_username(p_2)],
                ) = trueskill.rate_1vs1(
                    trueskills[convert_username(p_1)], trueskills[convert_username(p_2)]
                )
    return trueskills


async def league_battle(battle_format: str, n_challenges: int):
    poke_path = PokePath()
    players = [
        load_player(agent.stem, poke_path.league_dir, battle_format, 6)
        for agent in poke_path.league_dir.iterdir()
        if agent.is_dir()
    ]
    players.append(
        load_player("SimpleHeuristics", poke_path.league_dir, battle_format, 6)
    )

    cross_evaluation = await poke_env.player.cross_evaluate(
        players, n_challenges=n_challenges
    )

    table = [["-"] + [convert_username(p.username) for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append(
            [convert_username(p_1)] + [cross_evaluation[p_1][p_2] for p_2 in results]
        )
    print(tabulate(table))

    trueskills = battle_and_rate(cross_evaluation, n_challenges)
    trueskill_table = []
    for k, v in trueskills.items():
        trueskill_table.append([k, v.mu, v.sigma])
    print(tabulate(trueskill_table, headers=["Name", "Mu", "Sigma"]))

    OmegaConf.save(
        config={k: {"mu": v.mu, "sigma": v.sigma} for k, v in trueskills.items()},
        f=(poke_path.league_dir / "trueskills.yaml"),
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(league_battle("gen8ou", 1000))
