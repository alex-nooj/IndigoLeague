import pathlib
import typing

import trueskill
from omegaconf import OmegaConf


def save_agent_skills(
    league_path: pathlib.Path, agent_skills: typing.Dict[str, trueskill.Rating]
):
    dict_skills = {}

    for agent in league_path.iterdir():
        if agent.stem in agent_skills:
            dict_skills[agent.stem] = {
                "mu": agent_skills[agent.stem].mu,
                "sigma": agent_skills[agent.stem].sigma,
            }

    dict_skills["FixedHeuristics"] = {
        "mu": agent_skills["FixedHeuristics"].mu,
        "sigma": agent_skills["FixedHeuristics"].sigma,
    }
    OmegaConf.save(config=dict_skills, f=(league_path / "trueskills.yaml"))
