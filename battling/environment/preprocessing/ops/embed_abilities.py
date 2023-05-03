import pathlib
import typing

import gym
import numpy as np
import numpy.typing as npt
from poke_env.environment import AbstractBattle

from battling.environment.preprocessing.op import Op
from utils.gather_opponent_team import gather_opponent_team
from utils.gather_opponent_team import gather_team


class EmbedAbilities(Op):
    """Operation for converting abilities to an index."""

    def __init__(self, embedding_size: int):
        """Constructor for Op.

        Args:
            embedding_size: Size of the output of the embedding layer.  Rule of thumb is (size of codex)^(1/4).
        """
        pokemon_path = pathlib.Path(__file__).parent.parent.parent / "teams" / "gen8ou"
        self.abilities_lut = {}
        self._embedding_size = embedding_size
        counter = 0
        for mon in pokemon_path.iterdir():
            for moveset in mon.iterdir():
                with open(moveset, "r") as fp:
                    pokemon = fp.read()
                ability = (
                    pokemon.rsplit("Ability: ")[1]
                    .rsplit("\n")[0]
                    .lower()
                    .replace(" ", "")
                    .replace("-", "")
                )
                if ability not in self.abilities_lut:
                    self.abilities_lut[ability] = counter
                    counter += 1
        self.abilities_lut["None"] = len(self.abilities_lut)

    def embed_battle(
        self, battle: AbstractBattle, state: typing.Dict[str, npt.NDArray]
    ) -> typing.Dict[str, npt.NDArray]:
        """Convert the ability strings to an integer index value.

        Args:
            battle: Current state of the battle (functional)
            state: Current state of the battle (observation)

        Returns:
            Dict[str, NDArray]: The updated observational state.
        """
        # Gather all the abilities in the specific order of available switches.
        abilities = [mon.ability for mon in gather_team(battle)]

        # If we have fainted pokemon, we append the none ability.
        if len(abilities) < 6:
            abilities += ["None" for _ in range(6 - len(abilities))]

        # Gather the abilities of the opponent team.
        opp_abilities = [mon.ability for mon in gather_opponent_team(battle)]

        # If the opponent has fainted pokemon, we append the none ability.
        if len(opp_abilities) < 6:
            opp_abilities += ["None" for _ in range(6 - len(opp_abilities))]

        # Use the lookup table to convert abilities to indexes.
        ids = []
        for ability in abilities + opp_abilities:
            if ability in self.abilities_lut:
                ids.append(self.abilities_lut[ability])
            else:
                ids.append(self.abilities_lut["None"])
        state["ability_ids"] = np.asarray(ids)
        return state

    def describe_embedding(self) -> gym.spaces.Dict:
        """Describes the output of the observation space for this op.

        Returns:
            gym.spaces.Dict: Dictionary entry describing the observation space of this op alone.
        """
        return gym.spaces.Dict(
            {
                "ability_ids": gym.spaces.Box(
                    np.array([0 for _ in range(12)], dtype=np.int64),
                    np.array(
                        [len(self.abilities_lut) for _ in range(12)], dtype=np.int64
                    ),
                    dtype=np.int64,
                )
            }
        )

    def embedding_infos(self) -> typing.Dict[str, typing.Tuple[int, int, int]]:
        """Describes what the embedding layer should look like after this op.

        Returns:
            Dict[str, Tuple[int, int, int]]: The number of items in the codex, the embedding size, and the number of
                features
        """
        return {"ability_ids": (len(self.abilities_lut), self._embedding_size, 12)}
