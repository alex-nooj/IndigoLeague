import typing

import torch
from torch import nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PokemonFeatureExtractor(BaseFeaturesExtractor):
    """Feature Extractor for the Pokemon AI."""
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        embedding_infos: typing.Dict[str, typing.Tuple[int, int, int]],
        n_linear_layers: int,
        shared: typing.List[int],
    ):
        """Constructor function.

        Args:
            observation_space: Shape and type of the observation space of the environment.
            embedding_infos: Describes which items in the observation space need an embedding layer.
            n_linear_layers: Number of linear layers to append after the embedding.
            shared: Number of hiddens in each of the linear layers after concatenation.
        """
        super().__init__(observation_space, features_dim=shared[-1])
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.items():
            if key in embedding_infos:
                layers = [
                    nn.Embedding(embedding_infos[key][0], embedding_infos[key][1])
                ]
                for _ in range(n_linear_layers):
                    layers.append(
                        nn.Linear(
                            embedding_infos[key][1],
                            embedding_infos[key][1],
                        )
                    )
                    layers.append(nn.ReLU())
                extractors[key] = nn.Sequential(*layers)
                total_concat_size += embedding_infos[key][2] * embedding_infos[key][1]
            else:
                extractors[key] = nn.Identity()
                total_concat_size += subspace.shape[0]
        self.extractors = nn.ModuleDict(extractors)

        linears = []
        for in_size, out_size in zip([total_concat_size] + shared[:-1], shared):
            linears.append(nn.Linear(in_size, out_size))
            linears.append(nn.Tanh())
        self.linears = nn.Sequential(*linears)

    def forward(self, obs: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward function for the extractor.

        Args:
            obs: Dictionary with each observation of the current state.

        Returns:
            Tensor: Forward-passed output of the network.
        """
        ten_list = []
        for key, extractor in self.extractors.items():
            ten_list.append(extractor(obs[key].view(-1, 1).long()).view(obs[key].shape[0], -1).float())

        combined_tensor = torch.cat(ten_list, dim=1,)

        ret_ten = self.linears(combined_tensor)
        return ret_ten
