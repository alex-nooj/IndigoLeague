import typing
from collections import OrderedDict

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class PokemonTransformer(nn.Module):
    def __init__(
        self,
        in_size: int,
        d_encoder: int,
        seq_len: int,
        n_encoders: int,
        n_heads: int,
        d_feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.projection_layer = nn.Sequential(
            OrderedDict(
                [
                    ("Projection Layer", nn.Linear(in_size // seq_len, d_encoder - 1)),
                    ("Projection ReLU", nn.ReLU()),
                ]
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_encoder,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_encoders)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected_tensor = torch.reshape(
            self.projection_layer(x.view(-1, x.shape[-1])),
            (x.shape[0], x.shape[1], -1),
        )
        positional_encoding = (
            torch.linspace(0, 1, x.shape[1], device=x.device)
            .view(1, -1, 1)
            .expand(x.shape[0], -1, -1)
        )

        encoded_tensor = torch.cat((positional_encoding, projected_tensor), dim=-1)
        return torch.mean(self.transformer(encoded_tensor), dim=1)


class RemoveSeqLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.reshape(x, (x.shape[0], -1))
        return x


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """Feature Extractor for the Pokemon AI."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        embedding_infos: typing.Dict[str, typing.Tuple[int, int, int]],
        seq_len: int,
        shared: typing.List[int],
        n_linear_layers: int,
        n_encoders: int,
        n_heads: int,
        d_feedforward: int,
        dropout: float = 0.0,
    ):
        """Constructor function.

        Args:
            observation_space: Shape and type of the observation space of the environment.
            embedding_infos: Describes which items in the observation space need an embedding layer.
            n_linear_layers: Number of linear layers to append after the embedding.
            n_encoders: The number of encoder layers to use.
            n_heads: The number of heads in the multi-head attention layer.
            d_feedforward: The number of features in the feed-forward network of the transformer.
            dropout: The dropout percentage in the transformer

        """
        super().__init__(observation_space, features_dim=shared[-1])
        extractors = {}
        input_size = 0
        for key, subspace in observation_space.items():
            if key in embedding_infos:
                layers = [
                    nn.Embedding(embedding_infos[key][0], embedding_infos[key][1]),
                    nn.Linear(
                        embedding_infos[key][1],
                        embedding_infos[key][1],
                    ),
                    nn.ReLU(),
                ]
                extractors[key] = nn.Sequential(*layers)
                input_size += embedding_infos[key][2] * embedding_infos[key][1]
            else:
                extractors[key] = nn.Identity()
                input_size += subspace.shape[0]
        self.extractors = nn.ModuleDict(extractors)
        self.seq_len = seq_len
        print(f"Input size: {input_size}")
        layers = []
        if seq_len > 1:
            print(f"Encoder size: {input_size // seq_len}")
            layers = [
                (
                    "Transformer",
                    PokemonTransformer(
                        input_size,
                        512,
                        seq_len,
                        n_encoders,
                        n_heads,
                        d_feedforward,
                        dropout,
                    ),
                )
            ]
            in_vals = [512] + shared[:-1]
        else:
            print(f"Input size: {input_size * seq_len}")
            in_vals = [input_size] + shared[:-1]
            layers.append(("Remove Seq Len", RemoveSeqLayer()))

        for ix, (in_val, out_val) in enumerate(zip(in_vals, shared)):
            layers.append((f"Linear {ix}", nn.Linear(in_val, out_val)))
            layers.append((f"ReLU {ix}", nn.ReLU()))
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, obs: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward function for the extractor.

        Args:
            obs: Dictionary with each observation of the current state.

        Returns:
            Tensor: Forward-passed output of the network.
        """
        ten_list = []
        for key, extractor in self.extractors.items():
            ten_list.append(
                extractor(obs[key].view(-1, 1).long())
                .view(obs[key].shape[0], self.seq_len, -1)
                .float()
            )
        combined_tensor = torch.cat(
            ten_list,
            dim=2,
        )
        return self.layers(combined_tensor)
