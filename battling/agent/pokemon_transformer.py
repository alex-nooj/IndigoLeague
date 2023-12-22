from collections import OrderedDict

import torch
from torch import nn


class PokemonTransformer(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
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

        output_layers = []
        curr_out_size = d_encoder
        while curr_out_size > 2 * out_size:
            output_layers.append(
                (
                    f"Projection Layer {curr_out_size}",
                    nn.Linear(curr_out_size, curr_out_size // 2),
                )
            )
            output_layers.append((f"Projection ReLU {curr_out_size}", nn.ReLU()))
            curr_out_size = curr_out_size // 2
        output_layers.append(
            (f"Projection Layer {out_size}", nn.Linear(curr_out_size, out_size))
        )
        output_layers.append((f"Projection ReLU {out_size}", nn.ReLU()))
        self.second_projection_layer = nn.Sequential(OrderedDict(output_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, f = x.shape
        projected_tensor = torch.reshape(
            self.projection_layer(x.view(b * s, f)),
            (b, s, -1),
        )
        positional_encoding = (
            torch.linspace(0, 1, s, device=x.device)
            .view(1, -1, 1)
            .expand(b, -1, -1)
        )

        encoded_tensor = torch.cat((positional_encoding, projected_tensor), dim=-1)
        transformer_tensor = self.transformer(encoded_tensor)
        final_projection = self.second_projection_layer(transformer_tensor.view(-1, transformer_tensor.shape[-1]))
        return final_projection.view(b, -1)
