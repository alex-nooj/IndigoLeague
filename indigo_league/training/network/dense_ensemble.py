import math
import typing
from collections import OrderedDict

import torch
from pympler import asizeof
from torch import nn


class DenseEnsembleLayer(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, ensemble_size: int, bias: bool = True
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(ensemble_size, in_size, out_size))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = nn.Parameter(torch.rand(ensemble_size, 1, out_size))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = torch.zeros((ensemble_size, 1, out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.matmul(self.weights) + self.bias
        return y


from memory_profiler import profile


class EnsembleNetwork(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        ensemble_size: int,
        layer_sizes: typing.List[int],
        bias: bool = True,
        average_pool: bool = True,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_size = in_size
        self.out_size = out_size

        in_sizes = [in_size] + layer_sizes
        out_sizes = layer_sizes + [out_size]

        layers = []
        for ix, (input_size, output_size) in enumerate(zip(in_sizes, out_sizes)):
            layers.append(
                (
                    f"ensemble{ix}",
                    DenseEnsembleLayer(
                        input_size, output_size, ensemble_size, bias=bias
                    ),
                )
            )
            layers.append((f"relu{ix}", nn.LeakyReLU()))
        self.layers = nn.Sequential(OrderedDict(layers))
        self.average_pool = average_pool
        print(__name__, asizeof.asizeof(self) / 1e9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.average_pool:
            return self.layers.forward(x).mean(dim=0)
        else:
            return self.layers.forward(x)
