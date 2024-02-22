import torch

from indigo_league.training.network.dense_ensemble import DenseEnsembleLayer
from indigo_league.training.network.dense_ensemble import EnsembleNetwork


def test_dense_ensemble_layer_initialization():
    in_size, out_size, ensemble_size = 10, 20, 5
    layer = DenseEnsembleLayer(in_size, out_size, ensemble_size, bias=True)

    assert layer.weights.shape == (ensemble_size, in_size, out_size)
    assert layer.bias.shape == (ensemble_size, 1, out_size)


def test_dense_ensemble_layer_forward():
    in_size, out_size, ensemble_size = 10, 20, 5
    layer = DenseEnsembleLayer(in_size, out_size, ensemble_size)
    x = torch.rand(ensemble_size, 100, in_size)  # Batch size of 100
    out = layer(x)

    assert out.shape == (ensemble_size, 100, out_size)


def test_ensemble_network_initialization():
    in_size, out_size, ensemble_size, layer_sizes = 10, 2, 3, [20, 30]
    network = EnsembleNetwork(in_size, out_size, ensemble_size, layer_sizes)

    # Ensure layers are correctly added
    assert (
        len(network.layers) == 6
    )  # Input layer + DenseEnsembleLayer + 1 Output Layer + 3 ReLU
    assert all(isinstance(network.layers[i * 2], DenseEnsembleLayer) for i in range(2))
    assert all(isinstance(network.layers[i * 2 + 1], torch.nn.ReLU) for i in range(2))


def test_ensemble_network_forward():
    in_size, out_size, ensemble_size, layer_sizes = 10, 2, 3, [20, 30]
    network = EnsembleNetwork(in_size, out_size, ensemble_size, layer_sizes)
    x = torch.rand(ensemble_size, 100, in_size)  # Batch size of 100
    out = network(x)

    assert out.shape == (100, out_size)  # Averaged over the ensemble size


@torch.no_grad()
def test_ensemble_model_independence():
    in_size, out_size, ensemble_size, layer_sizes = (
        10,
        2,
        2,
        [20, 30],
    )  # ensemble_size is 2
    network = EnsembleNetwork(
        in_size, out_size, ensemble_size, layer_sizes, average_pool=False
    )

    # Original forward pass with random inputs
    x = torch.rand(ensemble_size, 1, in_size)  # Batch size of 1 for simplicity
    original_out = network(x)

    # Modify the weights of the first model in the ensemble
    for layer in network.layers:
        if isinstance(layer, DenseEnsembleLayer):
            layer.weights[0] += 1.0  # Add 1 to the weights of the first ensemble model

    # Forward pass after modifying the first model
    modified_out = network(x)

    # Check that the outputs for the second model in the ensemble before averaging remain unchanged
    # This step may require modifying the EnsembleNetwork to allow inspection of individual model outputs
    assert torch.allclose(
        original_out[1], modified_out[1], atol=1e-6
    ), "Model independence violated"
