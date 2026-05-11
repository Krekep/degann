import pytest
import os
import numpy as np

from degann.networks.imodel import IModel
from degann.networks.topology.DenseNet.config import DenseNetConfig
from tests.utils import array_compare, file_compare
from degann.networks.topology.trainer import train


@pytest.fixture
def folder_path():
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, "data")


@pytest.mark.parametrize(
    "inp, shape, act_init, decorator_params",
    [
        (np.array([[1]], dtype=float), [1, [1], 1], ["sigmoid", "linear"], None),
        (np.array([[1]], dtype=float), [1, [1], 1], "sigmoid", None),
        (np.array([[1]], dtype=float), [1, [1], 1], ["linear", "linear"], None),
        (np.array([[1, 1]], dtype=float), [2, [1], 1], "tanh", None),
        (np.array([[1], [1]], dtype=float), [1, [1], 1], "tanh", None),
    ],
)
def test_predict_is_same(inp, shape, act_init, decorator_params, folder_path):
    if isinstance(act_init, str):
        activation_list = [act_init] * len(shape[1]) + ["linear"]
    else:
        activation_list = act_init

    config = DenseNetConfig(
        input_size=shape[0],
        output_size=shape[2],
        layer_sizes=shape[1],
        activation_funcs=activation_list,
    )

    nn = IModel(config=config, net_type="DenseNet")

    expected = nn.feedforward(inp).numpy()
    nn.export_to_file(f"{folder_path}/test_export")

    nn_loaded = IModel.from_file(f"{folder_path}/test_export")
    actual = nn_loaded.feedforward(inp).numpy()

    assert array_compare(actual, expected)


@pytest.mark.parametrize(
    "inp, shape",
    [
        (np.array([[1]], dtype=float), [1, [1], 1]),
        (np.array([[1, 1]], dtype=float), [2, [1], 1]),
        (np.array([[1], [1]], dtype=float), [1, [1], 1]),
        (np.array([[1, 1], [1, 1]], dtype=float), [2, [1], 2]),
        (np.array([[1, 1], [1, 1]], dtype=float), [2, [1], 1]),
    ],
)
def test_file_is_same(inp, shape, folder_path):
    config = DenseNetConfig(
        input_size=shape[0],
        output_size=shape[2],
        layer_sizes=shape[1],
        activation_funcs=["tanh"] * len(shape[1]) + ["linear"],
    )

    nn = IModel(config=config, net_type="DenseNet")
    nn.export_to_file(f"{folder_path}/test_export")

    nn_loaded = IModel(config=config, net_type="DenseNet")
    nn_loaded.from_file(f"{folder_path}/test_export")
    nn_loaded.export_to_file(f"{folder_path}/test_export1")

    assert file_compare(
        f"{folder_path}/test_export.apg", f"{folder_path}/test_export1.apg"
    )


@pytest.mark.parametrize(
    "shape",
    [
        [1, [5], 1],
        [1, [10, 10], 2],
        [1, [32, 16, 8], 3],
        [1, [8, 23, 12], 4],
        [1, [64, 32, 16, 8], 5],
    ],
)
def test_weights_preserved_after_train_and_dict_roundtrip(shape):
    config = DenseNetConfig(
        input_size=shape[0],
        output_size=shape[2],
        layer_sizes=shape[1],
        activation_funcs=["tanh"] * len(shape[1]) + ["linear"],
    )

    train_data = np.random.rand(100, shape[0]), np.random.rand(100, shape[2])
    _, _, model_dict = train(config=config, num_epochs=5, data=train_data)

    loaded_nn = IModel.from_dict(model_dict)

    for orig, load in zip(model_dict["layer"], loaded_nn.network.blocks):
        assert array_compare(orig["weights"], load.w.numpy())
        assert array_compare(orig["biases"], load.b.numpy())

    assert array_compare(
        model_dict["out_layer"]["weights"], loaded_nn.network.out_layer.w.numpy()
    )
    assert array_compare(
        model_dict["out_layer"]["biases"], loaded_nn.network.out_layer.b.numpy()
    )


@pytest.mark.parametrize(
    "shape, act_init",
    [
        ([1, [10, 10], 1], ["tanh", "tanh", "tanh", "linear"]),
        ([2, [8, 4], 3], ["relu", "sigmoid", "relu", "linear"]),
        ([3, [16, 8], 2], ["gelu", "swish", "linear"]),
        ([1, [5, 10, 5], 1], ["sigmoid", "tanh", "elu", "linear"]),
        ([4, [32], 2], ["softplus", "linear"]),
    ],
)
def test_predict_is_same_dict(shape, act_init):
    if isinstance(act_init, str):
        activation_list = [act_init] * len(shape[1]) + ["linear"]
    else:
        activation_list = act_init

    inp = np.random.randn(100, shape[0])

    config = DenseNetConfig(
        input_size=shape[0],
        output_size=shape[2],
        layer_sizes=shape[1],
        activation_funcs=activation_list,
    )

    nn = IModel(config=config, net_type="DenseNet")

    expected = nn.feedforward(inp).numpy()
    conf = nn.to_dict()

    new_nn = IModel.from_dict(conf)
    actual = new_nn.feedforward(inp).numpy()

    assert array_compare(actual, expected)
