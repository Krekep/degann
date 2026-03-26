import pytest
import os
import numpy as np

from degann.networks.imodel import IModel
from degann.networks.topology.DenseNet.config import DenseNetConfig
from tests.utils import array_compare, file_compare


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

    nn_loaded = IModel(config=config, net_type="DenseNet")
    nn_loaded.from_file(f"{folder_path}/test_export")
    nn_loaded.export_to_file(f"{folder_path}/test_export1")
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
