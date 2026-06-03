import os
import pytest
import numpy as np
from degann.networks.imodel import IModel
from degann.networks.topology.DenseNet.config import DenseNetConfig


@pytest.fixture
def folder_path():
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, "data")


@pytest.mark.parametrize(
    "path_to_train_data, path_to_validate_data, shape, out_size",
    [
        (
            "/exp_150_train.csv",
            "/exp_150_validate.csv",
            [1, [10, 10]],
            1,
        ),
    ],
)
def test_densenet_predict(
    path_to_train_data, path_to_validate_data, folder_path, shape, out_size
):
    train_data = np.genfromtxt(folder_path + path_to_train_data, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((-1, 1))
    train_data_y = train_data_y.reshape((-1, 1))

    validation_data = np.genfromtxt(folder_path + path_to_validate_data, delimiter=",")
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((-1, 1))
    validation_data_y = validation_data_y.reshape((-1, 1))

    activation_list = ["relu"] * len(shape[1]) + ["linear"]
    config = DenseNetConfig(
        input_size=shape[0],
        output_size=out_size,
        layer_sizes=shape[1],
        activation_funcs=activation_list,
    )
    nn = IModel(config=config, net_type="DenseNet")
    nn.compile(optimizer="Adam", loss_func="MaxAbsoluteDeviation")

    loss_before_train = nn.evaluate(validation_data_x, validation_data_y, verbose=0)
    nn.train(train_data_x, train_data_y, verbose=0, epochs=10)
    loss_after_train = nn.evaluate(validation_data_x, validation_data_y, verbose=0)
    assert loss_after_train < loss_before_train
