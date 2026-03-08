import os
import pytest
import numpy as np
from degann.search_algorithms import grid_search, random_search, simulated_annealing
from degann.networks.topology.DenseNet.parameter_space import DenseNetParameterSpace
from degann.networks.topology.GAN.parameter_space import GANParameterSpace


@pytest.fixture
def train_file_name():
    return "exp_150_train.csv"


@pytest.fixture
def validate_file_name():
    return "exp_150_validate.csv"


@pytest.fixture
def equation_data(train_file_name, validate_file_name):
    test_dir = os.path.dirname(__file__)
    folder_path = os.path.join(test_dir, "data")
    train_data = np.genfromtxt(folder_path + "/" + train_file_name, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((-1, 1))
    train_data_y = train_data_y.reshape((-1, 1))

    validation_data = np.genfromtxt(
        folder_path + "/" + validate_file_name, delimiter=","
    )
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((-1, 1))
    validation_data_y = validation_data_y.reshape((-1, 1))

    return ((train_data_x, train_data_y), (validation_data_x, validation_data_y))


def test_grid_search(equation_data):
    """
    Test grid search.
    """
    train_data, val_data = equation_data

    params = DenseNetParameterSpace(
        input_size=1,
        output_size=1,
        optimizers=["Adam"],
        losses=["MeanSquaredError"],
        min_epoch=5,
        max_epoch=5,
        nn_min_length=1,
        nn_max_length=1,
        nn_alphabet=["0a"],
    )

    result = grid_search(
        data=train_data,
        params=params,
        val_data=val_data,
    )

    loss, epoch, loss_func, opt, net = result

    assert loss < 1e6
    assert loss >= 0


def test_random_search(equation_data):
    """
    Test random search.
    """
    train_data, val_data = equation_data

    params = DenseNetParameterSpace(
        input_size=1,
        output_size=1,
        optimizers=["Adam"],
        losses=["MeanSquaredError"],
        min_epoch=5,
        max_epoch=5,
        nn_min_length=1,
        nn_max_length=2,
        nn_alphabet=["0a", "f8"],
    )

    result = random_search(
        data=train_data,
        params=params,
        iterations=3,
        val_data=val_data,
    )

    loss, epoch, loss_func, opt, net = result

    assert loss < 1e6
    assert loss >= 0


def test_simulated_annealing(equation_data):
    """
    Test simulated annealing.
    """
    train_data, val_data = equation_data

    params = DenseNetParameterSpace(
        input_size=1,
        output_size=1,
        optimizers=["Adam"],
        losses=["MeanSquaredError"],
        min_epoch=5,
        max_epoch=5,
        nn_min_length=1,
        nn_max_length=2,
        nn_alphabet=["0a", "f8"],
    )

    result = simulated_annealing(
        data=train_data,
        params=params,
        max_iter=5,
        val_data=val_data,
        threshold=1.0,
    )

    loss, epoch, loss_func, opt, net, k = result

    assert loss < 1e6
    assert loss >= 0

    assert k > 0
    assert k <= 5


def test_grid_search_gan(equation_data):
    """
    Test GAN grid search.
    """
    train_data, val_data = equation_data

    params = GANParameterSpace(
        gen_input_size=1,
        gen_output_size=1,
        gen_layer_sizes=[8],
        gen_min_depth=1,
        gen_max_depth=1,
        gen_activation_funcs=["relu"],
        gen_out_activation="linear",
        disc_layer_sizes=[8],
        disc_min_depth=1,
        disc_max_depth=1,
        disc_activation_funcs=["leaky_relu"],
        gen_optimizers=["Adam"],
        disc_optimizers=["Adam"],
        gen_loss_funcs=["MeanSquaredError"],
        disc_loss_funcs=["MeanSquaredError"],
        epochs=[5],
    )

    result = grid_search(
        data=train_data,
        params=params,
        val_data=val_data,
    )

    loss, epoch, loss_func, opt, net = result

    assert loss < 1e6
    assert loss >= 0

    assert net.get("net_type") == "GAN"
    assert "generator" in net
    assert "discriminator" in net
