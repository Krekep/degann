import pytest

import numpy as np
from degann.networks.imodel import IModel
from degann.search_algorithms import pattern_search, grid_search, random_search_endless, simulated_annealing


@pytest.mark.parametrize(
    "path_to_train_data, path_to_validate_data",
    [
        (
            "./tests/data/exp_150_train.csv",
            "./tests/data/exp_150_validate.csv",
        ),
    ],
)
def test_pattern_search(path_to_train_data, path_to_validate_data):
    train_data = np.genfromtxt(path_to_train_data, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((1, -1)).T
    train_data_y = train_data_y.reshape((1, -1)).T

    validation_data = np.genfromtxt(path_to_validate_data, delimiter=",")
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((1, -1)).T
    validation_data_y = validation_data_y.reshape((1, -1)).T

    config = {
        "loss_functions": ["MeanSquaredError"],
        "optimizers": ["Adam"],
        "metrics": ["MaxAbsoluteDeviation", "MeanSquaredLogarithmicError"],
        "net_shapes": [
            [20],
            [10]
        ],
        "activations": ["parabolic", "exponential"],
        "validation_split": 0,
        "rates": [1e-2],
        "epochs": [20],
        "normalize": [False],
        "use_rand_net": False,
    }

    best_nns = pattern_search(
        x_data=train_data_x,
        y_data=train_data_y,
        x_val=validation_data_x,
        y_val=validation_data_y,
        **config
    )
    assert True


@pytest.mark.parametrize(
    "path_to_train_data, path_to_validate_data, in_size, out_size",
    [
        (
            "./tests/data/exp_150_train.csv",
            "./tests/data/exp_150_validate.csv",
            1,
            1,
        ),
    ],
)
def test_grid_search(path_to_train_data, path_to_validate_data, in_size, out_size):
    train_data = np.genfromtxt(path_to_train_data, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((1, -1)).T
    train_data_y = train_data_y.reshape((1, -1)).T

    validation_data = np.genfromtxt(path_to_validate_data, delimiter=",")
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((1, -1)).T
    validation_data_y = validation_data_y.reshape((1, -1)).T

    result_loss, result_epoch, result_loss_name, result_optimizer, result_nn = grid_search(
        input_size=in_size,
        output_size=out_size,
        data=(train_data_x, train_data_y),
        val_data=(validation_data_x, validation_data_y),
        optimizers=["Adam"],
        loss=["MeanAbsolutePercentageError"],
        min_epoch=10,
        max_epoch=20,
        epoch_step=10,
        nn_min_length=1,
        nn_max_length=2,
        nn_alphabet=["0a", "42"],
    )
    assert True


@pytest.mark.parametrize(
    "path_to_train_data, path_to_validate_data, in_size, out_size",
    [
        (
            "./tests/data/exp_150_train.csv",
            "./tests/data/exp_150_validate.csv",
            1,
            1,
        ),
    ],
)
def test_random_search(path_to_train_data, path_to_validate_data, in_size, out_size):
    train_data = np.genfromtxt(path_to_train_data, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((1, -1)).T
    train_data_y = train_data_y.reshape((1, -1)).T

    validation_data = np.genfromtxt(path_to_validate_data, delimiter=",")
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((1, -1)).T
    validation_data_y = validation_data_y.reshape((1, -1)).T

    result_loss, result_epoch, result_loss_name, result_optimizer, result_nn, final_iteration = random_search_endless(
        input_size=in_size,
        output_size=out_size,
        data=(train_data_x, train_data_y),
        val_data=(validation_data_x, validation_data_y),
        opt="Adam",
        loss="MaxAbsolutePercentageError",
        threshold=20,
        min_epoch=10,
        max_epoch=20,
        max_iter=10,
        nn_min_length=1,
        nn_max_length=3,
        nn_alphabet=["0a", "f8", "42"],
    )
    assert True


@pytest.mark.parametrize(
    "path_to_train_data, path_to_validate_data, in_size, out_size",
    [
        (
            "./tests/data/exp_150_train.csv",
            "./tests/data/exp_150_validate.csv",
            1,
            1,
        ),
    ],
)
def test_sam(path_to_train_data, path_to_validate_data, in_size, out_size):
    train_data = np.genfromtxt(path_to_train_data, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((1, -1)).T
    train_data_y = train_data_y.reshape((1, -1)).T

    validation_data = np.genfromtxt(path_to_validate_data, delimiter=",")
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((1, -1)).T
    validation_data_y = validation_data_y.reshape((1, -1)).T

    result_loss, result_epoch, result_loss_name, result_optimizer, result_nn, final_iteration = simulated_annealing(
        input_size=in_size,
        output_size=out_size,
        data=(train_data_x, train_data_y),
        val_data=(validation_data_x, validation_data_y),
        opt="Adam",
        loss="Huber",
        threshold=1,
        min_epoch=10,
        max_epoch=20,
        max_iter=10,
        nn_min_length=1,
        nn_max_length=3,
        nn_alphabet=["0a", "f8", "42"],
    )
    assert True
