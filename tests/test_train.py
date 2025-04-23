import pytest

import numpy as np
import tensorflow as tf

from degann.geometry import RectangleDomain
from degann.networks.imodel import IModel
from degann.networks.topology.tf_fbpinn import TensorflowFBPINN


@pytest.fixture
def folder_path():
    return "./data"


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
def test_densenet_training(
    path_to_train_data, path_to_validate_data, folder_path, shape, out_size
):
    train_data = np.genfromtxt(folder_path + path_to_train_data, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((1, -1)).T
    train_data_y = train_data_y.reshape((1, -1)).T

    validation_data = np.genfromtxt(folder_path + path_to_validate_data, delimiter=",")
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((1, -1)).T
    validation_data_y = validation_data_y.reshape((1, -1)).T

    nn = IModel(
        input_size=shape[0],
        block_size=shape[1],
        output_size=out_size,
    )
    nn.compile(optimizer="Adam", loss_func="MaxAbsoluteDeviation", run_eagerly=True)

    loss_before_train = nn.evaluate(validation_data_x, validation_data_y, verbose=0)
    nn.train(train_data_x, train_data_y, verbose=0)
    loss_after_train = nn.evaluate(validation_data_x, validation_data_y, verbose=0)
    assert loss_after_train < loss_before_train


@pytest.mark.parametrize(
    "in_size, out_size",
    [
        (
            1,
            1,
        ),
    ],
)
def test_densenet_training(in_size, out_size):
    def func(x):
        return np.power(np.e, 3 * x)

    def physic_loss(model: TensorflowFBPINN, x, **kwargs):
        tr_w = model.trainable_weights
        tr_v = model.trainable_variables

        with tf.GradientTape() as tape:
            u = model(x)
            gr = tape.gradient(u, x)
            u_x = gr[0]
            loss = u_x + 3 * u
            return tf.reduce_mean(tf.square(loss))

    nn = TensorflowFBPINN(
        input_size=in_size,
        output_size=out_size,
        domain=RectangleDomain([0], [1]),
        block_size=0.2,
        overlap=0.05,
        physic_loss=physic_loss,
    )

    nn.custom_compile(optimizer="Adam", loss_func="MSE", run_eagerly=False)

    # np.power(np.e, 3 * x)
    x = np.linspace(0, 1, num=400).reshape((-1, 1))
    y = func(x)
    loss_before_train = nn.evaluate(x, y, verbose=0)
    nn.fit(x, verbose=0)
    loss_after_train = nn.evaluate(x, y, verbose=0)
    assert loss_after_train < loss_before_train
