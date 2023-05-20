import pytest
import numpy as np

from networks.topology.densenet import DenseNet
from networks.imodel import IModel
from tests.utils import array_compare, init_params


@pytest.mark.parametrize(
    "inp, shape, act_init, w_init, b_init, out_size, expected",
    [
        (
            np.array([[1]], dtype=float),
            [1, [1]],
            "linear",
            "ones",
            "zeros",
            1,
            np.array([[1]], dtype=float),
        ),
        (
            np.array([[1]], dtype=float),
            [1, [1]],
            "linear",
            "zeros",
            "zeros",
            1,
            np.array([[0]], dtype=float),
        ),
        (
            np.array([[1]], dtype=float),
            [1, [1]],
            "linear",
            "zeros",
            "ones",
            1,
            np.array([[1]], dtype=float),
        ),
        (
            np.array([[1, 1]], dtype=float),
            [2, [1]],
            "linear",
            "ones",
            "zeros",
            1,
            np.array([[2]], dtype=float),
        ),
        (
            np.array([[1], [1]], dtype=float),
            [1, [1]],
            "linear",
            "ones",
            "zeros",
            1,
            np.array([[1], [1]], dtype=float),
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1]],
            "linear",
            "ones",
            "zeros",
            1,
            np.array([[2], [2]], dtype=float),
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1]],
            "linear",
            "ones",
            "zeros",
            2,
            np.array([[2, 2], [2, 2]], dtype=float),
        ),
        (
            np.array([[0]], dtype=float),
            [1, []],
            "linear",
            "ones",
            "zeros",
            1,
            np.array([[0]], dtype=float),
        ),
        (
            np.array([[0]], dtype=float),
            [1, []],
            "linear",
            "ones",
            "ones",
            1,
            np.array([[1]], dtype=float),
        ),
    ],
)
def test_densenet_predict(inp, shape, act_init, w_init, b_init, out_size, expected):
    weight_initializer, bias_initializer = init_params(
        weight_name=w_init, bias_name=b_init
    )

    nn = DenseNet(
        shape[0],
        shape[1],
        activation_func=act_init,
        weight=weight_initializer,
        biases=bias_initializer,
        output_size=out_size,
    )
    res = nn(inp).numpy()
    assert array_compare(res, expected)


@pytest.mark.parametrize(
    "inp, shape, act_init, w_init, b_init, out_size, expected",
    [
        (
            np.array([[1]], dtype=float),
            [1, [1]],
            "linear",
            "ones",
            "zeros",
            1,
            np.array([[1]], dtype=float),
        ),
        (
            np.array([[1]], dtype=float),
            [1, [1]],
            "linear",
            "zeros",
            "zeros",
            1,
            np.array([[0]], dtype=float),
        ),
        (
            np.array([[1]], dtype=float),
            [1, [1]],
            "linear",
            "zeros",
            "ones",
            1,
            np.array([[1]], dtype=float),
        ),
        (
            np.array([[1, 1]], dtype=float),
            [2, [1]],
            "linear",
            "ones",
            "zeros",
            1,
            np.array([[2]], dtype=float),
        ),
        (
            np.array([[1], [1]], dtype=float),
            [1, [1]],
            "linear",
            "ones",
            "zeros",
            1,
            np.array([[1], [1]], dtype=float),
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1]],
            "linear",
            "ones",
            "zeros",
            1,
            np.array([[2], [2]], dtype=float),
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1]],
            "linear",
            "ones",
            "zeros",
            2,
            np.array([[2, 2], [2, 2]], dtype=float),
        ),
        (
            np.array([[-1, -1], [-1, -1]], dtype=float),
            [2, [1]],
            "linear",
            "ones",
            "zeros",
            1,
            np.array([[-2], [-2]], dtype=float),
        ),
    ],
)
def test_neuron_predict(inp, shape, act_init, w_init, b_init, out_size, expected):
    weight_initializer, bias_initializer = init_params(
        weight_name=w_init, bias_name=b_init
    )

    nn = IModel.create_neuron(
        shape[0],
        out_size,
        shape[1],
        activation=act_init,
        weight=weight_initializer,
        biases=bias_initializer,
    )
    res = nn.feedforward(inp).numpy()
    assert array_compare(res, expected)
