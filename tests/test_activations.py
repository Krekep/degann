import numpy as np
import pytest

from degann.networks.activations import perceptron_threshold
from tests.utils import array_compare


@pytest.mark.parametrize(
    "threshold, inp, expected",
    [
        (1, np.array([10], dtype=float), np.array([1], dtype=float)),
        (1, np.array([1], dtype=float), np.array([1], dtype=float)),
        (1, np.array([-10], dtype=float), np.array([0], dtype=float)),
        (0, np.array([10], dtype=float), np.array([1], dtype=float)),
        (0, np.array([1], dtype=float), np.array([1], dtype=float)),
        (0, np.array([-10], dtype=float), np.array([0], dtype=float)),
    ],
)
def test_create_layer(threshold, inp, expected):
    act_func = perceptron_threshold
    assert array_compare(act_func(inp), expected)


def test_my():
    assert True
