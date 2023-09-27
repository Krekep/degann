import numpy as np
import pytest

from degann.networks.activations import perceptron_threshold, parabolic
from tests.utils import array_compare


@pytest.mark.parametrize(
    "threshold, inp, expected",
    [
        (2, np.array([10], dtype=float), np.array([1], dtype=float)),
        (2, np.array([1], dtype=float), np.array([0], dtype=float)),
        (2, np.array([-10], dtype=float), np.array([0], dtype=float)),
        (0, np.array([10], dtype=float), np.array([1], dtype=float)),
        (0, np.array([1], dtype=float), np.array([1], dtype=float)),
        (0, np.array([-10], dtype=float), np.array([0], dtype=float)),
    ],
)
def test_perceptron_activation(threshold, inp, expected):
    act_func = perceptron_threshold
    assert array_compare(act_func(inp, threshold), expected)


@pytest.mark.parametrize(
    "inp, beta, p, expected",
    [
        (np.array([10], dtype=float), 0, 1 / 5, np.array([2], dtype=float)),
        (np.array([1], dtype=float), 0, 1 / 50, np.array([0.2], dtype=float)),
        (np.array([-10], dtype=float), 0, 1 / 5, np.array([-2], dtype=float)),
        (np.array([10], dtype=float), 1, 1 / 5, np.array([3], dtype=float)),
        (np.array([1], dtype=float), 1, 2, np.array([3], dtype=float)),
        (np.array([-10], dtype=float), -3, 5, np.array([-13], dtype=float)),
        (np.array([0], dtype=float), 42, 1 / 5, np.array([42], dtype=float)),
    ],
)
def test_parabolic_activation(inp, beta, p, expected):
    act_func = parabolic
    assert array_compare(act_func(inp, beta, p), expected)
