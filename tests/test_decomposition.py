import math
import pytest

import numpy as np
from functools import reduce

import tensorflow as tf
from degann.geometry import Decomposition, RectangleDomain
from degann.geometry.decomposition import Block


@pytest.mark.parametrize(
    "left_corner, right_corner, block_size, overlap, expected_blocks_count",
    [
        ([0, 0], [9, 7], 3, 1, 12),
        ([0], [7], 3, 1, 3),
        ([0, 0], [1, 7], 3, 1, 3),
        ([1, 1], [3, 3], 3, 1, 1),
        ([0, 0], [4, 4], 3, 1, 4),
        ([0, 0], [5, 5], 3, 2, 9),
        ([0, 0], [4, 4], 3, 2, 4),
        ([0, 0], [3, 4], 3, 2, 2),
        ([0, 0, 0], [3, 3, 3], 3, 2, 1),
        ([0, 0, 0], [4, 4, 4], 3, 2, 8),
        ([0, 0, 0], [3, 3, 3], 3, 1, 1),
        ([1, 2], [10, 9], 3, 1, 12),
        ([0, 0], [1, 1], 3, 1, 1),
        ([0], [1], 0.2, 0.01, 6),
        ([0], [10], 2, 0.1, 6),
        ([0], [1], 0.4, 0.1, 3),
    ],
)
def test_decomposition(
    left_corner, right_corner, block_size, overlap, expected_blocks_count
):
    domain = RectangleDomain(left_corner, right_corner)
    actual_decomposition = Decomposition(domain, overlap, block_size)

    assert reduce(lambda x, y: x * y, actual_decomposition.blocks_per_axis) == len(
        actual_decomposition.blocks
    )
    assert len(actual_decomposition.blocks) == expected_blocks_count


@pytest.mark.parametrize(
    "left_corner, right_corner, overlap, input, expected",
    [
        ([0], [1], 0.05, 20, 0),
        ([4], [6], 0.05, 5, 0),
        ([4], [6], 2, 4, -1),
    ],
)
def test_block(left_corner, right_corner, overlap, input, expected):
    def get_window_function(overlap, left_corner, right_corner, omega: float = 0.1):
        def sigmoid(x):
            return 1 / (1 + tf.math.exp(-x))

        left_corner_np = np.array(left_corner)
        right_corner_np = np.array(right_corner)

        def window_function(x):
            left = sigmoid((x - (left_corner_np + overlap / 2.0)) / omega)
            right = sigmoid(((right_corner_np - overlap / 2.0) - x) / omega)
            return left * right

        return window_function

    wf = get_window_function(overlap, left_corner, right_corner)
    data = np.linspace(left_corner, right_corner, num=50)

    block = Block(left_corner, right_corner, wf, data)

    model = lambda x: x - 5
    x_norm = block.normalization(input)
    predicted = model(x_norm)
    predicted_unnorm: tf.Tensor = block.unnormalization(predicted)

    windowed = block.window_function(input)
    u = windowed * predicted_unnorm
    u_py = u.numpy().item()

    assert math.isclose(u_py, expected, abs_tol=1e-16)
