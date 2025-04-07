from math import ceil
from typing import Callable

import numpy as np
import numpy.typing as npt
import tensorflow as tf


class RectangleDomain:
    left_down_corner: list[float]
    right_up_corner: list[float]

    def __init__(self, left_corner: list[float], right_corner: list[float]) -> None:
        self.left_down_corner = left_corner
        self.right_up_corner = right_corner


class Block:
    __slots__ = [
        "window_function",
        "left_down_corner",
        "right_up_corner",
        "vmax",
        "vmin",
        "losses",
        "data",
        "mean",
        "std",
        "data_size",
    ]

    def get_data(self) -> npt.NDArray[np.float64]:
        # data = np.random.uniform(self.left_down_corner, self.right_up_corner, size=self.data_size)
        # data = np.linspace(self.left_down_corner, self.right_up_corner)
        return self.data

    def normalization(self, data: tf.Tensor) -> tf.Tensor:
        data_norm = (
            2.0 * ((data - self.vmin) / (self.vmax - self.vmin)) - 1.0
        )  # subdomain normalisation --- R -> [-1; 1]
        # data_norm = (data - self.mean) / self.std  # subdomain normalisation
        return data_norm

    def unnormalization(self, data: tf.Tensor) -> tf.Tensor:
        data_unnorm = (data + 1.0) * (
            self.vmax - self.vmin
        ) / 2 + self.vmin  # output unnormalisation --- [-1; 1] -> R
        # data_unnorm = data * self.std + self.mean  # output unnormalisation
        return data_unnorm

    def set_losses(self, losses: list[tuple[Callable, list[float]]]):
        res = []
        for loss, bound in losses:
            fl = True
            for lc, b, rc in zip(self.left_down_corner, bound, self.right_up_corner):
                if b is not None and (b < lc or rc < b):
                    fl = False
                    break
            if fl:
                res.append(loss)
        self.losses = res

    def __init__(self, left_corner, right_corner, window_function, data) -> None:
        self.left_down_corner: list[float] = left_corner
        self.right_up_corner: list[float] = right_corner
        self.window_function: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ] = tf.function(window_function)

        self.vmax: float = max(right_corner + left_corner)
        self.vmin: float = min(right_corner + left_corner)
        self.mean: float = np.mean(data)
        self.std: float = np.std(data)
        self.data = data
        self.data_size = data.shape


class Decomposition:
    blocks: list[Block]
    overlap: float

    def __init__(
        self,
        domain: RectangleDomain,
        overlap: float,
        block_size: float,
        offset: bool = False,
        points_per_block: int = 100,
    ) -> None:
        self.domain = domain
        self.overlap = overlap
        self.block_size = block_size

        self.blocks = []
        self.blocks_per_axis = []
        for i in range(len(domain.left_down_corner)):
            if offset:
                domain.left_down_corner[i] -= overlap
                domain.right_up_corner[i] += overlap

            number_of_blocks = 1
            last = domain.left_down_corner[i] + block_size
            while last < domain.right_up_corner[i]:
                number_of_blocks += 1
                last = last - overlap + block_size

            # size = domain.right_up_corner[i] - domain.left_down_corner[i]
            # number_of_blocks = ceil(max(0.0, size - block_size) / (block_size - overlap) + 1)
            self.blocks_per_axis.append(number_of_blocks)

        n = len(domain.left_down_corner)
        self.build_decomposition(0, [0] * n, n, points_per_block)

    def get_window_function(
        self, left_corner, right_corner, omega: float = 30
    ) -> Callable[[tf.Tensor], tf.Tensor]:
        def sigmoid(x: tf.Tensor) -> tf.Tensor:
            x_clipped = tf.clip_by_value(x, -50.0, 50.0)
            return tf.maximum(1 / (1 + tf.math.exp(-x_clipped)), 1e-10)

        left_corner_np = tf.constant(left_corner, dtype=tf.float32)
        right_corner_np = tf.constant(right_corner, dtype=tf.float32)

        def window_function(x: tf.Tensor) -> tf.Tensor:
            left = sigmoid((x - (left_corner_np + self.overlap / 2.0)) * omega)
            right = sigmoid(((right_corner_np - self.overlap / 2.0) - x) * omega)
            return left * right

        return window_function

    def build_decomposition(
        self,
        current_ax: int,
        current_idx: list[int],
        n: int,
        points_per_block: int = 50,
    ) -> None:
        if current_ax == n - 1:
            left_corner = []
            right_corner = []
            for j in range(n):
                curr_id = current_idx[j]
                lc = (
                    self.domain.left_down_corner[j]
                    + (self.block_size - self.overlap) * curr_id
                )
                rc = (
                    self.domain.left_down_corner[j]
                    + self.block_size
                    + (self.block_size - self.overlap) * curr_id
                )

                left_corner.append(lc)
                right_corner.append(rc)
            for i in range(self.blocks_per_axis[current_ax]):
                left_corner[-1] = (
                    self.domain.left_down_corner[-1]
                    + (self.block_size - self.overlap) * i
                )
                right_corner[-1] = (
                    self.domain.left_down_corner[-1]
                    + self.block_size
                    + (self.block_size - self.overlap) * i
                )
                lc_list = left_corner.copy()
                rc_list = right_corner.copy()
                for rc_i in range(len(rc_list)):
                    for rc_bound in self.domain.right_up_corner:
                        if rc_list[rc_i] == rc_bound:
                            rc_list[rc_i] += self.overlap
                window_function = self.get_window_function(lc_list, rc_list)
                data = np.random.uniform(lc_list, rc_list, size=points_per_block)
                # data = np.linspace(lc_list, rc_list, num=points_per_block)
                block = Block(lc_list, rc_list, window_function, data)
                self.blocks.append(block)
        else:
            while current_idx[current_ax] < self.blocks_per_axis[current_ax]:
                self.build_decomposition(current_ax + 1, current_idx, n)
                current_idx[current_ax] += 1
            current_idx[current_ax] = 0
