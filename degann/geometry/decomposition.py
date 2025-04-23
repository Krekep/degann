from math import ceil, inf
from typing import Callable

import numpy as np
import numpy.typing as npt
import tensorflow as tf


class RectangleDomain:
    left_down_corner: list[float]
    right_up_corner: list[float]

    def __init__(
        self, left_corner: list[float], right_corner: list[float], time_index: int = -1
    ) -> None:
        self.left_down_corner = left_corner
        self.right_up_corner = right_corner
        self.time_index = time_index  # for window function


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
        "__weakref__",
    ]

    @tf.function
    def get_data(self) -> tf.Tensor:
        data = tf.random.uniform(
            shape=self.data.shape,
            minval=self.left_down_corner,
            maxval=self.right_up_corner,
            dtype=tf.float32,
        )
        return data
        # return self.data

    @tf.function
    def normalization(self, data: tf.Tensor) -> tf.Tensor:
        data_norm = (
            2.0 * ((data - self.vmin) / (self.vmax - self.vmin)) - 1.0
        )  # subdomain normalisation --- R -> [-1; 1]
        # data_norm = (data - self.mean) / self.std  # subdomain normalisation
        return data_norm

    @tf.function
    def unnormalization(self, data: tf.Tensor) -> tf.Tensor:
        data_unnorm = (data + 1.0) * (
            self.vmax - self.vmin
        ) / 2 + self.vmin  # output unnormalisation --- [-1; 1] -> R
        # data_unnorm = data * self.std + self.mean  # output unnormalisation
        return data_unnorm

    def set_losses(
        self,
        losses: list[tuple[Callable, list[tuple[float]]]],
        time_input: bool = False,
    ) -> None:
        res = []
        if time_input:
            left_down_corner = [-inf] + self.left_down_corner
            right_up_corner = [inf] + self.right_up_corner
        else:
            left_down_corner = self.left_down_corner
            right_up_corner = self.right_up_corner
        for loss, var_bound in losses:
            fl = True
            for lc, var_b, rc in zip(left_down_corner, var_bound, right_up_corner):
                if var_b is not None:
                    for b in var_b:
                        if b < lc or rc < b:
                            fl = False
                            break
            if fl:
                res.append(loss)
        self.losses = res

    def __init__(self, left_corner, right_corner, window_function, data) -> None:
        self.left_down_corner: list[float] = left_corner
        self.right_up_corner: list[float] = right_corner
        self.window_function: Callable[
            [npt.NDArray[np.float64]],
            npt.NDArray[np.float64]
            # ] = tf.function(window_function)
        ] = window_function

        self.vmax: tf.Tensor = tf.constant(
            max(right_corner + left_corner), dtype=tf.float32
        )
        self.vmin: tf.Tensor = tf.constant(
            min(right_corner + left_corner), dtype=tf.float32
        )
        self.mean: float = np.mean(data)
        self.std: float = np.std(data)
        self.data: tf.Tensor = data
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
            self.blocks_per_axis.append(number_of_blocks)

        n = len(domain.left_down_corner)
        self.build_decomposition(0, [0] * n, n, points_per_block)

    def get_window_function(
        self, left_corner, right_corner, omega: float = 30
    ) -> Callable[[tf.Tensor], tf.Tensor]:
        def sigmoid(x: tf.Tensor) -> tf.Tensor:
            x_clipped = tf.clip_by_value(x, -50.0, 50.0)
            return tf.maximum(1 / (1 + tf.math.exp(-x_clipped)), 1e-10)

        left_corner_np = tf.constant(
            left_corner[self.domain.time_index + 1 :], dtype=tf.float32
        )
        right_corner_np = tf.constant(
            right_corner[self.domain.time_index + 1 :], dtype=tf.float32
        )

        def window_function(x_in: tf.Tensor) -> tf.Tensor:
            x = x_in[:, self.domain.time_index + 1 :]
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
                size = [points_per_block] + [len(lc_list)]
                data = tf.random.uniform(
                    # shape=size + [1],
                    shape=size,
                    minval=lc_list,
                    maxval=rc_list,
                    dtype=tf.float32,
                )
                data = tf.sort(data, axis=0)
                block = Block(lc_list, rc_list, window_function, data)
                self.blocks.append(block)
        else:
            while current_idx[current_ax] < self.blocks_per_axis[current_ax]:
                self.build_decomposition(
                    current_ax + 1, current_idx, n, points_per_block
                )
                current_idx[current_ax] += 1
            current_idx[current_ax] = 0
