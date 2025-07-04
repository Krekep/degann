import csv
import random
from random import randint

import numpy as np

import functions

__all__ = ["funcs", "sizes_of_samples", "generate_size"]

funcs = [
    (functions.multidim, "multidim")
]
sizes_of_samples = [50, 150, 400]
# sizes_of_samples = [400]
# sizes_of_samples = [400, 50]
generate_size = 100_000

if __name__ == "__main__":
    for func, func_name in funcs:
        nn_data_x = np.array(
            [
                [
                    random.uniform(1 / generate_size, 1),
                    random.uniform(1 / generate_size, 1),
                    random.uniform(1 / generate_size, 1),
                ]
                for i in range(generate_size + 1)
            ]
        )  # X data
        assert len(nn_data_x.shape) == 2 and nn_data_x.shape == (generate_size + 1, 3)
        nn_data_y = np.array([[func(*x)] for x in nn_data_x])
        assert len(nn_data_y.shape) == 2 and nn_data_y.shape == (generate_size + 1, 1)
        for size in sizes_of_samples:
            train_idx = [randint(0, generate_size) for _ in range(size)]
            train_idx.sort()
            val_data_x = nn_data_x[:]  # validation X data
            val_data_y = nn_data_y[:]  # validation Y data
            train_data_x = nn_data_x[train_idx, :]  # X data
            train_data_y = nn_data_y[train_idx, :]  # Y data

            with open(f"data/{func_name}_{size}_train.csv", "w", newline="") as file:
                csv_writer = csv.writer(file)
                data = list(zip(*train_data_x.T, *train_data_y.T))
                csv_writer.writerows(data)

            with open(f"data/{func_name}_{size}_validate.csv", "w", newline="") as file:
                csv_writer = csv.writer(file)
                data = list(zip(*val_data_x.T, *val_data_y.T))
                csv_writer.writerows(data)
