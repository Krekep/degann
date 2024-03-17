import csv
import random
from random import randint

import numpy as np

import functions

__all__ = ["funcs", "sizes_of_samples", "generate_size"]

funcs = [
    # (functions.lin, "lin"),
    # (functions.log, "log"),
    # (functions.sin, "sin"),
    # (functions.exp, "exp"),
    # (functions.gauss, "gauss"),
    # (functions.hyperbol, "hyperbol"),
    # (functions.const, "const"),
    # (functions.sig, "sig"),
    (functions.multidim, "multidim")
]
# sizes_of_samples = [50, 150, 400]
# sizes_of_samples = [400]
sizes_of_samples = [400, 50]
generate_size = 1_000

if __name__ == "__main__":
    for func, func_name in funcs:
        nn_data_x = np.array(
            [
                [
                    random.uniform(1 / generate_size, 1),
                    random.uniform(1 / generate_size, 1),
                    random.uniform(1 / generate_size, 1),
                ]
                for i in range(1, generate_size + 2)
            ]
        )  # X data
        nn_data_y = np.array([[func(*x)] for x in nn_data_x])
        for size in sizes_of_samples:
            train_idx = [randint(0, generate_size) for _ in range(size)]
            train_idx.sort()
            val_idx = [randint(0, generate_size) for _ in range(size // 2)]
            val_idx.sort()
            val_data_x = nn_data_x[val_idx, :]  # validation X data
            val_data_y = nn_data_y[val_idx, :]  # validation Y data
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
