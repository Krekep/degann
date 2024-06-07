from datetime import datetime
from itertools import product

import numpy as np

from degann import (
    MeasureTrainTime,
)
from degann.search_algorithms.nn_code import (
    alph_n_full,
    alphabet_activations_cut,
    alph_n_div3,
    alph_n_div2,
)
from degann.search_algorithms.grid_search import grid_search_step, grid_search
import gen_dataset

all_variants = [
    "".join(elem) for elem in product(alph_n_full, alphabet_activations_cut)
]
div3_variants = [
    "".join(elem) for elem in product(alph_n_div3, alphabet_activations_cut)
]
alph_n_div2_1 = alph_n_div2[: len(alph_n_div2) // 2]
alph_n_div2_2 = alph_n_div2[len(alph_n_div2) // 2 :]
div2_variants_1 = [
    "".join(elem) for elem in product(alph_n_div2_1, alphabet_activations_cut)
]
div2_variants_2 = [
    "".join(elem) for elem in product(alph_n_div2_2, alphabet_activations_cut)
]
div2_variants = [
    "".join(elem) for elem in product(alph_n_div2, alphabet_activations_cut)
]
print(
    len(all_variants),
    len(div2_variants),
    len(div2_variants_1),
    len(div2_variants_2),
    len(div3_variants),
)
opt = "Adam"
num_epoch = 250

losses = [
    "MeanAbsolutePercentageError",
    "MaxAbsolutePercentageError",
    "MaxAbsoluteDeviation",
    "RootMeanSquaredError",
]

for func, func_name in gen_dataset.funcs:
    file_name = f"results/LinearSearch_{func_name}"
    for loss in losses:
        for size in gen_dataset.sizes_of_samples:
            train_data_x, train_data_y = np.genfromtxt(
                f"data/{func_name}_{size}_train.csv", delimiter=",", unpack=True
            )
            val_data_x, val_data_y = np.genfromtxt(
                f"data/{func_name}_{size}_validate.csv", delimiter=",", unpack=True
            )

            train_data_x, train_data_y = train_data_x.reshape(
                -1, 1
            ), train_data_y.reshape(-1, 1)
            val_data_x, val_data_y = val_data_x.reshape(-1, 1), val_data_y.reshape(
                -1, 1
            )

            grid_search(
                1,
                1,
                (train_data_x, train_data_y),
                [opt],
                [loss],
                min_epoch=num_epoch,
                max_epoch=num_epoch,
                nn_min_length=1,
                nn_max_length=2,
                nn_alphabet=div2_variants,
                val_data=(val_data_x, val_data_y),
                logging=True,
                file_name=file_name,
                verbose=1,
            )
            print("END 1, 2", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
            grid_search(
                1,
                1,
                (train_data_x, train_data_y),
                [opt],
                [loss],
                min_epoch=num_epoch,
                max_epoch=num_epoch,
                nn_min_length=3,
                nn_max_length=3,
                nn_alphabet=div3_variants,
                val_data=(val_data_x, val_data_y),
                logging=True,
                file_name=file_name,
                verbose=1,
            )
            print("END 3", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))

            time_viewer = MeasureTrainTime()
            for i in range(4, 11):
                print(i, datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
                for size in alph_n_full:
                    for act in alphabet_activations_cut:
                        grid_search_step(
                            1,
                            1,
                            (size + act) * i,
                            num_epoch,
                            opt,
                            loss,
                            (train_data_x, train_data_y),
                            val_data=(val_data_x, val_data_y),
                            logging=True,
                            file_name=file_name,
                            callbacks=[time_viewer],
                        )
            print("END 4, 11", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
