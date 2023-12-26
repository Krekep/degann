from itertools import product

import numpy as np

from degann import MeasureTrainTime
from degann.networks.nn_code import (
    alph_n_full,
    alph_a,
    alph_n_div3,
    alph_n_div2,
)
from degann.networks.expert import random_search_endless
import gen_dataset

opt = "Adam"
num_epoch = 250

losses = {
    "MeanAbsolutePercentageError": [50, 25, 10],
    "MaxAbsolutePercentageError": [50, 25, 10],
    "MaxAbsoluteDeviation": [3, 1, 0.1],
    "RootMeanSquaredError": [3, 1, 0.1],
}
max_iter = 1000
time_viewer = MeasureTrainTime()

for func, func_name in gen_dataset.funcs:
    for size in gen_dataset.sizes_of_samples:
        train_data_x, train_data_y = np.genfromtxt(
            f"data/{func_name}_{size}_train.csv", delimiter=",", unpack=True
        )
        val_data_x, val_data_y = np.genfromtxt(
            f"data/{func_name}_{size}_validate.csv", delimiter=",", unpack=True
        )

        train_data_x, train_data_y = train_data_x.reshape(-1, 1), train_data_y.reshape(
            -1, 1
        )
        val_data_x, val_data_y = val_data_x.reshape(-1, 1), val_data_y.reshape(-1, 1)

        for loss in losses:
            for threshold in losses[loss]:
                alg_name = f"Random{threshold},"
                file_name = f"results/{alg_name}_{func_name}"
                for iter in range(1, 21):
                    iter_file_name = f"{file_name}_{iter}"
                    print(loss, threshold, func_name, size, iter)

                    (
                        nn_loss,
                        nn_epoch,
                        loss_f,
                        opt_n,
                        net,
                        iter_count,
                    ) = random_search_endless(
                        1,
                        1,
                        data=(train_data_x, train_data_y),
                        opt=opt,
                        loss=loss,
                        threshold=threshold,
                        val_data=(val_data_x, val_data_y),
                        max_iter=max_iter,
                        callbacks=[time_viewer],
                        logging=True,
                        file_name=iter_file_name,
                    )
