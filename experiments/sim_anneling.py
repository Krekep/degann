from itertools import product

import numpy as np

from degann import (
    generate_neighbor,
    MeasureTrainTime,
)
from degann.networks.nn_code import (
    alph_n_full,
    alph_a,
    alph_n_div3,
    alph_n_div2,
)
from degann.networks.expert import (
    simulated_annealing,
    distance_lin,
    temperature_exp,
    distance_const,
    temperature_lin,
)
import gen_dataset

opt = "Adam"

losses = {
    "MeanAbsolutePercentageError": [50, 25, 10],
    # "MeanAbsolutePercentageError": [10],
    # "MaxAbsolutePercentageError": [50, 25, 10],
    # "MaxAbsoluteDeviation": [3, 1, 0.1],
    # "RootMeanSquaredError": [3, 1, 0.1]
}
distances = [
    (distance_const(300), "dc[300]"),
    (distance_const(400), "dc[400]"),
    (distance_lin(50, 400), "dl[50][400]"),
]
temperatures = [temperature_lin, temperature_exp]
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
        for dist_m, dist_name in distances:
            for temp_m in temperatures:
                for loss in losses:
                    for threshold in losses[loss]:
                        for iter in range(1, 21):
                            alg_name = f"Ann,thres{threshold},{dist_name},{temp_m.__name__[-3:]}"
                            file_name = f"results/{alg_name}_{func_name}_{iter}"

                            print(
                                dist_name,
                                temp_m.__name__[-3:],
                                loss,
                                threshold,
                                func_name,
                                size,
                                iter,
                            )

                            (
                                nn_loss,
                                nn_epoch,
                                loss_f,
                                opt_n,
                                net,
                                _,
                            ) = simulated_annealing(
                                1,
                                1,
                                data=(train_data_x, train_data_y),
                                val_data=(val_data_x, val_data_y),
                                k_max=max_iter,
                                loss=loss,
                                threshold=threshold,
                                distance_method=dist_m,
                                method=generate_neighbor,
                                temperature_method=temp_m,
                                update_gen_cycle=10,
                                callbacks=[time_viewer],
                                logging=True,
                                file_name=file_name,
                            )
