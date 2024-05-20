import numpy as np

from degann import MeasureTrainTime

from degann.search_algorithms.random_search import random_search_endless
import gen_dataset

opt = "Adam"

losses = {
    "MeanAbsolutePercentageError": [50, 25, 10],
    "MaxAbsolutePercentageError": [50, 25, 10],
    "MaxAbsoluteDeviation": [3, 1, 0.1],
    "RootMeanSquaredError": [3, 1, 0.1],
}
max_iter = 1000
time_viewer = MeasureTrainTime()

input_size = 3

for func, func_name in gen_dataset.funcs:
    for size in gen_dataset.sizes_of_samples:
        train_data_x, train_data_y = np.genfromtxt(
            f"data/{func_name}_{size}_train.csv",
            delimiter=",",
            usecols=list(range(input_size)),
        ), np.genfromtxt(
            f"data/{func_name}_{size}_train.csv",
            delimiter=",",
            unpack=True,
            usecols=[input_size],
        )
        val_data_x, val_data_y = np.genfromtxt(
            f"data/{func_name}_{size}_validate.csv",
            delimiter=",",
            usecols=list(range(input_size)),
        ), np.genfromtxt(
            f"data/{func_name}_{size}_validate.csv",
            delimiter=",",
            unpack=True,
            usecols=[input_size],
        )

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
                        input_size,
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
