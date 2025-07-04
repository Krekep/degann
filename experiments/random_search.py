import numpy as np

if __name__ == "__main__":
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if len(gpus) > 0:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
from degann.networks.callbacks import MeasureTrainTime

from degann.search_algorithms.random_search import random_search_endless
from degann.search_algorithms.search_algorithms_parameters import (
    BaseSearchParameters,
    RandomEarlyStoppingSearchParameters,
)
from degann.search_algorithms.nn_code import default_alphabet
import gen_dataset

opt = "Adam"

losses = {
    "RelativeL1Loss": [0.25, 0.1, 0.01],
}
max_iter = 1000
time_viewer = MeasureTrainTime()

input_size = 1

for func, func_name in gen_dataset.funcs:
    for size in gen_dataset.sizes_of_samples:
        train_data_x, train_data_y = np.genfromtxt(
            f"./data/{func_name}_{size}_train.csv",
            delimiter=",",
            usecols=list(range(input_size)),
        ), np.genfromtxt(
            f"./data/{func_name}_{size}_train.csv",
            delimiter=",",
            unpack=True,
            usecols=[input_size],
        )
        val_data_x, val_data_y = np.genfromtxt(
            f"./data/{func_name}_{size}_validate.csv",
            delimiter=",",
            usecols=list(range(input_size)),
        ), np.genfromtxt(
            f"./data/{func_name}_{size}_validate.csv",
            delimiter=",",
            unpack=True,
            usecols=[input_size],
        )
        train_data_x = np.reshape(train_data_x, newshape=[size] * input_size + [1])
        train_data_y = np.reshape(train_data_y, newshape=[size, 1])
        val_data_x = np.reshape(
            val_data_x, newshape=[gen_dataset.generate_size + 1] * input_size + [1]
        )
        val_data_y = np.reshape(val_data_y, newshape=[gen_dataset.generate_size + 1, 1])
        train_data_x = tf.convert_to_tensor(train_data_x, dtype=tf.float32)
        train_data_y = tf.convert_to_tensor(train_data_y, dtype=tf.float32)
        val_data_x = tf.convert_to_tensor(val_data_x, dtype=tf.float32)
        val_data_y = tf.convert_to_tensor(val_data_y, dtype=tf.float32)

        for loss in losses:
            for threshold in losses[loss]:
                alg_name = f"Random{threshold},"
                file_name = f"results/{alg_name}_{func_name}"
                for iter in range(1, 21):
                    iter_file_name = f"{file_name}_{iter}"
                    print(loss, threshold, func_name, size, iter)

                    search_alg_params = BaseSearchParameters()
                    search_alg_params.input_size = input_size
                    search_alg_params.output_size = 1
                    search_alg_params.data = (train_data_x, train_data_y)
                    search_alg_params.val_data = (val_data_x, val_data_y)

                    random_search_parameters = RandomEarlyStoppingSearchParameters(
                        search_alg_params
                    )
                    random_search_parameters.optimizer = opt
                    random_search_parameters.loss_function = loss
                    random_search_parameters.min_epoch = 100
                    random_search_parameters.max_epoch = 700
                    random_search_parameters.metric_threshold = threshold
                    random_search_parameters.nn_min_length = 1
                    random_search_parameters.nn_max_length = 6
                    random_search_parameters.nn_alphabet = default_alphabet
                    random_search_parameters.iterations = 1
                    random_search_parameters.max_launches = max_iter
                    random_search_parameters.logging = True
                    random_search_parameters.file_name = iter_file_name
                    random_search_parameters.eval_metric = "RelativeL1Loss"
                    random_search_parameters.callbacks = [MeasureTrainTime()]

                    (
                        nn_loss,
                        nn_epoch,
                        loss_f,
                        opt_n,
                        net,
                        iter_count,
                    ) = random_search_endless(random_search_parameters)
