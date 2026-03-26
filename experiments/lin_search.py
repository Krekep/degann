from datetime import datetime

import numpy as np

from degann import MeasureTrainTime
from degann.networks.topology.DenseNet.parameter_space import DenseNetParameterSpace
from degann.networks.topology.DenseNet.config import DenseNetConfig
from degann.networks.imodel import IModel
from degann.search_algorithms.grid_search import grid_search
import gen_dataset

layer_sizes_full = [8, 11, 14, 17, 20, 23]
layer_sizes_div3 = [8, 14, 20]
layer_sizes_div2_1 = [8, 11, 14]
layer_sizes_div2_2 = [17, 20, 23]
layer_sizes_div2 = [8, 11, 14, 17, 20, 23]
activation_funcs = ["linear", "relu", "tanh", "sigmoid", "swish", "elu"]

print(
    len(layer_sizes_full),
    len(layer_sizes_div2),
    len(layer_sizes_div2_1),
    len(layer_sizes_div2_2),
    len(layer_sizes_div3),
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

            params_1_2 = DenseNetParameterSpace(
                input_size=1,
                output_size=1,
                optimizers=[opt],
                losses=[loss],
                layer_sizes=layer_sizes_div2,
                activation_funcs=activation_funcs,
                min_epoch=num_epoch,
                max_epoch=num_epoch,
                epoch_step=1,
                nn_min_depth=1,
                nn_max_depth=2,
            )
            grid_search(
                data=(train_data_x, train_data_y),
                params=params_1_2,
                val_data=(val_data_x, val_data_y),
                logging=True,
                file_name=file_name,
                verbose=True,
            )
            print("END 1, 2", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))

            params_3 = DenseNetParameterSpace(
                input_size=1,
                output_size=1,
                optimizers=[opt],
                losses=[loss],
                layer_sizes=layer_sizes_div3,
                activation_funcs=activation_funcs,
                min_epoch=num_epoch,
                max_epoch=num_epoch,
                epoch_step=1,
                nn_min_depth=3,
                nn_max_depth=3,
            )
            grid_search(
                data=(train_data_x, train_data_y),
                params=params_3,
                val_data=(val_data_x, val_data_y),
                logging=True,
                file_name=file_name,
                verbose=True,
            )
            print("END 3", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))

            time_viewer = MeasureTrainTime()
            for depth in range(4, 11):
                print(depth, datetime.today().strftime("%Y-%m-%d %H:%M:%S"))

                for layer_size in layer_sizes_full:
                    for act in activation_funcs:
                        config = DenseNetConfig(
                            layer_sizes=[layer_size] * depth,
                            activation_funcs=[act] * depth + ["linear"],
                            optimizer=opt,
                            loss_func=loss,
                            input_size=1,
                            output_size=1,
                        )
                        nn = IModel(config=config, net_type="DenseNet")
                        nn.compile(optimizer=opt, loss_func=loss)
                        nn.train(
                            train_data_x,
                            train_data_y,
                            epochs=num_epoch,
                            validation_data=(val_data_x, val_data_y),
                            callbacks=[time_viewer],
                            verbose=0,
                        )
                        loss_val = nn.evaluate(val_data_x, val_data_y, verbose=0)

                        with open(f"{file_name}_results.csv", "a") as f:
                            f.write(f"{depth},{layer_size},{act},{loss_val}\n")
            print("END 4, 11", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
