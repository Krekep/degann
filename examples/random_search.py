import time
from random import randint

import numpy as np

from degann.search_algorithms.simulated_annealing import (
    simulated_annealing,
    temperature_exp,
    temperature_lin,
    distance_const,
    distance_lin,
)
from degann.search_algorithms.generate import generate_neighbor, random_generate
from experiments.functions import (
    LF_ODE_1_solution,
    LH_ODE_1_solution,
    LF_ODE_3_solution,
    LH_ODE_2_solution,
    NLF_ODE_1_solution,
    NLF_ODE_2_solution,
)

name_to_funcs = {
    "LF_ODE_1": LF_ODE_1_solution,
    "LH_ODE_1": LH_ODE_1_solution,
    "LF_ODE_3": LF_ODE_3_solution,
    "LH_ODE_2": LH_ODE_2_solution,
    "NLF_ODE_1": NLF_ODE_1_solution,
    "NLF_ODE_2": NLF_ODE_2_solution,
}
# for func_name in ["LF_ODE_1", "LH_ODE_1"]:
# for func_name in ["LF_ODE_3", "NLF_ODE_2"]:
# for func_name in ["LH_ODE_2", "NLF_ODE_1"]:
distances = [
    (distance_const(300), "dc,300,"),
    (distance_const(400), "dc,400,"),
    (distance_lin(50, 400), "dl,50_400,"),
]
for func_name in ["LH_ODE_1"]:
    nn_data_x = np.array([[i / 1000] for i in range(100, 1_001)])  # X data
    nn_data_y = np.array([name_to_funcs[func_name](x) for x in nn_data_x])
    train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(40)]
    train_idx.sort()
    val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(20)]
    val_idx.sort()
    val_data_x = nn_data_x[val_idx, :]  # validation X data
    val_data_y = nn_data_y[val_idx, :]  # validation Y data
    nn_data_x = nn_data_x[train_idx, :]  # X data
    nn_data_y = nn_data_y[train_idx, :]  # Y data

    # for loss_name in ["Huber", "MeanAbsolutePercentageError", "MaxAbsoluteDeviation"]:
    for loss_name in ["MeanAbsolutePercentageError"]:
        for max_iter in [100]:
            for dist_m, dist_name in distances:
                for neigh_m in [generate_neighbor, random_generate]:
                    for temp_m in [temperature_lin, temperature_exp]:
                        print(max_iter, dist_name, neigh_m.__name__, temp_m.__name__)
                        launches = 10
                        for i in range(launches):
                            start_t = time.perf_counter()
                            nn_loss, nn_epoch, loss_f, opt_n, net = simulated_annealing(
                                1,
                                1,
                                data=(nn_data_x, nn_data_y),
                                val_data=(val_data_x, val_data_y),
                                k_max=max_iter,
                                loss=loss_name,
                                distance_method=dist_m,
                                method=neigh_m,
                                temperature_method=temp_m,
                                update_gen_cycle=10,
                                logging=True,
                                file_name=f"{func_name}_{max_iter}_{dist_name}_{neigh_m.__name__[:5]}_{temp_m.__name__[-3:]}",
                            )
                            end_t = time.perf_counter()
                            print(
                                i, net["block_size"], nn_loss, nn_epoch, end_t - start_t
                            )
