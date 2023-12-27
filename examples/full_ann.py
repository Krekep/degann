from itertools import product
from random import randint

import numpy as np

from degann import (
    full_search_step,
    alph_n_full,
    alph_a,
    IModel,
    simulated_annealing,
    temperature_exp,
    generate_neighbor,
    distance_const,
    encode,
    random_generate,
    distance_lin,
)
from experiments.functions import LH_ODE_1_solution

code = "e8e6e6"

num_epoch = 500
data_size = 40
nn_data_x = np.array([[i / 1000] for i in range(0, 1_001)])  # X data
nn_data_y = np.array([LH_ODE_1_solution(x) for x in nn_data_x])
train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(data_size)]
train_idx.sort()
val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(40)]
val_idx.sort()
val_data_x = nn_data_x[val_idx, :]  # validation X data
val_data_y = nn_data_y[val_idx, :]  # validation Y data
nn_data_x = nn_data_x[train_idx, :]  # X data
nn_data_y = nn_data_y[train_idx, :]  # Y data

opt = "Adam"
loss = "MaxAbsoluteDeviation"
best_l = 1e6
best_vl = 1e6
best_nn = dict()
f = open("full_ann.txt", "a")
for i in range(10):
    for _ in range(50):
        curr_l, curr_val_l, curr_nn = full_search_step(
            code,
            num_epoch,
            opt,
            loss,
            (nn_data_x, nn_data_y),
            # logging=True,
            # file_name=file_name,
            val_data=(val_data_x, val_data_y),
        )
        if curr_l < best_l:
            best_l = curr_l
            best_vl = curr_val_l
            best_nn = curr_nn

    nn = IModel(1, [], 1)
    nn.from_dict(best_nn)
    nn.compile(optimizer=opt, loss_func=loss)
    val_l = nn.evaluate(val_data_x, val_data_y, verbose=0, return_dict=True)["loss"]
    print(best_vl, val_l)

    nn_loss, nn_epoch, loss_f, opt_n, net = simulated_annealing(
        1,
        1,
        data=(nn_data_x, nn_data_y),
        val_data=(val_data_x, val_data_y),
        k_max=100,
        loss=loss,
        distance_method=distance_lin(30, 100),
        method=generate_neighbor,
        temperature_method=temperature_exp,
        logging=True,
        file_name=f"LH_ODE_1_{code}",
    )

    nn.from_dict(net)
    nn.compile(optimizer=opt, loss_func=loss)
    val_l = nn.evaluate(val_data_x, val_data_y, verbose=0, return_dict=True)["loss"]
    print(nn_loss, best_l, "|", val_l, best_vl)
    print(f"**{i}**")
    nn_code = encode(nn)
    f.write(f"{nn_code},{val_l},{nn_loss},{nn_epoch},{loss_f},{opt_n}\n")
