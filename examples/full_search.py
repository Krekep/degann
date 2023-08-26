from datetime import datetime
from itertools import product
from random import randint

import numpy as np

from degann import (
    LF_ODE_1_solution,
    MeasureTrainTime,
    LH_ODE_1_solution,
    LF_ODE_3_solution,
    LH_ODE_2_solution,
    NLF_ODE_1_solution,
    NLF_ODE_2_solution,
)
from degann.networks.nn_code import alph_n_full, alph_a, alph_n_div3, alph_n_div2
from degann.networks.expert import full_search, full_search_step

num_epoch = 200
data_size = 40
file_name = f"LH_ODE_1_solution"
nn_data_x = np.array([[i / 1000] for i in range(0, 1_001)])  # X data
nn_data_y = np.array([LH_ODE_1_solution(x) for x in nn_data_x])
train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(data_size)]
train_idx.sort()
val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(20)]
val_idx.sort()
val_data_x = nn_data_x[val_idx, :]  # validation X data
val_data_y = nn_data_y[val_idx, :]  # validation Y data
nn_data_x = nn_data_x[train_idx, :]  # X data
nn_data_y = nn_data_y[train_idx, :]  # Y data


all_variants = ["".join(elem) for elem in product(alph_n_full, alph_a)]
div3_variants = ["".join(elem) for elem in product(alph_n_div3, alph_a)]
div2_variants = ["".join(elem) for elem in product(alph_n_div2, alph_a)]
print(file_name)
print(len(all_variants), len(div2_variants), len(div3_variants))
opt = "Adam"
loss = "MeanAbsolutePercentageError"

full_search(
    (nn_data_x, nn_data_y),
    (1, 2),
    all_variants,
    (num_epoch, num_epoch, 10),
    [opt],
    [loss],
    (val_data_x, val_data_y),
    logging=True,
    file_name=file_name,
    verbose=1
)
print("END 1, 2", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
full_search(
    (nn_data_x, nn_data_y),
    (3, 3),
    div2_variants,
    (num_epoch, num_epoch, 10),
    [opt],
    [loss],
    (val_data_x, val_data_y),
    logging=True,
    file_name=file_name,
    verbose=1
)
print("END 3", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))

time_viewer = MeasureTrainTime()
for i in range(4, 11):
    print(i, datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
    for size in alph_n_full:
        for act in alph_a:
            full_search_step(
                (size + act) * i,
                num_epoch,
                opt,
                loss,
                (nn_data_x, nn_data_y),
                val_data=(val_data_x, val_data_y),
                logging=True,
                file_name=file_name,
                callbacks=[time_viewer]
            )
print("END 4, 11", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))


num_epoch = 500
data_size = 40
test_code = "e968e6"
file_name = f"run_{test_code}"
for _ in range(1000):
    full_search_step(
        test_code,
        num_epoch,
        opt,
        loss,
        (nn_data_x, nn_data_y),
        logging=True,
        file_name=file_name,
        val_data=(val_data_x, val_data_y),
        callbacks=[time_viewer]
    )
