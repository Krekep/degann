from random import randint

import numpy as np

from degann import (
    random_search_endless,
    build_plot,
    IModel,
    encode,
)
from experiments.functions import LH_ODE_1_solution

data_size = 50
nn_data_x = np.array([[i / 1000] for i in range(0, 1_001)])  # X data
true_x = np.array([[i / 20] for i in range(2, 20)])
true_y = np.array([LH_ODE_1_solution(x) for x in true_x])
nn_data_y = np.array([LH_ODE_1_solution(x) for x in nn_data_x])
train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(data_size)]
train_idx.sort()
val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(20)]
val_idx.sort()
val_data_x = nn_data_x[val_idx, :]  # validation X data
val_data_y = nn_data_y[val_idx, :]  # validation Y data
data_x = nn_data_x[train_idx, :]  # X data
data_y = nn_data_y[train_idx, :]  # Y data

loss_f = "MeanAbsolutePercentageError"
opt_n = "Adam"
func = "sin(10x)"
t = [25]
for tt in t:
    # loss, epoch, _, _, net, iterations = simulated_annealing(
    #     1, 1, (data_x, data_y), k_max=1000, opt=opt_n, loss=loss_f, threshold=tt
    # )
    # iterations += 1
    loss, epoch, _, _, net, iterations = random_search_endless(
        1, 1, (data_x, data_y), opt_n, loss_f, tt, verbose=1
    )
    nn = IModel(1, [], 1)
    nn.from_dict(net)
    code = encode(nn, offset=8)
    print(
        f"Random search. {tt} loss threshold. {iterations} iterations. {loss} Loss, {epoch} epoch"
    )
    build_plot(
        nn,
        (0.1, 1.0),
        0.01,
        true_data=(true_x, true_y),
        labels=[code, func],
        title=f"Random search. {tt} loss threshold. {iterations} iterations",
    )
