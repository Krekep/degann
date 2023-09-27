from random import randint

import numpy as np

from degann import (
    random_search_endless,
    LH_ODE_1_solution,
    LH_ODE_2_solution,
    LF_ODE_1_solution,
    build_plot,
    IModel,
    LH_ODE_2_solution,
    NLF_ODE_1_solution,
    encode,
)

nn_data_x = np.array([[i / 1000] for i in range(100, 1_001)])  # X data
true_x = np.array([[i / 20] for i in range(2, 20)])
true_y = np.array([NLF_ODE_1_solution(x) for x in true_x])
nn_data_y = np.array([NLF_ODE_1_solution(x) for x in nn_data_x])
train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(40)]
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
t = [10, 5, 3, 1]
for tt in t:
    loss, epoch, _, _, net, iterations = random_search_endless(
        1, 1, (data_x, data_y), opt_n, loss_f, tt, verbose=1
    )
    nn = IModel(1, [], 1)
    nn.from_dict(net)
    code = encode(nn, offset=8)
    print(f"{tt} loss threshold. {iterations} iterations. {loss} Loss, {epoch} epoch")
    # 90 loss threshold. 88 iterations. 73.68174743652344 Loss, 632 epoch
    build_plot(
        nn,
        (0.1, 1.0),
        0.01,
        true_data=(true_x, true_y),
        labels=[code, func],
        title=f"{tt} loss threshold. {iterations} iterations",
    )

"""
90 loss threshold. 9 iterations. 89.02749633789062 Loss, 460 epoch
80 loss threshold. 34 iterations. 51.840110778808594 Loss, 557 epoch
70 loss threshold. 18 iterations. 47.40713119506836 Loss, 698 epoch
"""
