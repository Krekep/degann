from datetime import datetime
from random import randint

import numpy as np

from degann.search_algorithms import grid_search
from degann.networks.topology.DenseNet.parameter_space import DenseNetParameterSpace

from experiments.functions import LH_ODE_1_solution

#
# Prepare data for training. Equation is `sin(10 * x)`
#
num_epoch = 500
data_size = 40
file_name = f"LH_ODE_1_solution"
nn_data_x = np.array([[i / 1000] for i in range(0, 1_001)])  # X data
nn_data_y = np.array([LH_ODE_1_solution(x) for x in nn_data_x])
train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(data_size)]
train_idx.sort()
val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(30)]
val_idx.sort()
val_data_x = nn_data_x[val_idx, :]  # validation X data
val_data_y = nn_data_y[val_idx, :]  # validation Y data
nn_data_x = nn_data_x[train_idx, :]  # X data
nn_data_y = nn_data_y[train_idx, :]  # Y data

#
# To complete the work faster, we will not go through all the variants,
# but truncated ones by three
#
layer_sizes = [8, 11, 14, 17, 20]
activation_funcs = [
    "linear",
    "relu",
    "tanh",
    "sigmoid",
]
print(file_name)
print(f"Layer sizes: {len(layer_sizes)}, Activations: {len(activation_funcs)}")

opt = "Adam"  # optimizer
loss = "MaxAbsoluteDeviation"  # loss function

#
# Start full search over specified parameters
#
params = DenseNetParameterSpace(
    input_size=1,
    output_size=1,
    optimizers=[opt],
    losses=[loss],
    layer_sizes=layer_sizes,
    activation_funcs=activation_funcs,
    min_epoch=num_epoch,
    max_epoch=num_epoch,
    epoch_step=10,
    nn_min_depth=1,
    nn_max_depth=4,
)

grid_search(
    data=(nn_data_x, nn_data_y),
    params=params,
    val_data=(val_data_x, val_data_y),
    logging=True,
    file_name="full_search_example",
    verbose=True,
)
print("END 1, 4", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
