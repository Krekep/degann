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
# Specify the number of samples to randomly select for the training subset
data_size = 40
# Set the base identifier for experiment logging
file_name = "LH_ODE_1_solution"

# Create a dense input grid from 0.0 to 1.0 with a step of 0.001
nn_data_x = np.array([[i / 1000] for i in range(1_001)])
# Evaluate the exact ODE solution for each input point to form target values
nn_data_y = np.array([LH_ODE_1_solution(x) for x in nn_data_x])

# Generate random indices for the training subset
train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(data_size)]
train_idx.sort()
# Generate random indices for the validation subset
val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(30)]
val_idx.sort()

# Slice the full dataset to obtain the validation features and targets
val_data_x = nn_data_x[val_idx, :]
val_data_y = nn_data_y[val_idx, :]
# Slice the full dataset to obtain the training features and targets
nn_data_x = nn_data_x[train_idx, :]
nn_data_y = nn_data_y[train_idx, :]

# Print the experiment name for console tracking
print(file_name)
# Define the search space for layer widths and output the cardinality of each space
layer_sizes = [8, 11, 14, 17, 20]
activation_funcs = ["linear", "relu", "tanh", "sigmoid"]
print(f"Layer sizes: {len(layer_sizes)}, Activations: {len(activation_funcs)}")

# Fix the optimizer to be used during model compilation
opt = "Adam"
# Fix the loss function to be minimized during training
loss = "MaxAbsoluteDeviation"

# Instantiate the parameter space according to the library API (expects a list of epochs)
params = DenseNetParameterSpace(
    input_size=1,
    output_size=1,
    optimizers=[opt],
    losses=[loss],
    layer_sizes=layer_sizes,
    activation_funcs=activation_funcs,
    epochs=[num_epoch],
    nn_min_depth=1,
    nn_max_depth=4,
)

# Execute the grid search algorithm over the defined parameter space
grid_search(
    data=(nn_data_x, nn_data_y),
    params=params,
    val_data=(val_data_x, val_data_y),
    logging=True,
    file_name="full_search_example",
    verbose=True,
)
# Output the completion timestamp for the first search phase
print("END 1, 4", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
