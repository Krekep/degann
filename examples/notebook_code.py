from degann.networks.imodel import IModel
from degann.networks.topology.DenseNet.parameter_space import DenseNetParameterSpace
from degann.networks.topology.DenseNet.config import DenseNetConfig
from degann.search_algorithms import simulated_annealing, pattern_search, grid_search
from degann.search_algorithms.random_search import random_search_threshold
from degann.expert.tags import ExpertSystemTags
from degann.expert.tags import (
    EquationType,
    ModelPredictTime,
    DataSize,
    RequiredModelPrecision,
)
from degann.expert.selector import suggest_parameters
from degann.expert.pipeline import execute_pipeline
from degann.equations import build_plot
from degann.equations import equation_solve, str_eq_to_params

import numpy as np
from random import randint


# Prepare data for neural network training
def f_3x2(x):
    return 3 * x**2


data_size = 1000
x_data = np.array([[i / data_size] for i in range(0, data_size + 1)])
y_data = np.array([f_3x2(x) for x in x_data])

train_data_size = 200
train_idx = [randint(0, len(x_data) - 1) for _ in range(train_data_size)]
train_idx.sort()
train_data_x = x_data[train_idx, :]  # X data
train_data_y = y_data[train_idx, :]  # Y data

conf = DenseNetConfig(
    input_size=1,
    output_size=1,
    layer_sizes=[32, 16, 8],
    activation_funcs=["tanh", "tanh", "tanh"],
    optimizer="Adam",
    loss_func="MeanSquaredError",
)
nn_1_32_16_8_1 = IModel(conf, "DenseNet")
print(nn_1_32_16_8_1)

conf = DenseNetConfig(
    input_size=1,
    output_size=1,
    layer_sizes=[10, 10],
    activation_funcs=["swish", "relu"],
    optimizer="Adam",
    loss_func="MeanSquaredError",
)
nn_1_10_10_3 = IModel(conf, "DenseNet")

print("Activation functions per layer for n_1_10_10_3")
acts = nn_1_10_10_3.get_activations
for i, act_name in enumerate(["swish", "relu"]):
    print(i, act_name)
print(nn_1_10_10_3)

# Prepare network for training
nn_1_32_16_8_1.compile(
    optimizer="Adam",
    loss_func="MaxAbsoluteDeviation",  # max(abs(y_true - y_prediction))
)

# Train network (something about 5 sec. on Google Colab)
loss_before_train = nn_1_32_16_8_1.evaluate(x_data, y_data, verbose=0)
nn_1_32_16_8_1.train(train_data_x, train_data_y, epochs=50, verbose=0)
loss_after_train = nn_1_32_16_8_1.evaluate(x_data, y_data, verbose=0)

print(f"Loss before training = {loss_before_train}")
print(f"Loss after training = {loss_after_train}")

nn_1_32_16_8_1.export_to_file("some_path")
nn_1_32_16_8_1.export_to_cpp("some_path")

# Pattern Search Example
config = {
    "loss_functions": ["MeanSquaredError"],
    "optimizers": ["Adam"],
    "metrics": ["MaxAbsoluteDeviation"],
    "net_shapes": [[], [5, 5]],  # neural network without hidden layers
    "activations": ["parabolic", "exponential"],
    "validation_split": 0,
    "rates": [1e-2],
    "epochs": [10],
    "normalize": [False],
    "use_rand_net": False,
}
best_nns = pattern_search(
    x_data=train_data_x, y_data=train_data_y, x_val=x_data, y_val=y_data, **config
)
print(best_nns)

# Grid Search Example
grid_params = DenseNetParameterSpace(
    input_size=1,
    output_size=1,
    optimizers=["Adam"],
    losses=["MeanSquaredError"],
    epochs=[10, 20],
    nn_min_depth=1,
    nn_max_depth=2,
    layer_sizes=[8, 12],
    activation_funcs=["softplus", "gelu"],
)

result_loss, result_epoch, result_loss_name, result_optimizer, result_nn = grid_search(
    data=(train_data_x, train_data_y), params=grid_params, verbose=True
)

print(f"Loss: {result_loss}")
print(f"Layers: {result_nn['config']['layer_sizes']}")
print(f"Activations: {result_nn['config']['activation_funcs'][:-1]}")
print(f"Optimizer: {result_optimizer}")
print(f"Loss function: {result_loss_name}")

# Random Search Example
random_params = DenseNetParameterSpace(
    input_size=1,
    output_size=1,
    optimizers=["Adam"],
    losses=["MeanSquaredError"],
    epochs=[i for i in range(10, 21)],
    nn_min_depth=1,
    nn_max_depth=3,
    layer_sizes=[8, 23, 12],
    activation_funcs=["softplus", "gelu", "swish"],
)

(
    result_loss,
    result_epoch,
    result_loss_name,
    result_optimizer,
    result_nn,
) = random_search_threshold(
    data=(train_data_x, train_data_y),
    params=random_params,
    max_iter=10,
    threshold=0.01,
    verbose=True,
)

print(f"Loss: {result_loss}")
print(f"Layers: {result_nn['config']['layer_sizes']}")
print(f"Activations: {result_nn['config']['activation_funcs'][:-1]}")
print(f"Optimizer: {result_optimizer}")
print(f"Loss function: {result_loss_name}")

# Simulated Annealing Example
SA_params = DenseNetParameterSpace(
    input_size=1,
    output_size=1,
    optimizers=["Adam"],
    losses=["Huber"],
    epochs=[i for i in range(10, 21)],
    nn_min_depth=1,
    nn_max_depth=3,
    layer_sizes=[8, 23, 12],
    activation_funcs=["softplus", "gelu", "swish"],
)

(
    result_loss,
    result_epoch,
    result_loss_name,
    result_optimizer,
    result_nn,
    final_iteration,
) = simulated_annealing(
    data=(train_data_x, train_data_y),
    params=SA_params,
    max_iter=10,
    threshold=0.01,
    verbose=True,
)

print(f"Loss: {result_loss}")
print(f"Layers: {result_nn['config']['layer_sizes']}")
print(f"Activations: {result_nn['config']['activation_funcs'][:-1]}")
print(f"Optimizer: {result_optimizer}")
print(f"Loss function: {result_loss_name}")

# Expert System Example
tags = ExpertSystemTags()
tags.equation_type = EquationType.EXP
tags.model_precision = RequiredModelPrecision.MINIMAL
tags.predict_time = ModelPredictTime.MEDIUM
tags.data_size = DataSize.MEDIAN

meta, space = suggest_parameters(tags=tags)

print("Resulting parameters by expert system for search algorithms")
print(meta)
print(space)

epochs = list(range(space.min_epoch, space.max_epoch + 1, space.epoch_step))
expert_params = DenseNetParameterSpace(
    input_size=1,
    output_size=1,
    optimizers=["Adam"],
    losses=["MeanSquaredError"],
    layer_sizes=space.layer_sizes,
    activation_funcs=["tanh", "relu"],
    epochs=epochs,
    nn_min_depth=space.nn_min_depth,
    nn_max_depth=space.nn_max_depth,
)

(
    result_loss,
    result_epoch,
    result_loss_name,
    result_optimizer,
    result_nn,
) = execute_pipeline(
    data=(train_data_x, train_data_y),
    params=expert_params,
    config=meta,
    run_grid_search=False,
)

print(f"Loss: {result_loss}")
print(f"Layers: {result_nn['config']['layer_sizes']}")
print(f"Activations: {result_nn['config']['activation_funcs'][:-1]}")
print(f"Optimizer: {result_optimizer}")
print(f"Loss function: {result_loss_name}")

# Building plots
model_from_expert_system = IModel.from_dict(result_nn)  # restore model from dict

indices = []
for _ in range(30):
    indices.append(randint(0, len(x_data) - 1))
indices.sort()
x_plot_data = x_data[indices, :]
y_plot_data = y_data[indices, :]

build_plot(
    network=[
        model_from_expert_system,
        nn_1_32_16_8_1,
    ],  # list of models (or single model)
    interval=(0, 1),  # from where to where to build a plot
    step=0.1,  # step inside bounds
    title="Approximation of parabola",
    labels=[
        "expert nn",
        "[32_16_8] nn",
        "f(x) = 3 * x^2",
    ],  # labels for plots. Last label for true values
    true_data=[x_plot_data, y_plot_data],
)

# Building dataset tables for system of ODE
from degann.equations import SystemODE, system_ode_from_string

str_sode = "y1 * y2 y0(0)=0\n" + "-y0 * y2 y1(0)=1\n" + "-0.5 * y0 * y1 y2(0)=1"
parsed = system_ode_from_string(str_sode)  # transform to list of strings
sode = SystemODE()
sode.prepare_equations(len(parsed), parsed)  # build functions for each equation
sode.solve((0, 3), 10)  # solve SODE on the interval
table = sode.build_table()
x_ode_data = table[:, :1]
y_ode_data = table[:, 1:]

for feature, value in zip(x_ode_data, y_ode_data):
    print(feature, value)

# Building dataset tables for functions
function = "3 * x + 2 * y + 4 * z"
bounds = {
    "x": "0, 2, 1",  # x from 0 to 2 with step = 1
    "y": "0, 2, 2",  # y from 0 to 2 with step = 2
    "z": "0, 0, 1",  # z = 0
}  # 6 points

variables = str_eq_to_params(bounds)
eq_table = equation_solve(function, variables)

x_eq_data = eq_table[:, :-1]
y_eq_data = eq_table[:, -1:]

for feature, value in zip(x_eq_data, y_eq_data):
    print(feature, value)
