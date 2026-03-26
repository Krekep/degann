from degann.expert import ExpertSystemTags
from degann.expert.tags import (
    EquationType,
    RequiredModelPrecision,
    ModelPredictTime,
    DataSize,
)
from degann.networks.imodel import IModel
from degann.networks.topology.DenseNet.config import DenseNetConfig
from degann.networks.topology.DenseNet.parameter_space import DenseNetParameterSpace
from degann.search_algorithms import (
    simulated_annealing,
    grid_search,
)

from degann.expert.selector import suggest_parameters
from degann.expert.pipeline import execute_pipeline

from degann.equations import (
    build_plot,
    equation_solve,
    SystemODE,
    system_ode_from_string,
    str_eq_to_params,
)

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


config_1 = DenseNetConfig(
    layer_sizes=[32, 16, 8],
    activation_funcs=["tanh", "tanh", "tanh", "linear"],
    input_size=1,
    output_size=1,
)
nn_1_32_16_8_1 = IModel(config=config_1, net_type="DenseNet")
print(nn_1_32_16_8_1)

shape = [10, 10]
activations = ["swish", "relu", "linear"]

config_2 = DenseNetConfig(
    layer_sizes=shape,
    activation_funcs=activations,
    input_size=1,
    output_size=1,
)
nn_1_10_10_3 = IModel(config=config_2, net_type="DenseNet")

print("Activation functions per layer for nn_1_10_10_3")
acts = nn_1_10_10_3.get_activations
for i, act_name in enumerate(acts):
    print(i, act_name)

print(nn_1_10_10_3)

# Prepare network for training

nn_1_32_16_8_1.compile(
    optimizer="Adam",
    loss_func="MaxAbsoluteDeviation",  # max(abs(y_true - y_prediction))
)

# Train network (something about 5 sec. on Google Colab)

loss_before_train = nn_1_32_16_8_1.evaluate(x_data, y_data, verbose=0)

nn_1_32_16_8_1.train(
    x_data=train_data_x,
    y_data=train_data_y,
    epochs=50,
    verbose=0,
)

loss_after_train = nn_1_32_16_8_1.evaluate(x_data, y_data, verbose=0)

print(f"Loss before training = {loss_before_train}")
print(f"Loss after training = {loss_after_train}")

nn_1_32_16_8_1.export_to_file("some_path")
nn_1_32_16_8_1.export_to_cpp("some_path")

# Grid Search Example

params = DenseNetParameterSpace(
    input_size=1,
    output_size=1,
    optimizers=["Adam"],
    losses=["MeanSquaredError"],
    layer_sizes=[10, 5],
    activation_funcs=["parabolic", "exponential"],
    min_epoch=10,
    max_epoch=10,
    epoch_step=1,
    nn_min_depth=0,
    nn_max_depth=2,
)

best_loss, best_epoch, best_loss_func, best_opt, best_net = grid_search(
    data=(train_data_x, train_data_y),
    params=params,
    val_data=(x_data, y_data),
    logging=False,
    file_name="grid_search_example",
    verbose=True,
)
print(f"Best loss: {best_loss}, Best epoch: {best_epoch}")


# Simulated Annealing Example

sa_params = DenseNetParameterSpace(
    input_size=1,
    output_size=1,
    optimizers=["Adam"],
    losses=["Huber"],
    layer_sizes=[10, 16, 23],
    activation_funcs=["relu", "tanh", "swish"],
    min_epoch=10,
    max_epoch=20,
    nn_min_depth=1,
    nn_max_depth=3,
)

(
    result_loss,
    result_epoch,
    result_loss_func,
    result_opt,
    result_net,
    iterations,
) = simulated_annealing(
    data=(train_data_x, train_data_y),
    params=sa_params,
    val_data=(x_data, y_data),
    max_iter=10,
    threshold=1,
    verbose=True,
)
print(f"SA Result: loss={result_loss}, epochs={result_epoch}, iterations={iterations}")

# Expert System Example

selector_tags = ExpertSystemTags()
selector_tags.equation_type = (
    EquationType.EXP
)  # type of function in data --- in this case it is the parabola (3 * x^2)
selector_tags.model_precision = (
    RequiredModelPrecision.MINIMAL
)  # Shows how important the accuracy of the solution is to us
selector_tags.predict_time = (
    ModelPredictTime.MEDIUM
)  # Shows how important the operating time (predict) of the resulting neural network is to us
selector_tags.data_size = DataSize.MEDIAN  # Training dataset size
algorithms_parameters = suggest_parameters(tags=selector_tags)

print("Resulting parameters by expert system for search algorithms")
for k, v in algorithms_parameters.__dict__.items():
    print(f"{k}: {v}")

# Execute Pipeline

expert_params = DenseNetParameterSpace(
    input_size=1,
    output_size=1,
    optimizers=[algorithms_parameters.optimizer],
    losses=[algorithms_parameters.loss_function],
    layer_sizes=[8, 16, 32],
    activation_funcs=["relu", "tanh", "linear"],
    min_epoch=algorithms_parameters.min_train_epoch,
    max_epoch=algorithms_parameters.max_train_epoch,
    nn_min_depth=1,
    nn_max_depth=4,
)

(
    result_loss,
    result_epoch,
    result_loss_func,
    result_optimizer,
    result_nn,
) = execute_pipeline(
    data=(train_data_x, train_data_y),
    params=expert_params,
    parameters={
        "launch_count_random_search": algorithms_parameters.launch_count_random_search,
        "launch_count_simulated_annealing": algorithms_parameters.launch_count_simulated_annealing,
        "iteration_count": algorithms_parameters.iteration_count,
        "loss_threshold": algorithms_parameters.metric_threshold,
    },
    val_data=(x_data, y_data),
    run_grid_search=False,
)
print("Resulting loss value =", result_loss)
print("Resulting network:")
print(result_nn)

# Restore model from dict and plot

model_from_expert_system_config = DenseNetConfig(
    layer_sizes=result_nn["layer_sizes"],
    activation_funcs=result_nn["activation_funcs"],
    input_size=result_nn["input_size"],
    output_size=result_nn["output_size"],
)
model_from_expert_system = IModel(
    config=model_from_expert_system_config,
    net_type=result_nn["net_type"],
)
model_from_expert_system.from_dict(result_nn)  # restore model from dict

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
    interval=(0.0, 1.0),  # from where to where to build a plot
    step=0.02,  # step inside bounds
    title="Approximation of parabola",
    labels=[
        "expert nn",
        "[32_16_8] nn",
        "f(x) = 3*x^2",
    ],  # labels for plots. Last label for true values
    true_data=(x_plot_data, y_plot_data),
)

str_sode = "y1*y2 y0(0)=0\n" + "-y0*y2 y1(0)=1\n" + "-0.5*y0*y1 y2(0)=1"
parsed = system_ode_from_string(str_sode)  # transform to list of strings
sode = SystemODE()
sode.prepare_equations(len(parsed), parsed)  # build functions for each equation
sode.solve((0, 3), 10)  # solve SODE on the interval
table = sode.build_table()
x_ode_data = table[:, :1]
y_ode_data = table[:, 1:]

for feature, value in zip(x_ode_data, y_ode_data):
    print(feature, value)

function = "3*x+2*y+4*z"
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
