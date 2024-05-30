import time

import numpy as np

import degann.search_algorithms.pattern_search
from degann.networks import losses, callbacks
from degann.networks import IModel
from experiments.functions import LF_ODE_1_solution, ST_S_ODE_3_table
from tests.utils import init_params

#
# Get not random initializers
#
weight_initializer1, bias_initializer = init_params(
    weight_name="ones", bias_name="zeros"
)

#
# Create simple neural network with 1 input neuron, 1 output neuron and one hidden layer with 1 neuron
#
nn = IModel.create_neuron(
    1, 1, [1], weight=weight_initializer1, biases=bias_initializer
)

# print description for nn
print(nn)
for i, layer in enumerate(nn.network.blocks):
    print(f"Layer {i} have weights {layer.w.numpy()} and biases {layer.b.numpy()}")
print(
    f"Last layer have weights {nn.network.out_layer.w.numpy()} and biases {nn.network.out_layer.b.numpy()}"
)

# print activation function for layers
acts = nn.get_activations
acts_name = []
for i in range(len(acts)):
    acts_name.append(acts[i])
print(acts_name)

#
# Export example
#

inp = np.array([[1]], dtype=float)
nn = IModel(
    1,  # input size
    [1],  # size of hidden layers
    1,  # output size
)
nn.export_to_file("./test_export")

nn_loaded = IModel(
    1,  # input size
    [1],  # size of hidden layers
    1,  # output size
)
nn_loaded.from_file(
    "./test_export"
)  # restore neural network from file (now weights must be equal to nn weights)

print(nn.feedforward(inp).numpy())  # call feedforward function for neural networks
print(nn_loaded.feedforward(inp).numpy())  # must be same result

# print first (exported) neural network description
print("FIRST")
for i, layer in enumerate(nn.network.blocks):
    print(layer.w.numpy(), layer.b.numpy())
print(
    nn.network.out_layer.w.numpy(),
    nn.network.out_layer.b.numpy(),
    nn.network.out_layer.activation_name,
)

print()

# print second (imported) neural network description
print("SECOND")
for i, layer in enumerate(nn_loaded.network.blocks):
    print(layer.w.numpy(), layer.b.numpy())

print(
    nn_loaded.network.out_layer.w.numpy(),
    nn_loaded.network.out_layer.b.numpy(),
    nn_loaded.network.out_layer.activation_name,
)


#
# Train neural network on quadratic function
#


def f_x2(x):
    return x**2


x_data = np.array([[i / 10] for i in range(0, 101)])
f_x_data = np.array([f_x2(x) for x in x_data])

# # Create neural network with 1 input, 1 output and hidden layer with 2 neurons and default activation
# nn = IModel.create_neuron(1, 1, [2])
# # Configure neural network for train with default parameters
# nn.compile()
# # Train on prepared X and Y data
# nn.train(x_data, f_x_data)

#
# Export neural network to cpp
#

nn = IModel.create_neuron(2, 2, [2])
# Export neural network as c-style function
nn.export_to_cpp("test1")
# Export neural network as c++-style function
nn.export_to_cpp("test2", array_type="vector")
# Export neural network as c++-style function and specify path to compiler
nn.export_to_cpp("test3", array_type="vector", path_to_compiler="g++")
# Export neural network as parameters
nn.export_to_file("test_desc")

#
# Predict sin with different loss functions and build plot for every loss
#

from degann.equations import build_plot
from math import sin
import matplotlib.pyplot as plt


def sin_x(x):
    return sin(x)


x_data = np.array(
    [[i / 10] for i in range(0, 101)]
)  # X data, 100 points on [0, 10] with unifrom distribution
f_x_data = np.array([sin_x(x) for x in x_data])  # Y data

all_l = [key for key in losses.get_all_loss_functions()]  # All loss function names
for los in all_l:
    nn = IModel(
        input_size=1,  # Input vector len
        block_size=[10, 10],  # Size of hidden layers (two layer, each have 10 neurons)
        output_size=1,  # Output vector size
        activation_func=["swish", "swish", "linear"],
        # activation functions for layers (2 hidden layer, 1 output layer)
    )
    nn.compile(optimizer="Adam", loss_func=los)

    his = nn.train(
        x_data,
        f_x_data,
        epochs=10,
        verbose=0,  # train progress output disable
    )
    print(los, his.history["loss"][-1])

    # build plot based on the responses of neural network on the interval, with the given step
    build_plot(nn, (0.0, 10.0), 0.01, title=los)
# build sin function
plt.plot(x_data, f_x_data)
plt.title("original")
plt.show()

#
# Train two networks with different activations and optimizers for predict solution of y' + 3y = 0
#

x_data = np.array([[i / 50] for i in range(0, 51)])  # X data, 200 points
f_x_data = np.array([LF_ODE_1_solution(x) for x in x_data])  # Y data

# sizes of hidden layers in neural network
shape = [10, 10, 10, 10, 10, 10]

acts = [
    ["swish"] * 6 + ["linear"],  # first type of activation functions
    ["sigmoid"] * 6 + ["linear"],  # second type of activation functions
    ["linear"] * 7,  # third type of activation functions
]

optimizers = ["SGD", "Adam", "RMSprop"]  # training algorithms

# loss function for neural network
los = "Huber"
# count of training epochs
epochs = 50

input_len = 1
output_len = 1

for opt in optimizers:
    nets = []
    for act in acts:
        nets.append(
            IModel(
                input_size=input_len,
                block_size=shape,
                output_size=output_len,
                activation_func=act,
            )
        )
    for i, nn in enumerate(nets):
        # configure
        nn.compile(optimizer=opt, loss_func=los)

        # train
        his = nn.train(
            x_data,
            f_x_data,
            epochs=epochs,
            verbose=0,
        )
        print(opt, his.history["loss"][-1])

    # build plot based on the responses of neural networks on the interval, with the given step
    build_plot(
        nets,
        (0.0, 1.0),
        0.001,
        title=opt,
        labels=[acts[0][0], acts[1][0], acts[2][0], "f(x) = e^(3x)"],
        true_data=(x_data, f_x_data),
    )
