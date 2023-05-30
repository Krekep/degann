import time
from random import randint, random
import numpy as np

from degann.equations import build_plot
from degann.networks import callbacks
from degann.networks.imodel import IModel
from degann.testlaunches.functions import ST_S_ODE_3_table


def mean(a):
    return sum(a) / len(a)


def standart_deviation(a):
    m = mean(a)
    s = 0
    for i in range(len(a)):
        s += (a[i] - m) ** 2
    return (s / len(a)) ** 0.5


def confidence_interval(times):
    z_95 = 1.96
    m = mean(times)
    s = standart_deviation(times)
    left = m - z_95 * s / (len(times) ** 0.5)
    right = m + z_95 * s / (len(times) ** 0.5)
    return left, right, m, z_95 * s / (len(times) ** 0.5)


#
# Measure predict time row by row
#

input_size = 1
shapes = [
    [10, 10, 10, 10, 10],
    [100, 100, 100],
    [500, 500, 500],
]  # sizes of hidden layers
output_size = 1

# X data size
single_data_size_call = [500, 5_000, 25_000]
# X data
single_x_data_call = [
    np.array([[[random() * 10]] for _ in range(0, size)])
    for size in single_data_size_call
]

for i, shape in enumerate(shapes):
    nn = IModel(input_size, shape, output_size)
    for size in single_x_data_call:
        times = []
        for _ in range(20):
            start_time = time.perf_counter()
            for row in size:
                a = nn.feedforward(row)
            end_time = time.perf_counter()
            call_time = end_time - start_time
            times.append(call_time)
        l, r, m, d = confidence_interval(times)
        print(
            f"Confidence interval for neural network {shape} single time prediction on {len(size)} data size is [{l}, {r}] s, mean is {m} s, dev is +-{d}"
        )
    nn.export_to_cpp(f"time_measure_{i}")

#
# Measure train time
#
print("*********")
nn_data_x = [i / 100 for i in range(0, 4_001)]  # X data
table = ST_S_ODE_3_table(nn_data_x)
temp = np.hsplit(table, np.array([1, 4]))
train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(40)]
nn_data_x = temp[0][train_idx, :]  # X data
nn_data_y = temp[1][train_idx, :]  # Y data

true_idx = [randint(0, len(nn_data_x) - 1) for _ in range(20)]
true_x = nn_data_x[true_idx, :]
true_y = nn_data_y[true_idx, :]

shapes = [10, 10, 10, 10, 10, 10]  # sizes of hidden layers

acts = ["swish"] * 6 + ["linear"]  # activation functions for layers

los = "Huber"  # loss function for training
epochs = 200

input_len = 1
output_len = 3

times = []
for _ in range(20):
    nn = IModel(
        input_size=input_len,
        block_size=shapes,
        output_size=output_len,
        activation_func=acts,
    )
    opt = "Adam"  # training algorithm

    nn.compile(optimizer=opt, loss_func=los)

    time_measurer = callbacks.MeasureTrainTime()  # Callback for measure time
    his = nn.train(
        nn_data_x,
        nn_data_y,
        epochs=epochs,
        verbose=0,
        callbacks=[time_measurer],  # pass callback as parameter
    )
    times.append(nn.network.trained_time["train_time"])

    build_plot(nn, (0.0, 40.0), 0.1, true_data=[true_x, true_y])

    nn.export_to_cpp("train_time_measure")
l, r, m, d = confidence_interval(times)
print(
    f"Confidence interval for neural network {shapes} train time on {len(nn_data_x)} data size is [{l}, {r}] s, mean is {m} s, dev is +-{d}"
)
