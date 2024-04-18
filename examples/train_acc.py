from random import randint
import numpy as np
from degann.networks.imodel import IModel
from experiments.functions import NLF_ODE_1_solution


def mean(a):
    return sum(a) / len(a)


#
# Prepare data
#
t = [[i / 100] for i in range(10, 101)]
nn_data_x = np.array(t, dtype=float)  # X data
nn_data_y = NLF_ODE_1_solution(nn_data_x)
train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(40)]
train_x = nn_data_x[train_idx, :]  # X data
train_y = nn_data_y[train_idx, :]  # Y data

true_idx = [randint(0, len(nn_data_x) - 1) for _ in range(40)]
true_x = nn_data_x[true_idx, :]
true_y = nn_data_y[true_idx, :]

shapes = [10, 10, 10, 10, 10]  # sizes of hidden layers

acts = ["swish"] * 6 + ["linear"]  # activation functions for layers

los = "MeanAbsolutePercentageError"  # loss function for training
epochs = [100, 150, 200, 300]  # number of epochs

input_len = 1
output_len = 1

for epoch in epochs:
    losses = []  # loss function values
    threshold = 5  # threshold for loss function error
    good = 0  # number of networks whose loss function error is less than a threshold
    bad = 0  # number of networks whose loss function error is greater than a threshold
    for _ in range(100):
        nn = IModel(
            input_size=input_len,
            block_size=shapes,
            output_size=output_len,
            activation_func=acts,
        )
        opt = "Adam"  # training algorithm

        nn.compile(optimizer=opt, loss_func=los, metrics=[])

        nn.train(
            train_x,
            train_y,
            epochs=epoch,
            verbose=0,
        )
        loss_value = nn.evaluate(
            nn_data_x,
            nn_data_y,
            verbose=0,
        )
        if loss_value <= threshold:
            good += 1
        else:
            bad += 1
        losses.append(loss_value)

        # build_plot(nn, (0.1, 1.0), 0.01, true_data=[true_x, true_y])

    mean_loss = mean(losses)
    print(f"Average {los} of {epoch} for {shapes} shape neural network is {mean_loss}")
    print(
        f"Train accuracy of {epoch} for {shapes} shape neural network is {good / (good + bad)}. Good = {good}, bad = {bad}"
    )
