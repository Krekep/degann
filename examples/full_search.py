from datetime import datetime
from itertools import product
from random import randint

import numpy as np

from degann import full_search
from degann.expert.nn_code import (
    alphabet_activations_cut,
    alph_n_div3,
)
from experiments.functions import LH_ODE_1_solution


#
# Prepare data for training. Equation is `sin(10 * x)`
#
num_epoch = 500
data_size = 40
file_name = f"LH_ODE_1_solution"
nn_data_x = np.array([[i / 1000] for i in range(0, 1_001)])  # X data
true_x = np.array([[i / 20] for i in range(0, 20)])
true_y = np.array([LH_ODE_1_solution(x) for x in true_x])
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
# all_variants = ["".join(elem) for elem in product(alph_n_full, alphabet_activations_cut)]
div3_variants = [
    "".join(elem) for elem in product(alph_n_div3, alphabet_activations_cut)
]
print(file_name)
print(len(div3_variants))

opt = "Adam"  # optimizer
loss = "MaxAbsoluteDeviation"  # loss function

#
# Start full search over specified parameters
#

full_search(
    in_size=1,  # size of input data (x)
    out_size=1,  # size of output data (y)
    data=(nn_data_x, nn_data_y),  # dataset
    opt=[opt],  # list of optimizers
    loss=[loss],  # list of loss functions
    min_epoch=num_epoch,  # starting number of epochs
    max_epoch=num_epoch,  # final number of epochs
    epoch_step=10,  # step between `min_epoch` and `max_epoch`
    nn_min_length=1,  # starting number of hidden layers of neural networks
    nn_max_length=4,  # final number of hidden layers of neural networks
    nn_alphabet=div3_variants,  # list of possible sizes of hidden layers with activations for them
    logging=True,  # logging search process to file
    file_name=file_name,  # file for logging
    verbose=True,  # print additional information to console during the searching
)
print("END 1, 4", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
