import copy
import csv
import math
import random
import time
from datetime import datetime
from itertools import product
from random import randint
from typing import Callable

import numpy as np

from degann import LH_ODE_1_solution
from degann.networks.expert import full_search, full_search_step
from degann.networks.nn_code import decode, alph_a

code = "e8e6e6"
print(decode(code, offset=8))
epoch = 500
data_size = 40
opt = "Adam"
loss_func = "MeanAbsolutePercentageError"

file_name = f"LH_ODE_1_solution"
nn_data_x = np.array([[i / 1000] for i in range(0, 1_001)])  # X data
nn_data_y = np.array([LH_ODE_1_solution(x) for x in nn_data_x])
train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(data_size)]
train_idx.sort()
val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(20)]
val_idx.sort()
val_data_x = nn_data_x[val_idx, :]  # validation X data
val_data_y = nn_data_y[val_idx, :]  # validation Y data
nn_data_x = nn_data_x[train_idx, :]  # X data
nn_data_y = nn_data_y[train_idx, :]  # Y data

history = {
    "shapes": [],
    "activations": [],
    "code": [],
    "epoch": [],
    "optimizer": [],
    "loss function": [],
    "loss": [],
    "validation loss": [],
    "train_time": [],
}

lower_ep = 520
upper_ep = 520
# for epoch in range(lower_ep, upper_ep, 30):
#     fn = f"{file_name}_{data_size}_{epoch}_{loss_func}_{opt}.csv"
#     with open(
#             f"./{fn}",
#             "w",
#             newline="",
#     ) as outfile:
#         writer = csv.writer(outfile)
#         writer.writerow(history.keys())

temp_alph = "def"
all_variants = ["".join(elem) for elem in product(temp_alph, alph_a)]
print(len(all_variants), lower_ep, upper_ep)
for _ in range(10):
    full_search(
        (nn_data_x, nn_data_y),
        (3, 3),
        all_variants,
        (lower_ep, upper_ep, 30),
        [opt],
        [loss_func],
        (val_data_x, val_data_y),
        logging=True,
        file_name=file_name,
        verbose=1,
    )
#
# print("Start 4")
# for _ in range(10):
#     for el in all_variants:
#         new_code = code + el
#         for epoch in range(200, 520, 30):
#             full_search_step(
#                 new_code,
#                 epoch,
#                 opt,
#                 loss_func,
#                 data=(nn_data_x, nn_data_y),
#                 val_data=(val_data_x, val_data_y),
#                 logging=True,
#                 file_name=file_name,
#             )
#         for epoch in range(520, 610, 30):
#             full_search_step(
#                 new_code,
#                 epoch,
#                 opt,
#                 loss_func,
#                 data=(nn_data_x, nn_data_y),
#                 val_data=(val_data_x, val_data_y),
#                 logging=True,
#                 file_name=file_name,
#             )
# print("End 4")
# print("Start 2")
# for _ in range(10):
#     for epoch in range(200, 520, 30):
#         full_search_step(
#             code[:-2],
#             epoch,
#             opt,
#             loss_func,
#             data=(nn_data_x, nn_data_y),
#             val_data=(val_data_x, val_data_y),
#             logging=True,
#             file_name=file_name,
#         )
#     for epoch in range(520, 610, 30):
#         full_search_step(
#             code[:-2],
#             epoch,
#             opt,
#             loss_func,
#             data=(nn_data_x, nn_data_y),
#             val_data=(val_data_x, val_data_y),
#             logging=True,
#             file_name=file_name,
#         )
# print("End 2")
