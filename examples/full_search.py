from datetime import datetime
from itertools import product
from random import randint

import numpy as np

from degann import (
    LF_ODE_1_solution,
    MeasureTrainTime,
    LH_ODE_1_solution,
    LF_ODE_3_solution,
    LH_ODE_2_solution,
    NLF_ODE_1_solution,
    NLF_ODE_2_solution,
    IModel,
    build_plot,
)
from degann.networks.nn_code import (
    alph_n_full,
    alph_a,
    alph_n_div3,
    alph_n_div2,
    encode,
)
from degann.networks.expert import full_search, full_search_step

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


all_variants = ["".join(elem) for elem in product(alph_n_full, alph_a)]
div3_variants = ["".join(elem) for elem in product(alph_n_div3, alph_a)]
alph_n_div2_1 = alph_n_div2[: len(alph_n_div2) // 2]
alph_n_div2_2 = alph_n_div2[len(alph_n_div2) // 2 :]
div2_variants_1 = ["".join(elem) for elem in product(alph_n_div2_1, alph_a)]
div2_variants_2 = ["".join(elem) for elem in product(alph_n_div2_2, alph_a)]
div2_variants = ["".join(elem) for elem in product(alph_n_div2, alph_a)]
print(file_name)
print(
    len(all_variants),
    len(div2_variants),
    len(div2_variants_1),
    len(div2_variants_2),
    len(div3_variants),
)
opt = "Adam"
loss = "MeanAbsolutePercentageError"
loss = "MaxAbsoluteDeviation"
#
# full_search(
#     (nn_data_x, nn_data_y),
#     (1, 2),
#     all_variants,
#     (num_epoch, num_epoch, 10),
#     [opt],
#     [loss],
#     (val_data_x, val_data_y),
#     logging=True,
#     file_name=file_name,
#     verbose=1,
# )
# print("END 1, 2", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
# full_search(
#     (nn_data_x, nn_data_y),
#     (3, 3),
#     div2_variants,
#     (num_epoch, num_epoch, 10),
#     [opt],
#     [loss],
#     (val_data_x, val_data_y),
#     logging=True,
#     file_name=file_name,
#     verbose=1,
# )
# print("END 3", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
#
# time_viewer = MeasureTrainTime()
# for i in range(4, 11):
#     print(i, datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
#     for size in alph_n_full:
#         for act in alph_a:
#             full_search_step(
#                 (size + act) * i,
#                 num_epoch,
#                 opt,
#                 loss,
#                 (nn_data_x, nn_data_y),
#                 val_data=(val_data_x, val_data_y),
#                 logging=True,
#                 file_name=file_name,
#                 callbacks=[time_viewer],
#             )
# print("END 4, 11", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))


num_epoch = 800
data_size = 40
test_code = "e8e6e6"
func = "sin(10x)"
l = 1e6
while l > 90:
    l, _, net = full_search_step(
        test_code,
        num_epoch,
        opt,
        loss,
        (nn_data_x, nn_data_y),
        # logging=True,
        # file_name=file_name,
        # val_data=(val_data_x, val_data_y),
    )
    print(f"Candidate full search. Last loss = {int(l)}")
nn = IModel(1, [], 1)
nn.from_dict(net)
code = encode(nn, offset=8)
# build_plot(
#     nn,
#     (0.0, 1.0),
#     0.01,
#     true_data=(true_x, true_y),
#     labels=[code, func],
#     title=f"Full search candidate. Loss ({loss}) = {int(l)}"
# )
print(f"Candidate loss {l}")

# file_name = f"run_{test_code}"
t = [50, 40, 30, 20]
t = [0.5, 0.3, 0.1]
for thr in t:
    l = 1e6
    i = 0
    while l > thr:
        l, _, net = full_search_step(
            test_code,
            num_epoch,
            opt,
            loss,
            (nn_data_x, nn_data_y),
            # logging=True,
            # file_name=file_name,
            # val_data=(val_data_x, val_data_y),
        )
        i += 1
        print(
            f"Full search until less than threshold. Last loss = {l}. Iterations = {i}"
        )
    nn = IModel(1, [], 1)
    nn.from_dict(net)
    code = encode(nn, offset=8)
    build_plot(
        nn,
        (0.0, 1.0),
        0.01,
        true_data=(true_x, true_y),
        labels=[code, func],
        title=f"{loss} loss function, {thr} loss threshold. {i} iterations",
    )
    print(f"Loss {l}")

"""
Candidate loss 38.25630187988281
threshold 50. Loss 38.15449905395508
threshold 40. Loss 39.71052169799805
threshold 30. Loss 28.639209747314453

threshold 0.5. Loss 0.3835499882698059
threshold 0.3. Loss 0.27521783113479614
threshold 0.1. Loss 0.0967613235116005
"""
