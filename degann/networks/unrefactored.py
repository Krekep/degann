import copy
import csv
import math
import random
import time
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
)
from degann.networks import imodel

act_to_hex = {
    "elu": "0",
    "relu": "1",
    "gelu": "2",
    "selu": "3",
    "exponential": "4",
    "linear": "5",
    "sigmoid": "6",
    "hard_sigmoid": "7",
    "swish": "8",
    "tanh": "9",
    "softplus": "a",
    "softsign": "b",
}
reverse_act_to_hex = {v: k for k, v in act_to_hex.items()}


def encode(nn: imodel.IModel):
    blocks = nn.get_shape
    activations = nn.get_activations
    res = ""

    offset = min(blocks)
    for layer, act in zip(blocks, activations):
        curr = hex(layer - offset)[2:] + act_to_hex[act]
        res += curr

    return res


def decode(s: str, block_size: int = 1, offset: int = 0):
    blocks = []
    activations = []
    for block in range(0, len(s), block_size + 1):
        temp = s[block : block + block_size]
        blocks.append(int(temp, 16) + offset)
        activations.append(reverse_act_to_hex[s[block + block_size]])
    return blocks, activations


file_name = "LH_ODE_1_solution.csv"
nn_data_x = np.array([[i / 1000] for i in range(0, 1_001)])  # X data
nn_data_y = np.array([LF_ODE_1_solution(x) for x in nn_data_x])
train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(40)]
train_idx.sort()
nn_data_x = nn_data_x[train_idx, :]  # X data
nn_data_y = nn_data_y[train_idx, :]  # Y data

alph_n_full = "0123456789abcdef"
alph_n_div3 = "0369cf"
alph_a = "0689"
all_variants = ["".join(elem) for elem in product(alph_n_full, alph_a)]
div3_variants = ["".join(elem) for elem in product(alph_n_div3, alph_a)]
print(len(all_variants), len(div3_variants))
opt = "Adam"
loss = "MeanSquaredError"

history = dict()
history["shapes"] = []
history["activations"] = []
history["code"] = []
history["epoch"] = []
history["optimizer"] = []
history["loss function"] = []
history["loss"] = []
history["train_time"] = []
time_viewer = MeasureTrainTime()


with open(
    f"./{file_name}",
    "w",
    newline="",
) as outfile:
    writer = csv.writer(outfile)
    writer.writerow(history.keys())


def train_step(code):
    for value in history.values():
        del value[:]
    b, a = decode(code, block_size=1, offset=8)
    nn = imodel.IModel(1, b, 1, a + ["linear"])
    nn.compile(optimizer=opt, loss_func=loss)
    temp_his = nn.train(
        nn_data_x, nn_data_y, epochs=200, verbose=0, callbacks=[time_viewer]
    )

    history["loss"].append(temp_his.history["loss"][-1])
    history["train_time"].append(nn.network.trained_time["train_time"])
    history["shapes"].append(nn.get_shape)
    history["activations"].append(a)
    history["code"].append(code)
    history["epoch"].append(200)
    history["optimizer"].append(opt)
    history["loss function"].append(loss)

    with open(
        f"./{file_name}",
        "a",
        newline="",
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerows(zip(*history.values()))


def run(start, finish, alph):
    for i in range(start, finish + 1):
        print(i, datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
        codes = product(alph, repeat=i)
        for elem in codes:
            code = "".join(elem)
            train_step(code)


run(1, 2, all_variants)
print("END 1, 2", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
run(3, 3, div3_variants)
print("END 3", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))

for i in range(4, 11):
    print(i, datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
    for size in alph_n_full:
        for act in alph_a:
            train_step((size + act) * i)
print("END 4, 11", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))


class MetaParameter:
    def distance(self, other):
        pass

    def value(self):
        pass


class CodeParameter(MetaParameter):
    block_size = 1

    def __init__(self, s):
        self.code = s
        self.blocks = [
            self.code[i : i + self.block_size + 1]
            for i in range(0, len(self.code), self.block_size + 1)
        ]

    def distance(self, other):
        if isinstance(other, str):
            other = CodeParameter(other)
        val_a = 0
        act_a = []
        for block in self.blocks:
            dec = decode(block, block_size=self.block_size, offset=8)
            val_a += sum(dec[0])
            act_a.append(*dec[1])

        val_b = 0
        act_b = []
        for block in other.blocks:
            dec = decode(block, block_size=self.block_size, offset=8)
            val_b += sum(dec[0])
            act_b.append(*dec[1])

        diff = 0
        for i in range(min(len(act_a), len(act_b))):
            if act_a[i] != act_b[i]:
                diff += 1
        diff += abs(len(act_a) - len(act_b))
        return abs(val_a - val_b) + diff

    def value(self):
        return "".join(self.blocks)

    def block_cost(self, block, offset=8):
        return sum(decode(block, block_size=self.block_size, offset=offset)[0])


class EpochParameter(MetaParameter):
    log_value = 1.1
    pow_scale = 3

    def __init__(self, epoch):
        self.epoch = epoch

    def distance(self, other):
        if isinstance(other, int):
            return math.log(
                self.epoch**EpochParameter.pow_scale, EpochParameter.log_value
            ) - math.log(other**EpochParameter.pow_scale, EpochParameter.log_value)
        return math.log(
            self.epoch**EpochParameter.pow_scale, EpochParameter.log_value
        ) - math.log(other.epoch**EpochParameter.pow_scale, EpochParameter.log_value)

    def value(self):
        return self.epoch


def random_generate():
    block = random.randint(1, 6)
    code = ""
    for i in range(block):
        code += (
            alph_n_full[random.randint(0, len(alph_n_full) - 1)]
            + alph_a[random.randint(0, len(alph_a) - 1)]
        )
    epoch = random.randint(5, 500)
    return code, epoch


def generate_neighbor(parameters, distance: int = 30):
    code = parameters[0]
    epoch = parameters[1]
    is_stop = 0
    new_epoch = EpochParameter(epoch)
    new_code = CodeParameter(code)

    while distance > 0 and is_stop == 0:
        branch = random.random()
        curr_code = CodeParameter(new_code.value())
        if branch < 0.5:  # change epoch
            sign = random.random()
            if sign < 0.5:  # new epoch more than previous
                new_epoch = EpochParameter(
                    min(
                        random.randint(
                            epoch, int(EpochParameter.log_value**distance * epoch)
                        ),
                        500,
                    )
                )
            else:  # new epoch less than previous
                new_epoch = EpochParameter(
                    max(
                        random.randint(
                            int(epoch / (EpochParameter.log_value**distance)), epoch
                        ),
                        5,
                    )
                )
            distance -= abs(new_epoch.distance(epoch))
        elif branch < 1 and distance >= 1:  # change code
            chosen_block = random.randint(0, len(curr_code.blocks) - 1)

            command = random.randint(1, 5)
            match command:
                case 1:  # add block
                    if distance >= 9:
                        new_block = (
                            alph_n_full[
                                random.randint(
                                    0, min(int(distance - 9), len(alph_n_full) - 1)
                                )
                            ]
                            + alph_a[random.randint(0, len(alph_a) - 1)]
                        )
                        curr_code = CodeParameter("".join(curr_code.blocks) + new_block)
                case 2:  # increase block size
                    new_block = (
                        hex(
                            min(
                                int(curr_code.blocks[chosen_block][:-1], 16)
                                + random.randint(1, max(1, int(distance))),
                                15**CodeParameter.block_size,
                            )
                        )[2:]
                        + curr_code.blocks[chosen_block][-1]
                    )
                    new_block = (
                        "0" * (CodeParameter.block_size - len(new_block) + 1)
                        + new_block
                    )
                    curr_code.blocks[chosen_block] = new_block
                case 3:  # decrease size of block
                    new_block = (
                        hex(
                            max(
                                int(curr_code.blocks[chosen_block][:-1], 16)
                                - random.randint(1, max(1, int(distance))),
                                0,
                            )
                        )[2:]
                        + curr_code.blocks[chosen_block][-1]
                    )
                    new_block = (
                        "0" * (CodeParameter.block_size - len(new_block) + 1)
                        + new_block
                    )
                    curr_code.blocks[chosen_block] = new_block
                case 4:  # remove last block
                    if len(curr_code.blocks) > 1:
                        temp = CodeParameter(curr_code.value())
                        temp.blocks.pop()
                        if abs(temp.distance(curr_code)) < distance:
                            curr_code.blocks.pop()
                case 5:  # change activation for block
                    new_block = (
                        curr_code.blocks[chosen_block][:-1]
                        + alph_a[random.randint(0, len(alph_a) - 1)]
                    )
                    curr_code.blocks[chosen_block] = new_block
            distance -= abs(new_code.distance(curr_code))
            if distance < 0:
                distance += abs(new_code.distance(curr_code))
                tttt = abs(new_code.distance(curr_code))
                print(
                    "DEBUG",
                    new_code.blocks,
                    curr_code.blocks,
                    chosen_block,
                    command,
                    distance,
                    tttt,
                )
                distance -= tttt
            new_code = curr_code
        is_stop = random.randint(0, 3)
    return new_code, new_epoch


def temperature_exp(t, alpha):
    return t * alpha


def temperature_lin(k, k_max):
    return 1 - (k + 1) / k_max


def simulated_annealing(
    in_size,
    out_size,
    x_data,
    y_data,
    val_x=None,
    val_y=None,
    k_max: int = 100,
    opt: str = "Adam",
    loss: str = "Huber",
):
    gen = random_generate()
    b, a = decode(gen[0], offset=8)
    curr_best = imodel.IModel(in_size, b, out_size, a + ["linear"])
    curr_best.compile(optimizer=opt, loss_func=loss)
    curr_epoch = gen[1]
    hist = curr_best.train(x_data, y_data, epochs=curr_epoch, verbose=0)
    curr_loss = hist.history["loss"][-1]
    best_val_loss = curr_best.evaluate(val_x, val_y, verbose=0, return_dict=True)[
        "loss"
    ]
    best_epoch = curr_epoch
    best_shape = curr_best.get_shape
    best_act = a
    best_loss = curr_loss

    gen = (CodeParameter(gen[0]), EpochParameter(gen[1]))
    k = 0
    t = 1
    while k < k_max - 1 and curr_loss > 1e-20:
        t = temperature_lin(k, k_max)
        gen_neighbor = generate_neighbor((gen[0].value(), gen[1].value()), distance=50)
        b, a = decode(gen_neighbor[0].value(), offset=8)
        neighbor = imodel.IModel(in_size, b, out_size, a + ["linear"])
        neighbor.compile(optimizer=opt, loss_func=loss)
        neighbor_hist = neighbor.train(
            x_data, y_data, epochs=gen_neighbor[1].value(), verbose=0
        )
        neighbor_loss = neighbor_hist.history["loss"][-1]

        if (
            neighbor_loss < curr_loss
            or math.e ** ((curr_loss - neighbor_loss) / t) > random.random()
        ):
            curr_best = neighbor
            gen = gen_neighbor
            curr_epoch = gen_neighbor[1].value()
            curr_loss = neighbor_loss
            if curr_loss < best_loss:
                best_loss = curr_loss
                best_epoch = curr_epoch
                best_shape = curr_best.get_shape
                best_val_loss = curr_best.evaluate(
                    val_x, val_y, verbose=0, return_dict=True
                )["loss"]
                best_act = []
                for i in range(len(curr_best.get_activations) - 1):
                    best_act.append(curr_best.get_activations[i])
        k += 1

    return best_shape, best_act, best_loss, best_epoch, best_val_loss


name_to_funcs = {
    "LF_ODE_1": LF_ODE_1_solution,
    "LH_ODE_1": LH_ODE_1_solution,
    "LF_ODE_3": LF_ODE_3_solution,
    "LH_ODE_2": LH_ODE_2_solution,
    "NLF_ODE_1": NLF_ODE_1_solution,
    "NLF_ODE_2": NLF_ODE_2_solution,
}
for func_name in ["LF_ODE_1", "LH_ODE_1", "LF_ODE_3"]:
    # for func_name in ["LH_ODE_2", "NLF_ODE_1", "NLF_ODE_2"]:
    nn_data_x = np.array([[i / 1000] for i in range(100, 1_001)])  # X data
    nn_data_y = np.array([name_to_funcs[func_name](x) for x in nn_data_x])
    train_idx = [randint(0, len(nn_data_x) - 1) for _ in range(40)]
    train_idx.sort()
    val_idx = [randint(0, len(nn_data_x) - 1) for _ in range(20)]
    val_idx.sort()
    val_data_x = nn_data_x[val_idx, :]  # validation X data
    val_data_y = nn_data_y[val_idx, :]  # validation Y data
    nn_data_x = nn_data_x[train_idx, :]  # X data
    nn_data_y = nn_data_y[train_idx, :]  # Y data

    for loss_name in ["Huber", "MeanAbsolutePercentageError"]:
        # f = open(f"otzhig_{func_name}_{loss_name}.txt", "w")
        # f.write("shape loss epoch\n")
        # print("shape loss val_loss epoch time")
        max_iter = 50
        avg_loss = 0
        avg_val_loss = 0
        launches = 100
        for i in range(launches):
            start_t = time.perf_counter()
            nn_shape, nn_acts, nn_loss, nn_epoch, val_loss = simulated_annealing(
                1,
                1,
                nn_data_x,
                nn_data_y,
                val_x=val_data_x,
                val_y=val_data_y,
                k_max=max_iter,
                loss=loss_name,
            )
            end_t = time.perf_counter()
            avg_loss += nn_loss
            avg_val_loss += val_loss
            f.write(
                str(nn_shape)
                + " "
                + str(nn_acts)
                + " "
                + str(nn_loss)
                + " "
                + str(val_loss)
                + " "
                + str(nn_epoch)
                + "\n"
            )
            print(i, nn_shape, nn_acts, nn_loss, val_loss, nn_epoch, end_t - start_t)
        avg_loss = avg_loss / launches
        avg_val_loss = avg_val_loss / launches
        print(f"avg_loss = {avg_loss}, avg_val_loss = {avg_val_loss}")
        f.write(f"avg_loss = {avg_loss}, avg_val_loss = {avg_val_loss}\n")
        f.close()

        f = open(f"random_{func_name}_{loss_name}.txt", "w")
        f.write("shape loss epoch\n")
        print("shape loss val_loss epoch time")
        avg_loss = 0
        avg_val_loss = 0
        for i in range(launches):
            start_t = time.perf_counter()
            t_loss = 100000
            t_epoch = 100000
            t_val_loss = 100000
            t_shape = []
            t_acts = []

            for j in range(max_iter):
                gen = random_generate()
                b, a = decode(gen[0], offset=8)
                nn = imodel.IModel(1, b, 1, a + ["linear"])
                nn.compile(optimizer="Adam", loss_func=loss_name)
                hist = nn.train(nn_data_x, nn_data_y, epochs=gen[1], verbose=0)
                nn_loss = hist.history["loss"][-1]
                val_loss = nn.evaluate(
                    val_data_x, val_data_y, verbose=0, return_dict=True
                )["loss"]
                if t_loss > nn_loss:
                    t_loss = nn_loss
                    t_val_loss = val_loss
                    t_epoch = gen[1]
                    t_shape = b
                    t_acts = a
            end_t = time.perf_counter()
            avg_loss += t_loss
            avg_val_loss += t_val_loss
            f.write(
                str(t_shape)
                + " "
                + str(t_acts)
                + " "
                + str(t_loss)
                + " "
                + str(t_val_loss)
                + " "
                + str(t_epoch)
                + "\n"
            )
            print(i, t_shape, t_acts, t_loss, t_val_loss, t_epoch, end_t - start_t)
        avg_loss = avg_loss / launches
        avg_val_loss = avg_val_loss / launches
        print(f"avg_loss = {avg_loss}, avg_val_loss = {avg_val_loss}")
        f.write(f"avg_loss = {avg_loss}, avg_val_loss = {avg_val_loss}\n")
        f.close()
