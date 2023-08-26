import csv
import math
import random
from datetime import datetime
from itertools import product
from typing import Callable, List, Tuple

from degann.networks.callbacks import MeasureTrainTime
from degann.networks import imodel
from degann.networks.nn_code import decode
from degann.networks.generate import choose_neighbor, EpochParameter, CodeParameter, random_generate, generate_neighbor


def temperature_exp(t, alpha, **kwargs):
    return t * alpha


def temperature_lin(k, k_max, **kwargs):
    return 1 - (k + 1) / k_max


def distance_const(d):
    def d_c(**kwargs):
        return d
    return d_c


def distance_lin(offset, multiplier):
    def d_l(temperature, **kwargs):
        return offset + temperature * multiplier
    return d_l


def simulated_annealing(
        in_size,
        out_size,
        data,
        val_data=None,
        k_max: int = 100,
        start_net: dict = None,
        method: Callable = generate_neighbor,
        temperature_method: Callable = temperature_lin,
        distance_method: Callable = distance_const(150),
        opt: str = "Adam",
        loss: str = "Huber",
        file_name: str = "",
        logging: bool = False
):
    gen = random_generate()
    if start_net is None:
        b, a = decode(gen[0].value(), offset=8)
        curr_best = imodel.IModel(in_size, b, out_size, a + ["linear"])
        curr_best.compile(optimizer=opt, loss_func=loss)
    else:
        curr_best = imodel.IModel(in_size, [], out_size, ["linear"])
        curr_best = curr_best.from_dict(start_net)
    curr_epoch = gen[1].value()
    hist = curr_best.train(data[0], data[1], epochs=curr_epoch, verbose=0)
    curr_loss = hist.history["loss"][-1]
    best_val_loss = curr_best.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)[
        "loss"
    ] if val_data is not None else None
    best_epoch = curr_epoch
    best_nn = curr_best.to_dict()
    best_gen = gen
    best_a = curr_best.get_activations
    best_loss = curr_loss

    gen = (CodeParameter(gen[0].value()), EpochParameter(gen[1].value()))
    k = 0
    t = 1
    while k < k_max - 1 and curr_loss > 1e-20:
        t = temperature_method(k=k, k_max=k_max, t=1, alpha=0.97)
        distance = distance_method(temperature=t)

        gen_neighbor = choose_neighbor(method, parameters=(gen[0].value(), gen[1].value()), distance=distance)
        b, a = decode(gen_neighbor[0].value(), offset=8)
        neighbor = imodel.IModel(in_size, b, out_size, a + ["linear"])
        neighbor.compile(optimizer=opt, loss_func=loss)
        neighbor_hist = neighbor.train(
            data[0], data[1], epochs=gen_neighbor[1].value(), verbose=0
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

            history = {"code": [gen[0].value()], "epoch": [gen[1].value()], "shapes": [curr_best.get_shape],
                       "activations": [a], "loss function": [loss], "optimizer": [opt], "loss": [curr_loss]}
            curr_val_loss = curr_best.evaluate(
                val_data[0], val_data[1], verbose=0, return_dict=True
            )["loss"] if val_data is not None else None
            history["validation loss"] = [curr_val_loss]

            if logging:
                fn = f"{file_name}_{len(data[0])}_ann_{loss}_{opt}"
                with open(
                        f"./{fn}.csv",
                        "a",
                        newline="",
                ) as outfile:
                    writer = csv.writer(outfile)
                    writer.writerows(zip(*history.values()))

            if curr_loss < best_loss:
                best_loss = curr_loss
                best_epoch = curr_epoch
                best_nn = curr_best.to_dict()
                best_gen = gen
                best_a = a.copy()
                best_val_loss = curr_val_loss
        k += 1

    history = {"code": [best_gen[0].value()], "epoch": [best_gen[1].value()], "shapes": [curr_best.get_shape],
               "activations": [best_a], "loss function": [loss], "optimizer": [opt], "loss": [best_loss],
               "validation loss": [best_val_loss]}

    if logging:
        fn = f"{file_name}_{len(data[0])}_annealing_{loss}_{opt}"
        with open(
                f"./{fn}.csv",
                "a",
                newline="",
        ) as outfile:
            writer = csv.writer(outfile)
            writer.writerows(zip(*history.values()))
    return best_loss, best_epoch, loss, opt, best_nn


def random_search(
        data,
        opt,
        loss,
        iterations,
        val_data=None,
        callbacks=None,
        logging=False,
        file_name: str = ""
):
    best_net = None
    best_loss = 1e6
    best_epoch = None
    for _ in range(iterations):
        gen = random_generate()
        b, a = decode(gen[0].value(), offset=8)
        curr_best = imodel.IModel(1, b, 1, a + ["linear"])
        curr_best.compile(optimizer=opt, loss_func=loss)
        curr_epoch = gen[1].value()
        hist = curr_best.train(data[0], data[1], epochs=curr_epoch, verbose=0, callbacks=callbacks)
        curr_loss = hist.history["loss"][-1]
        history = {"code": [gen[0].value()], "epoch": [gen[1].value()], "shapes": [curr_best.get_shape],
                   "activations": [a], "loss": [curr_loss], "loss function": [loss], "optimizer": [opt]}

        if val_data is not None:
            curr_val_loss = curr_best.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)[
                "loss"
            ]
            history["validation loss"] = [curr_val_loss]

        if curr_loss < best_loss:
            best_epoch = curr_epoch
            best_net = curr_best.to_dict()
            best_loss = curr_loss
        if logging:
            fn = f"{file_name}_{len(data[0])}_random_{loss}_{opt}"
            with open(
                    f"./{fn}.csv",
                    "a",
                    newline="",
            ) as outfile:
                writer = csv.writer(outfile)
                writer.writerows(zip(*history.values()))
    return best_loss, best_epoch, loss, opt, best_net


def full_search_step(
        code,
        num_epoch,
        opt,
        loss,
        data,
        val_data=None,
        logging=False,
        file_name: str = "",
        callbacks=None
):
    history = dict()
    b, a = decode(code, block_size=1, offset=8)
    nn = imodel.IModel(1, b, 1, a + ["linear"])
    nn.compile(optimizer=opt, loss_func=loss)
    temp_his = nn.train(
        data[0], data[1], epochs=num_epoch, verbose=0, callbacks=callbacks
    )

    history["shapes"] = [nn.get_shape]
    history["activations"] = [a]
    history["code"] = [code]
    history["epoch"] = [num_epoch]
    history["optimizer"] = [opt]
    history["loss function"] = [loss]
    history["loss"] = [temp_his.history["loss"][-1]]
    history["validation loss"] = [nn.evaluate(
            val_data[0], val_data[1], verbose=0, return_dict=True
        )["loss"]] if val_data is not None else [None]
    history["train_time"] = [nn.network.trained_time["train_time"]]

    if logging:
        file_name = f"{file_name}_{len(data[0])}_{num_epoch}_{loss}_{opt}"
        with open(
                f"./{file_name}.csv",
                "a",
                newline="",
        ) as outfile:
            writer = csv.writer(outfile)
            writer.writerows(zip(*history.values()))
    return (history["loss"][0], history["validation loss"][0], nn.to_dict())


def full_search(
        data: tuple,
        net_size: Tuple[int, int],
        alph,
        epoch_bound: Tuple[int, int, int],
        optimizers: List[str],
        losses: List[str],
        val_data=None,
        logging=False,
        file_name: str = "",
        verbose=False
):
    best_net = None
    best_loss = 1e6
    best_epoch = None
    best_loss_func = None
    best_opt = None
    time_viewer = MeasureTrainTime()
    for i in range(net_size[0], net_size[1] + 1):
        if verbose:
            print(i, datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
        codes = product(alph, repeat=i)
        for elem in codes:
            code = "".join(elem)
            for epoch in range(epoch_bound[0], epoch_bound[1] + 1, epoch_bound[2]):
                for opt in optimizers:
                    for loss in losses:
                        curr_loss, curr_val_loss, curr_nn = full_search_step(
                            code=code,
                            num_epoch=epoch,
                            opt=opt,
                            loss=loss,
                            data=data,
                            val_data=val_data,
                            callbacks=[time_viewer],
                            logging=logging,
                            file_name=file_name,
                        )
                        if best_loss > curr_loss:
                            best_net = curr_nn
                            best_loss = curr_loss
                            best_epoch = epoch
                            best_loss_func = loss,
                            best_opt = opt
    return best_loss, best_epoch, best_loss_func, best_opt, best_net
