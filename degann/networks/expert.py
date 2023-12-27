import csv
import math
import random
from datetime import datetime
from itertools import product
from typing import Callable, List, Tuple

import numpy.random
import tensorflow as tf

from degann.networks.callbacks import MeasureTrainTime
from degann.networks import imodel
from degann.networks.nn_code import decode
from degann.networks.generate import (
    choose_neighbor,
    EpochParameter,
    CodeParameter,
    random_generate,
    generate_neighbor,
)

_algorithms_for_random_generator = {0: "auto_select", 1: "philox", 2: "threefry"}
tf.random.set_global_generator(numpy.random.default_rng())


def update_random_generator(curr_iter: int, cycle_size: int = 0) -> None:
    """
    Set global tensorflow random generator to random state every *cycle_size* times

    Parameters
    ----------
    curr_iter: int
        Counter showing whether it's time to update the random number generator
    cycle_size: int
        How often should we update random number generator (if not positive, then the generator does not change)

    Returns
    -------
    """
    if False and cycle_size > 0 and curr_iter % cycle_size == 0:
        new_g = tf.random.Generator.from_non_deterministic_state(
            alg=_algorithms_for_random_generator[
                random.randint(0, len(_algorithms_for_random_generator) - 1)
            ]
        )
        tf.random.set_global_generator(new_g)
    else:
        pass


def temperature_exp(t: float, alpha: float, **kwargs) -> float:
    """
    Calculate new temperature for simulated annealing as *t * alpha*

    Parameters
    ----------
    t: float
        Current temperature
    alpha: float
        Exponential exponent

    Returns
    -------
    new_t: float
        New temperature
    """
    return t * alpha


def temperature_lin(k: int, k_max: int, **kwargs) -> float:
    """
    Calculate new temperature for simulated annealing as *1 - (k + 1) / k_max*

    Parameters
    ----------
    k: float
        Current iteration
    k_max: float
        Amount of all iterations

    Returns
    -------
    new_t: float
        New temperature
    """
    return 1 - (k + 1) / k_max


def distance_const(d: float) -> Callable:
    """
    Calculate distance to neighbour for simulated annealing as constant

    Parameters
    ----------
    d: float
        Constant distance

    Returns
    -------
    d_c: Callable
        Function returning a constant distance
    """

    def d_c(**kwargs) -> float:
        return d

    return d_c


def distance_lin(offset, multiplier):
    """
    Calculate distance to neighbour for simulated annealing as *offset + temperature * multiplier*

    Parameters
    ----------
    offset: float
    multiplier: float

    Returns
    -------
    d_l: Callable
        Function returning a new distance depending on current temperature
    """

    def d_l(temperature, **kwargs):
        return offset + temperature * multiplier

    return d_l


def log_to_file(history: dict, fn: str):
    with open(
        f"./{fn}.csv",
        "a",
        newline="",
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerows(zip(*history.values()))


def simulated_annealing(
    in_size,
    out_size,
    data,
    val_data=None,
    k_max: int = 100,
    threshold: float = -1,
    start_net: dict = None,
    method: Callable = generate_neighbor,
    temperature_method: Callable = temperature_lin,
    distance_method: Callable = distance_const(150),
    opt: str = "Adam",
    loss: str = "Huber",
    update_gen_cycle: int = 0,
    callbacks: list = None,
    file_name: str = "",
    logging: bool = False,
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
    hist = curr_best.train(
        data[0], data[1], epochs=curr_epoch, verbose=0, callbacks=callbacks
    )
    curr_loss = hist.history["loss"][-1]
    best_val_loss = (
        curr_best.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)[
            "loss"
        ]
        if val_data is not None
        else None
    )
    best_epoch = curr_epoch
    best_nn = curr_best.to_dict()
    best_gen = gen
    best_a = curr_best.get_activations
    best_loss = curr_loss

    history = dict()
    history["shapes"] = [curr_best.get_shape]
    history["activations"] = [best_a]
    history["code"] = [best_gen[0].value()]
    history["epoch"] = [best_gen[1].value()]
    history["optimizer"] = [opt]
    history["loss function"] = [loss]
    history["loss"] = [curr_loss]
    history["validation loss"] = [best_val_loss]
    history["train_time"] = [curr_best.network.trained_time["train_time"]]
    if logging:
        fn = f"{file_name}_{len(data[0])}_0_{loss}_{opt}"
        log_to_file(history, fn)

    gen = (CodeParameter(gen[0].value()), EpochParameter(gen[1].value()))
    k = 0
    t = 1
    while k < k_max - 1 and curr_loss > threshold:
        history = dict()

        update_random_generator(k, cycle_size=update_gen_cycle)
        t = temperature_method(k=k, k_max=k_max, t=1, alpha=0.95)
        distance = distance_method(temperature=t)

        gen_neighbor = choose_neighbor(
            method, parameters=(gen[0].value(), gen[1].value()), distance=distance
        )
        b, a = decode(gen_neighbor[0].value(), offset=8)
        neighbor = imodel.IModel(in_size, b, out_size, a + ["linear"])
        neighbor.compile(optimizer=opt, loss_func=loss)
        neighbor_hist = neighbor.train(
            data[0],
            data[1],
            epochs=gen_neighbor[1].value(),
            verbose=0,
            callbacks=callbacks,
        )
        neighbor_val_loss = (
            neighbor.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)[
                "loss"
            ]
            if val_data is not None
            else None
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
            curr_val_loss = neighbor_val_loss

            if curr_loss < best_loss:
                best_loss = curr_loss
                best_epoch = curr_epoch
                best_nn = curr_best.to_dict()
                best_gen = gen
                best_a = a.copy()
                best_val_loss = curr_val_loss
        k += 1

        history["shapes"] = [neighbor.get_shape]
        history["activations"] = [a]
        history["code"] = [gen_neighbor[0].value()]
        history["epoch"] = [gen_neighbor[1].value()]
        history["optimizer"] = [opt]
        history["loss function"] = [loss]
        history["loss"] = [neighbor_loss]
        history["validation loss"] = [neighbor_val_loss]
        history["train_time"] = [neighbor.network.trained_time["train_time"]]
        if logging:
            fn = f"{file_name}_{len(data[0])}_0_{loss}_{opt}"
            log_to_file(history, fn)
    return best_loss, best_epoch, loss, opt, best_nn, k


def random_search(
    in_size,
    out_size,
    data,
    opt,
    loss,
    iterations,
    val_data=None,
    callbacks=None,
    logging=False,
    update_gen_cycle: int = 0,
    file_name: str = "",
):
    best_net = None
    best_loss = 1e6
    best_epoch = None
    for i in range(iterations):
        history = dict()
        update_random_generator(i, cycle_size=update_gen_cycle)
        gen = random_generate()
        b, a = decode(gen[0].value(), offset=8)
        curr_best = imodel.IModel(in_size, b, out_size, a + ["linear"])
        curr_best.compile(optimizer=opt, loss_func=loss)
        curr_epoch = gen[1].value()
        hist = curr_best.train(
            data[0], data[1], epochs=curr_epoch, verbose=0, callbacks=callbacks
        )
        curr_loss = hist.history["loss"][-1]
        curr_val_loss = (
            curr_best.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)[
                "loss"
            ]
            if val_data is not None
            else None
        )

        history["shapes"] = [curr_best.get_shape]
        history["activations"] = [a]
        history["code"] = [gen[0].value()]
        history["epoch"] = [gen[1].value()]
        history["optimizer"] = [opt]
        history["loss function"] = [loss]
        history["loss"] = [curr_loss]
        history["validation loss"] = [curr_val_loss]
        history["train_time"] = [curr_best.network.trained_time["train_time"]]
        if logging:
            fn = f"{file_name}_{len(data[0])}_0_{loss}_{opt}"
            log_to_file(history, fn)

        if curr_loss < best_loss:
            best_epoch = curr_epoch
            best_net = curr_best.to_dict()
            best_loss = curr_loss
    return best_loss, best_epoch, loss, opt, best_net


def random_search_endless(
    in_size,
    out_size,
    data,
    opt,
    loss,
    threshold,
    max_iter=-1,
    val_data=None,
    callbacks=None,
    logging=False,
    file_name: str = "",
    verbose=False,
):
    nn_loss, nn_epoch, loss_f, opt_n, net = random_search(
        in_size,
        out_size,
        data,
        opt,
        loss,
        1,
        val_data=val_data,
        callbacks=callbacks,
        logging=logging,
        file_name=file_name,
    )
    i = 1
    best_net = net
    best_loss = nn_loss
    best_epoch = nn_epoch
    while nn_loss > threshold and i != max_iter:
        if verbose:
            print(
                f"Random search until less than threshold. Last loss = {nn_loss}. Iterations = {i}"
            )
        nn_loss, nn_epoch, loss_f, opt_n, net = random_search(
            in_size,
            out_size,
            data,
            opt,
            loss,
            1,
            val_data=val_data,
            callbacks=callbacks,
            logging=logging,
            file_name=file_name,
        )
        i += 1
        if nn_loss < best_loss:
            best_net = net
            best_loss = nn_loss
            best_epoch = nn_epoch
    return best_loss, best_epoch, loss, opt, best_net, i


def full_search_step(
    code,
    num_epoch,
    opt,
    loss,
    data,
    repeat: int = 1,
    offset: int = 8,
    val_data=None,
    update_gen_cycle: int = 0,
    logging=False,
    file_name: str = "",
    callbacks=None,
):
    best_net = None
    best_loss = 1e6
    best_val_loss = 1e6
    for i in range(repeat):
        update_random_generator(i, cycle_size=update_gen_cycle)
        history = dict()
        b, a = decode(code, block_size=1, offset=offset)
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
        history["validation loss"] = (
            [nn.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)["loss"]]
            if val_data is not None
            else [None]
        )
        history["train_time"] = [nn.network.trained_time["train_time"]]

        if logging:
            fn = f"{file_name}_{len(data[0])}_{num_epoch}_{loss}_{opt}"
            log_to_file(history, fn)
        if history["loss"][0] < best_loss:
            best_loss = history["loss"][0]
            best_val_loss = history["validation loss"][0]
            best_net = nn.to_dict()
    return (best_loss, best_val_loss, best_net)


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
    verbose=False,
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
                            best_loss_func = (loss,)
                            best_opt = opt
    return best_loss, best_epoch, best_loss_func, best_opt, best_net
