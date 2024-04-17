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
from degann.expert.nn_code import decode, alph_n_full, alphabet_activations
from degann.expert.generate import (
    choose_neighbor,
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


def temperature_exp(alpha: float) -> Callable[[float], float]:
    """
    Calculate new temperature for simulated annealing as *t * alpha*

    Parameters
    ----------
    alpha: float
        Exponential exponent

    Returns
    -------
    t_e: Callable[[float], float]
        Temperature function
    """

    def t_e(t: float) -> float:
        """
        Parameters
        ----------
        t: float
            Current temperature

        Returns
        -------
        new_t: float
            New temperature
        """
        return t * alpha

    return t_e


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
    max_iter: int = 100,
    threshold: float = -1,
    start_net: dict = None,
    method: Callable = generate_neighbor,
    temperature_method: Callable = temperature_lin,
    distance_method: Callable = distance_const(150),
    min_epoch=100,
    max_epoch=700,
    opt: str = "Adam",
    loss: str = "Huber",
    nn_min_length: int = 1,
    nn_max_length: int = 6,
    nn_alphabet: list[str] = [
        "".join(elem) for elem in product(alph_n_full, alphabet_activations)
    ],
    alphabet_block_size: int = 1,
    alphabet_offset: int = 8,
    update_gen_cycle: int = 0,
    callbacks: list = None,
    file_name: str = "",
    logging: bool = False,
):
    gen = random_generate(
        min_epoch=min_epoch,
        max_epoch=max_epoch,
        min_length=nn_min_length,
        max_length=nn_max_length,
        alphabet=nn_alphabet,
    )
    if start_net is None:
        b, a = decode(
            gen[0].value(), block_size=alphabet_block_size, offset=alphabet_offset
        )
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

    k = 0
    t = 1
    while k < max_iter - 1 and curr_loss > threshold:
        history = dict()

        update_random_generator(k, cycle_size=update_gen_cycle)
        t = temperature_method(k=k, k_max=max_iter, t=t)
        distance = distance_method(temperature=t)

        gen_neighbor = choose_neighbor(
            method,
            alphabet=nn_alphabet,
            parameters=(gen[0].value(), gen[1].value()),
            distance=distance,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            min_length=nn_min_length,
            max_length=nn_max_length,
        )
        b, a = decode(
            gen_neighbor[0].value(),
            block_size=alphabet_block_size,
            offset=alphabet_offset,
        )
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
    min_epoch=100,
    max_epoch=700,
    val_data=None,
    callbacks=None,
    logging=False,
    nn_min_length: int = 1,
    nn_max_length: int = 6,
    nn_alphabet: list[str] = [
        "".join(elem) for elem in product(alph_n_full, alphabet_activations)
    ],
    alphabet_block_size: int = 1,
    alphabet_offset: int = 8,
    update_gen_cycle: int = 0,
    file_name: str = "",
):
    best_net = None
    best_loss = 1e6
    best_epoch = None
    for i in range(iterations):
        history = dict()
        update_random_generator(i, cycle_size=update_gen_cycle)
        gen = random_generate(
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            min_length=nn_min_length,
            max_length=nn_max_length,
            alphabet=nn_alphabet,
        )

        b, a = decode(
            gen[0].value(), block_size=alphabet_block_size, offset=alphabet_offset
        )
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
    min_epoch=100,
    max_epoch=700,
    val_data=None,
    callbacks=None,
    nn_min_length: int = 1,
    nn_max_length: int = 6,
    nn_alphabet: list[str] = [
        "".join(elem) for elem in product(alph_n_full, alphabet_activations)
    ],
    alphabet_block_size: int = 1,
    alphabet_offset: int = 8,
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
        min_epoch=min_epoch,
        max_epoch=max_epoch,
        val_data=val_data,
        nn_min_length=nn_min_length,
        nn_max_length=nn_max_length,
        nn_alphabet=nn_alphabet,
        alphabet_block_size=alphabet_block_size,
        alphabet_offset=alphabet_offset,
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
            min_epoch=min_epoch,
            max_epoch=max_epoch,
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
    in_size: int,
    out_size: int,
    code: str,
    num_epoch: int,
    opt: str,
    loss: str,
    data,
    repeat: int = 1,
    alphabet_block_size: int = 1,
    alphabet_offset: int = 8,
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
        b, a = decode(code, block_size=alphabet_block_size, offset=alphabet_offset)
        nn = imodel.IModel(in_size, b, out_size, a + ["linear"])
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
    in_size: int,
    out_size: int,
    data: tuple,
    opt: List[str],
    loss: List[str],
    min_epoch: int = 100,
    max_epoch: int = 700,
    epoch_step: int = 50,
    nn_min_length: int = 1,
    nn_max_length: int = 6,
    nn_alphabet: list[str] = [
        "".join(elem) for elem in product(alph_n_full, alphabet_activations)
    ],
    alphabet_block_size: int = 1,
    alphabet_offset: int = 8,
    val_data=None,
    logging=False,
    file_name: str = "",
    verbose=False,
) -> Tuple[float, int, str, str, dict]:
    """
    An algorithm for exhaustively enumerating a given set of parameters
    with training a neural network for each configuration of parameters
    and selecting the best one.

    Parameters
    ----------
    in_size: int
       Size of input data
    out_size: int
        Size of output data
    data: tuple
        dataset
    opt: list
        List of optimizers
    loss: list
        list of loss functions
    min_epoch: int
        Starting number of epochs
    max_epoch: int
        Final number of epochs
    epoch_step: int
        Step between `min_epoch` and `max_epoch`
    nn_min_length: int
        Starting number of hidden layers of neural networks
    nn_max_length: int
        Final number of hidden layers of neural networks
    nn_alphabet: list
        List of possible sizes of hidden layers with activations for them
    alphabet_block_size: int
        Number of literals in each `alphabet` symbol that indicate the size of hidden layer
    alphabet_offset: int
        Indicate the minimal number of neurons in hidden layer
    val_data: tuple
        Validation dataset
    logging: bool
        Logging search process to file
    file_name: str
        Path to file for logging
    verbose: bool
        Print additional information to console during the searching
    Returns
    -------
    search_results: tuple[float, int, str, str, dict]
        Results of the algorithm are described by these parameters

        best_loss: float
            The value of the loss function during training of the best neural network
        best_epoch: int
            Number of training epochs for the best neural network
        best_loss_func: str
            Name of the loss function of the best neural network
        best_opt: str
            Name of the optimizer of the best neural network
        best_net: dict
            Best neural network presented as a dictionary
    """
    best_net: dict = dict()
    best_loss: float = 1e6
    best_epoch: int = 0
    best_loss_func: str = ""
    best_opt: str = ""
    time_viewer = MeasureTrainTime()
    for i in range(nn_min_length, nn_max_length + 1):
        if verbose:
            print(i, datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
        codes = product(nn_alphabet, repeat=i)
        for elem in codes:
            code = "".join(elem)
            for epoch in range(min_epoch, max_epoch + 1, epoch_step):
                for opt in opt:
                    for loss_func in loss:
                        curr_loss, curr_val_loss, curr_nn = full_search_step(
                            in_size=in_size,
                            out_size=out_size,
                            code=code,
                            num_epoch=epoch,
                            opt=opt,
                            loss=loss_func,
                            data=data,
                            alphabet_block_size=alphabet_block_size,
                            alphabet_offset=alphabet_offset,
                            val_data=val_data,
                            callbacks=[time_viewer],
                            logging=logging,
                            file_name=file_name,
                        )
                        if best_loss > curr_loss:
                            best_net = curr_nn
                            best_loss = curr_loss
                            best_epoch = epoch
                            best_loss_func = loss_func
                            best_opt = opt
    return best_loss, best_epoch, best_loss_func, best_opt, best_net
