import math
import random
from itertools import product
from typing import Callable

from .nn_code import alph_n_full, alphabet_activations, decode
from degann.networks import imodel
from degann.search_algorithms.generate import (
    generate_neighbor,
    random_generate,
    choose_neighbor,
)
from .utils import log_to_file, update_random_generator


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

    def t_e(t: float, **kwargs) -> float:
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
