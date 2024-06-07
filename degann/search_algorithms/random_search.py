from itertools import product
from typing import Tuple

from .nn_code import alph_n_full, alphabet_activations, decode
from degann.networks import imodel
from degann.search_algorithms.generate import random_generate
from .utils import update_random_generator, log_to_file


def random_search(
    input_size: int,
    output_size: int,
    data: tuple,
    opt: str,
    loss: str,
    iterations: int,
    min_epoch: int = 100,
    max_epoch: int = 700,
    val_data: tuple = None,
    callbacks: list = None,
    nn_min_length: int = 1,
    nn_max_length: int = 6,
    nn_alphabet: list[str] = [
        "".join(elem) for elem in product(alph_n_full, alphabet_activations)
    ],
    alphabet_block_size: int = 1,
    alphabet_offset: int = 8,
    update_gen_cycle: int = 0,
    logging: bool = False,
    file_name: str = "",
) -> Tuple[float, int, str, str, dict]:
    """
    Algorithm for random search in the space of parameters of neural networks

    Parameters
    ----------
    input_size: int
       Size of input data
    output_size: int
        Size of output data
    data: tuple
        dataset
    opt: str
        Name of optimizer
    loss: str
        Name of loss function
    iterations: int
        The number of iterations that will be carried out within the algorithm before completion
        (specifically, the number of trained neural networks)
    min_epoch: int
        Lower bound of epochs
    max_epoch: int
        Upper bound of epochs
    val_data: tuple
        Validation dataset
    callbacks: list
        Callbacks for neural networks training
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
    update_gen_cycle: int
        Refresh tensorflow random generator per update_gen_cycle
    logging: bool
        Logging search process to file
    file_name: str
        Path to file for logging

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
        curr_best = imodel.IModel(input_size, b, output_size, a + ["linear"])
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
    input_size: int,
    output_size: int,
    data: tuple,
    opt: str,
    loss: str,
    threshold: float,
    max_iter: int = -1,
    min_epoch: int = 100,
    max_epoch: int = 700,
    val_data: tuple = None,
    callbacks: list = None,
    nn_min_length: int = 1,
    nn_max_length: int = 6,
    nn_alphabet: list[str] = [
        "".join(elem) for elem in product(alph_n_full, alphabet_activations)
    ],
    alphabet_block_size: int = 1,
    alphabet_offset: int = 8,
    logging: bool = False,
    file_name: str = "",
    verbose: bool = False,
) -> Tuple[float, int, str, str, dict, int]:
    """
    Algorithm for random search in the space of parameters of neural networks

    Parameters
    ----------
    input_size: int
       Size of input data
    output_size: int
        Size of output data
    data: tuple
        dataset
    opt: str
        Name of optimizer
    loss: str
        Name of loss function
    threshold: float
        Training will stop when the value of the loss function is less than this threshold
    max_iter: int
        Training will stop when the number of iterations of the algorithm exceeds this parameter
    min_epoch: int
        Lower bound of epochs
    max_epoch: int
        Upper bound of epochs
    val_data: tuple
        Validation dataset
    callbacks: list
        Callbacks for neural networks training
    logging: bool
        Logging search process to file
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
    logging: bool
        Logging search process to file
    file_name: str
        Path to file for logging
    verbose: bool
        If True, it will show additional information when searching

    Returns
    -------
    search_results: tuple[float, int, str, str, dict, int]
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
        last_iter: int
            Count of iterations in search algorithm
    """
    nn_loss, nn_epoch, loss_f, opt_n, net = random_search(
        input_size,
        output_size,
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
            input_size,
            output_size,
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
        i += 1
        if nn_loss < best_loss:
            best_net = net
            best_loss = nn_loss
            best_epoch = nn_epoch
    return best_loss, best_epoch, loss, opt, best_net, i
