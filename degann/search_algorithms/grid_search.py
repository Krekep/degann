from datetime import datetime
from itertools import product
from typing import List, Tuple

from .nn_code import alph_n_full, alphabet_activations, decode
from degann.networks.callbacks import MeasureTrainTime
from degann.networks import imodel
from .utils import update_random_generator, log_to_file


def grid_search_step(
    input_size: int,
    output_size: int,
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
        nn = imodel.IModel(input_size, b, output_size, a + ["linear"])
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


def grid_search(
    input_size: int,
    output_size: int,
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
    input_size: int
       Size of input data
    output_size: int
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
                        curr_loss, curr_val_loss, curr_nn = grid_search_step(
                            input_size=input_size,
                            output_size=output_size,
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
