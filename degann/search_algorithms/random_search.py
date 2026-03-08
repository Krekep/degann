from datetime import datetime
from typing import Tuple, Optional
from .utils import update_random_generator
from degann.networks.topology.abstracts import ParameterSpace


def random_search(
    data: tuple,
    params: ParameterSpace,
    iterations: int,
    val_data: Optional[tuple] = None,
    logging: bool = False,
    file_name: str = "",
    callbacks: Optional[list] = None,
    verbose: bool = False,
    update_gen_cycle: int = 0,
) -> Tuple[float, int, str, str, dict]:
    """
    Perform random search with fixed number of iterations.

    Parameters
    ----------
    data: Tuple[Any, Any]
        Training data.
    params: ParameterSpace
        Parameter space for neural network configurations.
    iterations: int
        Number of iterations.
    val_data: Tuple[Any, Any]
        Validation data.
    logging: bool
        Flag to enable logging.
    file_name: str
        Name for log files.
    callbacks: List[Any]
        List of training callbacks.
    verbose: bool
        Flag to enable verbose output.
    update_gen_cycle: int
        Cycle size for random generator update.

    Returns
    -------
    best_loss: float
        Best training loss achieved.
    best_epoch: int
        Number of epochs for best configuration.
    loss: str
        Loss function name for best configuration.
    opt: str
        Optimizer name for best configuration.
    best_net: dict
        Dictionary representation of the best network.
    """
    best_net = None
    best_loss = 1e6
    best_epoch = 0
    loss = ""
    opt = ""

    for i in range(iterations):
        update_random_generator(i, cycle_size=update_gen_cycle)
        if verbose:
            print(
                f"{i + 1}/{iterations}",
                datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            )

        config, epoch = params.get_random_config()

        curr_loss, curr_val_loss, curr_nn = params.train(
            config=config,
            num_epochs=epoch,
            data=data,
            val_data=val_data,
            logging=logging,
            file_name=file_name,
            callbacks=callbacks,
        )

        if curr_loss < best_loss:
            best_epoch = epoch
            best_net = curr_nn
            best_loss = curr_loss
            loss = config.get_loss_func
            opt = config.get_optimizer

    return best_loss, best_epoch, loss, opt, best_net


def random_search_threshold(
    data: tuple,
    params: ParameterSpace,
    threshold: float = 1,
    max_iter: int = 1,
    val_data: Optional[tuple] = None,
    logging: bool = False,
    file_name: str = "",
    callbacks: Optional[list] = None,
    verbose: bool = False,
    update_gen_cycle: int = 0,
) -> Tuple[float, int, str, str, dict]:
    """
    Perform random search until loss is below threshold.

    Parameters
    ----------
    data: Tuple[Any, Any]
        Training data.
    params: ParameterSpace
        Parameter space for neural network configurations.
    threshold: float
        Loss threshold for early stopping.
    max_iter: int
        Maximum number of iterations.
    val_data: Tuple[Any, Any]
        Validation data.
    logging: bool
        Flag to enable logging.
    file_name: str
        Name for log files.
    callbacks: List[Any]
        List of training callbacks.
    verbose: bool
        Flag to enable verbose output.
    update_gen_cycle: int
        Cycle size for random generator update.

    Returns
    -------
    best_loss: float
        Best training loss achieved.
    best_epoch: int
        Number of epochs for best configuration.
    loss: str
        Loss function name for best configuration.
    opt: str
        Optimizer name for best configuration.
    best_net: dict
        Dictionary representation of the best network.
    """
    best_net = None
    best_loss = 1e6
    best_epoch = 0
    loss = ""
    opt = ""

    config, epoch = params.get_random_config()
    curr_loss, curr_val_loss, curr_nn = params.train(
        config=config,
        num_epochs=epoch,
        data=data,
        val_data=val_data,
        logging=logging,
        file_name=file_name,
        callbacks=callbacks,
    )

    i = 1
    while curr_loss > threshold and i < max_iter:
        if verbose:
            print(
                f"Random search until less than threshold. Last loss = {curr_loss}. Iterations = {i}"
            )
        update_random_generator(i, cycle_size=update_gen_cycle)
        config, epoch = params.get_random_config()

        curr_loss, curr_val_loss, curr_nn = params.train(
            config=config,
            num_epochs=epoch,
            data=data,
            val_data=val_data,
            logging=logging,
            file_name=file_name,
            callbacks=callbacks,
        )
        i += 1

        if curr_loss < best_loss:
            best_epoch = epoch
            best_net = curr_nn
            best_loss = curr_loss
            loss = config.get_loss_func
            opt = config.get_optimizer

    return best_loss, best_epoch, loss, opt, best_net
