from datetime import datetime
from typing import Tuple
from .utils import update_random_generator
from degann.networks.topology.parameter_space import ParameterSpace


def random_search(
    data: tuple,
    params: ParameterSpace,
    iterations: int,
    threshold: float = 1,
    max_iter: int = 1,
    val_data: tuple = None,
    logging=False,
    file_name: str = "",
    callbacks: list = None,
    verbose: bool = False,
    update_gen_cycle: int = 0,
) -> Tuple[float, int, str, str, dict]:
    """
    Perform random search algorithm to find the best neural network configuration.

    Parameters
    ----------
    data: Tuple[Any, Any]
        Training data.
    params: ParameterSpace
        Parameter space for neural network configurations.
    iterations: int
        Number of iterations for fixed search, -1 for threshold-based search.
    threshold: float
        Loss threshold for early stopping in threshold-based search.
    max_iter: int
        Maximum number of iterations for threshold-based search.
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
    best_epoch = None
    loss = None
    opt = None

    if iterations == -1:
        start_config = params.get_random_config()
        curr_loss, curr_val_loss, curr_nn = params.train(
            config=start_config,
            data=data,
            val_data=val_data,
            logging=logging,
            file_name=file_name,
            callbacks=callbacks,
        )
        i = 1
        while curr_loss > threshold and i != max_iter:
            if verbose:
                print(
                    f"Random search until less than threshold. Last loss = {curr_loss}. Iterations = {i}"
                )
            update_random_generator(i, cycle_size=update_gen_cycle)
            config = params.get_random_config()

            curr_loss, curr_val_loss, curr_nn = params.train(
                config=config,
                data=data,
                val_data=val_data,
                logging=logging,
                file_name=file_name,
                callbacks=callbacks,
            )
            i += 1
            if curr_loss < best_loss:
                best_epoch = config.num_epoch
                best_net = curr_nn
                best_loss = curr_loss
                loss = config.get_loss_func
                opt = config.get_optimizer
    else:
        for i in range(iterations):
            update_random_generator(i, cycle_size=update_gen_cycle)
            if verbose:
                print(
                    f"{i + 1}/{iterations}",
                    datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
                )

            config = params.get_random_config()

            curr_loss, curr_val_loss, curr_nn = params.train(
                config=config,
                data=data,
                val_data=val_data,
                logging=logging,
                file_name=file_name,
                callbacks=callbacks,
            )

            if curr_loss < best_loss:
                best_epoch = config.num_epoch
                best_net = curr_nn
                best_loss = curr_loss
                loss = config.get_loss_func
                opt = config.get_optimizer
    return best_loss, best_epoch, loss, opt, best_net
