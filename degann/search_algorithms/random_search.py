from datetime import datetime
from typing import Tuple, Optional
from .utils import update_random_generator
from degann.networks.topology.abstracts import ParameterSpace
from degann.networks.topology.trainer import train


def random_search(
    data: tuple,
    params: ParameterSpace,
    iterations: int,
    threshold: Optional[float] = None,
    max_iter: int = 100,
    val_data: Optional[tuple] = None,
    logging: bool = False,
    file_name: str = "",
    callbacks: Optional[list] = None,
    verbose: bool = False,
    update_gen_cycle: int = 0,
) -> Tuple[float, int, str, str, dict]:
    """
    Perform random search for the best neural network configuration.

    Parameters
    ----------
    data: Tuple[Any, Any]
        Training data.
    params: ParameterSpace
        Parameter space for neural network configurations.
    iterations: int
        Number of iterations to perform.
    threshold: Optional[float]
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
    iters = max_iter if (iterations == 1 and threshold is not None) else iterations

    best_net = None
    best_loss = 1e6
    best_epoch = 0
    loss = ""
    opt = ""
    i = 0

    while i < iters or (max_iter == 0 and threshold is not None):
        update_random_generator(i, cycle_size=update_gen_cycle)
        if verbose:
            out = f"{i + 1}" if max_iter == 0 else f"{i + 1}/{iters}"
            print(
                out,
                datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            )

        config, epoch = next(params.get_random_config())

        curr_loss, curr_val_loss, curr_nn = train(
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

        if threshold is not None and curr_loss <= threshold:
            if verbose:
                print(f"Threshold {threshold} reached at iteration {i}.")
            break
        i += 1

    return best_loss, best_epoch, loss, opt, best_net
