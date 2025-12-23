from degann.networks.topology.parameter_space import ParameterSpace
from datetime import datetime
from typing import Tuple
from degann.networks.callbacks import MeasureTrainTime


def grid_search(
        data: tuple,
        params: ParameterSpace,
        val_data: tuple = None,
        logging=False,
        file_name: str = "",
        verbose=False,
) -> Tuple[float, int, str, str, dict]:
    """
    Perform grid search algorithm to find the best neural network configuration.

    Parameters
    ----------
    data: Tuple[Any, Any]
        Training data.
    params: ParameterSpace
        Parameter space for neural network configurations.
    val_data: Tuple[Any, Any]
        Validation data.
    logging: bool
        Flag to enable logging.
    file_name: str
        Name for log files.
    verbose: bool
        Flag to enable verbose output.

    Returns
    -------
    best_loss: float
        Best training loss achieved.
    best_epoch: int
        Number of epochs for best configuration.
    best_loss_func: str
        Loss function name for best configuration.
    best_opt: str
        Optimizer name for best configuration.
    best_net: dict
        Dictionary representation of the best network.
    """

    best_net: dict = dict()
    best_loss: float = 1e6
    best_epoch: int = 0
    best_loss_func: str = ""
    best_opt: str = ""
    time_viewer = MeasureTrainTime()
    configs = params.create_parameter_space()
    for i, config in enumerate(configs):
        if verbose:
            print(f"{i + 1}/{len(configs)}", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
        curr_loss, curr_val_loss, curr_nn = params.train(
            config=config,
            data=data,
            val_data=val_data,
            logging=logging,
            file_name=file_name,
            callbacks=[time_viewer],
        )

        if best_loss > curr_loss:
            best_net = curr_nn
            best_loss = curr_loss
            best_epoch = config.num_epoch
            best_loss_func = config.get_loss_func
            best_opt = config.get_optimizer
    return best_loss, best_epoch, best_loss_func, best_opt, best_net
