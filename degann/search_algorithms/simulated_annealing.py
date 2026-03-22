import math
import random
import copy
from datetime import datetime
from typing import Callable, Tuple, Any, Optional
from .utils import update_random_generator
from degann.networks.topology.abstracts import ParameterSpace, NetConfig
from degann.networks.topology.trainer import train


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


def distance_const(d: float) -> Callable[..., float]:
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


def temperature_exp(alpha: float) -> Callable[[float, Any], float]:
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


def distance_lin(offset: float, multiplier: float) -> Callable[..., float]:
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
    data: tuple,
    params: ParameterSpace,
    val_data: Optional[tuple] = None,
    max_iter: int = 100,
    threshold: float = 1,
    start_config: Optional[NetConfig] = None,
    start_epochs: int = 0,
    temperature_method: Optional[Callable] = None,
    distance_method: Optional[Callable] = None,
    update_gen_cycle: int = 0,
    logging: bool = False,
    file_name: str = "",
    callbacks: Optional[list] = None,
    verbose: bool = False,
) -> Tuple[float, int, str, str, dict, int]:
    """
    Performs a simulated annealing algorithm to find the best neural network configuration.

    Parameters
    ----------
    data: Tuple[Any, Any]
        Training data.
    params: ParameterSpace
        Parameter space for neural network configurations.
    val_data: Tuple[Any, Any]
        Validation data.
    max_iter: int
        Maximum number of iterations.
    threshold: float
        Loss threshold for early stopping.
    start_config: NetConfig
        Starting configuration for the algorithm.
    start_epochs: int
        Starting number of epochs.
    temperature_method: Callable
        Function for temperature calculation.
    distance_method: Callable
        Function for distance calculation.
    update_gen_cycle: int
        Cycle size for random generator update.
    logging: bool
        Flag to enable logging.
    file_name: str
        Name for log files.
    callbacks: List[Any]
        List of training callbacks.
    verbose: bool
        Flag to enable verbose output.

    Returns
    -------
    best_loss: float
        Best training loss achieved.
    best_config: NetConfig
        Best configuration object.
    best_net: dict
        Dictionary representation of the best network.
    k: int
        Number of iterations performed.
    """

    if temperature_method is None:
        temperature_method = temperature_lin

    if distance_method is None:
        distance_method = distance_const(150)

    if start_config is None:
        curr_config, curr_epochs = params.get_random_config()
    else:
        curr_config = copy.deepcopy(start_config)
        if start_epochs == 0:
            curr_epochs = 10
        else:
            curr_epochs = start_epochs

    train_result = train(
        config=curr_config,
        num_epochs=curr_epochs,
        data=data,
        val_data=val_data,
        logging=logging,
        file_name=file_name,
        callbacks=callbacks,
    )
    curr_loss = train_result[0]
    curr_net = train_result[2]

    best_epochs = curr_epochs
    best_loss = curr_loss
    loss_func = curr_config.get_loss_func
    opt = curr_config.get_optimizer
    best_net = curr_net
    k = 0
    t = 1.0

    while k < max_iter and curr_loss > threshold:
        update_random_generator(k, cycle_size=update_gen_cycle)
        if verbose:
            print(f"{k + 1}/{max_iter}", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))

        t = temperature_method(k=k, k_max=max_iter, t=t)
        distance = distance_method(temperature=t)

        neighbour_config, neighbour_epochs = params.generate_neighbour_config(
            curr_config, curr_epochs, distance
        )

        neighbour_result = train(
            config=neighbour_config,
            num_epochs=neighbour_epochs,
            data=data,
            val_data=val_data,
            logging=logging,
            file_name=file_name,
            callbacks=callbacks,
        )
        neighbour_loss = neighbour_result[0]
        neighbour_net = neighbour_result[2]

        if (
            neighbour_loss < curr_loss
            or math.exp((curr_loss - neighbour_loss) / max(t, 1e-8)) > random.random()
        ):
            curr_config = neighbour_config
            curr_epochs = neighbour_epochs
            curr_loss = neighbour_loss
            curr_net = neighbour_net

            if curr_loss < best_loss:
                best_loss = curr_loss
                loss_func = curr_config.get_loss_func
                opt = curr_config.get_optimizer
                best_net = neighbour_net
                best_epochs = curr_epochs

        k += 1

    return best_loss, best_epochs, loss_func, opt, best_net, k
