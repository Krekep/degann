import math
import random
from typing import Callable, Tuple, Dict, Any
from .utils import update_random_generator
from degann.networks.topology.parameter_space import ParameterSpace


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
    input_size: int,
    output_size: int,
    data: tuple,
    params: ParameterSpace,
    val_data: tuple = None,
    max_iter: int = 100,
    threshold: float = -1,
    start_config: Dict[str, Any] = None,
    temperature_method: Callable = None,
    distance_method: Callable = None,
    update_gen_cycle: int = 0,
    logging: bool = False,
    file_name: str = "",
    callbacks: list = None,
    verbose: bool = False,
) -> Tuple[float, int, str, str, dict, int]:

    if temperature_method is None:
        temperature_method = temperature_lin

    if distance_method is None:
        distance_method = distance_const(150)

    if start_config is None:
        curr_config = params.get_random_config()
    else:
        curr_config = start_config.copy()

    curr_loss, curr_val_loss, curr_net = params.train(
        config=curr_config,
        input_size=input_size,
        output_size=output_size,
        data=data,
        val_data=val_data,
        logging=logging,
        file_name=file_name,
        callbacks=callbacks,
        update_gen_cycle=update_gen_cycle,
    )

    best_loss = curr_loss
    best_epoch = curr_config["num_epoch"]
    best_loss_func = curr_config["loss_func"]
    best_opt = curr_config["optimizer"]
    best_net = curr_net

    k = 0
    t = 1.0

    while k < max_iter and curr_loss > threshold:
        update_random_generator(k, cycle_size=update_gen_cycle)

        t = temperature_method(k=k, k_max=max_iter, t=t)
        distance = distance_method(temperature=t)

        neighbour_config = params.generate_neighbour_config(curr_config, distance)

        neighbour_loss, neighbour_val_loss, neighbour_net = params.train(
            config=neighbour_config,
            input_size=input_size,
            output_size=output_size,
            data=data,
            val_data=val_data,
            logging=logging,
            file_name=file_name,
            callbacks=callbacks,
            update_gen_cycle=update_gen_cycle,
        )

        if (
            neighbour_loss < curr_loss
            or math.exp((curr_loss - neighbour_loss) / t) > random.random()
        ):
            curr_config = neighbour_config
            curr_loss = neighbour_loss
            curr_val_loss = neighbour_val_loss
            if curr_loss < best_loss:
                best_loss = curr_loss
                best_epoch = neighbour_config["num_epoch"]
                best_loss_func = neighbour_config["loss_func"]
                best_opt = neighbour_config["optimizer"]
                best_net = neighbour_net

        if verbose:
            print(
                f"Iter {k+1}/{max_iter}: loss={curr_loss:.5f}, best={best_loss:.5f}, temp={t:.3f}"
            )

        k += 1

    return best_loss, best_epoch, best_loss_func, best_opt, best_net, k
