from typing import Callable, Any


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


def distance_const(d: float, **kwargs) -> Callable[..., float]:
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
