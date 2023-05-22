"""
Provide some helpful functions for DE
"""

from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from src.networks import imodel


def system_ode_from_string(system: str) -> List[List[str]]:
    s = system.split("\n")
    parsed_s = []
    for eq in s:
        parsed_s.append(eq.split())
    return parsed_s


def extract_iv(eq: str) -> Tuple[float, float]:
    """
    Extract initial value for Cauchy problem
    Format --- yi(value1)=value2 without spaces

    Parameters
    ----------
    eq: str
        Condition
    Returns
    -------
    iv: tuple[float, float]
        Pair of point and value at point
    """
    left_v = float(eq[eq.index("(") + 1 : eq.index(")")])
    right_v = float(eq[eq.index("=") + 1 :])
    return left_v, right_v


def build_plot(
    network: list[imodel.IModel] | imodel.IModel,
    interval: Tuple[float, float],
    step: float,
    title="",
    labels: list[str] = None,
    true_data: tuple[list, list] = None,
    is_debug=False,
) -> None:
    """
    Builds a two-dimensional graph on an interval with a given step.

    Parameters
    ----------
    network: list[imodel.IModel]
        Neural network for plotting
    interval: Tuple[float, float]
        The interval for which the plot will be built
    step: float
        Interval coverage step (number of points per interval)
    title: str
        Title for plot
    labels: list[str]
        Labels for each network and true data
    true_data: tuple[list, list]
        x and f(x)
    is_debug: bool
        Console debug output marker
    Returns
    -------
    None
    """
    x = []
    a = interval[0]
    b = interval[1]
    while a <= b:
        x.append(a)
        a += step
    if is_debug:
        print("End build x data")

    if isinstance(network, imodel.IModel):
        network = [network]
    if labels is None:
        labels = [""] * len(network)

    for num_nn, nn in enumerate(network):
        output_size = nn.get_output_size
        y = []
        for i in range(output_size):
            y.append([])

        for i in x:
            temp = nn.feedforward(np.array([[i]]))
            for j in range(output_size):
                y[j].append(temp[0][j].numpy())
            if is_debug and i % (len(x) // 10) == 0:
                print("Build y data is ready ---", i % (len(x) // 10))
        if is_debug:
            print("End build y data from network")
        for i, y_i in enumerate(y):
            plt.plot(x, y_i, "-", label=f"{i} {labels[num_nn]}")
    if true_data is not None:
        if len(labels) == len(network):
            plt.plot(true_data[0], true_data[1], ".", label="function")
        else:
            plt.plot(true_data[0], true_data[1], ".", label=f"{labels[-1]}")
    plt.title(title)
    plt.legend()
    plt.show()
