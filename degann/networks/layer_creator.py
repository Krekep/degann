from collections import defaultdict

import keras.initializers
import numpy as np
from tensorflow import Tensor

from degann.networks.layers.dense import MyDense


def create(
    inp_size,
    shape,
    activation="linear",
    weight=keras.initializers.get("ones"),
    bias=keras.initializers.get("zeros"),
    layer_type="Dense",
    is_debug=False,
    **kwargs
) -> object:
    """
    Create layer by parameters

    Parameters
    ----------
    inp_size: int
        layer input size
    shape: int
        amount of neurons in layer
    activation: str
        activation function for neurons
    weight
    bias
    layer_type: str
        type of layer for create
    is_debug: bool
    kwargs

    Returns
    -------
    layer
        Created layer
    """
    layer = _create_functions[layer_type](
        inp_size, shape, activation, weight, bias, is_debug=is_debug, **kwargs
    )
    return layer


def create_dense(
    inp_size,
    shape,
    activation="linear",
    weight=keras.initializers.get("ones"),
    bias=keras.initializers.get("zeros"),
    **kwargs
) -> MyDense:
    """
    Create dense layer by parameters

    Parameters
    ----------
    inp_size: int
        layer input size
    shape: int
        amount of neurons in layer
    activation: str
        activation function for neurons
    weight
    bias
    kwargs

    Returns
    -------
    layer
        Created dense layer
    """
    layer = create(
        inp_size, shape, activation, weight, bias, layer_type="Dense", **kwargs
    )
    return layer


def from_dict(config):
    """
    Restore layer from dictionary of parameters

    Parameters
    ----------
    config: dict

    Returns
    -------
    layer
        Restored layer
    """
    res = create(
        inp_size=config["inp_size"],
        shape=config["shape"],
        layer_type=config["layer_type"],
    )
    res.from_dict(config)

    return res


_create_functions = defaultdict(lambda: MyDense)
_create_functions["Dense"] = MyDense
