from typing import Callable

import tensorflow as tf
import keras


def perceptron_threshold(x, threshold: float = 1.0):
    return tf.where(x >= threshold, 1.0, 0.0)


_activation_name = {
    # "perceptron_threshold": perceptron_threshold,
    "elu": tf.keras.activations.elu,
    "relu": tf.keras.activations.relu,
    "gelu": tf.keras.activations.gelu,
    "selu": tf.keras.activations.selu,
    "exponential": tf.keras.activations.exponential,
    "linear": tf.keras.activations.linear,
    "sigmoid": tf.keras.activations.sigmoid,
    "hard_sigmoid": tf.keras.activations.hard_sigmoid,
    "swish": tf.keras.activations.swish,
    "tanh": tf.keras.activations.tanh,
    "softplus": tf.keras.activations.softplus,
    "softmax": tf.keras.activations.softmax,
    "softsign": tf.keras.activations.softsign,
}


def get(name: str) -> Callable:
    """
    Get activation function by name
    Parameters
    ----------
    name: str
        name of activation function
    Returns
    -------
    func: Callable
        activation function
    """
    return _activation_name[name]


def get_all_activations() -> dict[str, Callable]:
    """
    Get all activation functions
    Parameters
    ----------

    Returns
    -------
    func: dict[str, Callable]
        dictionary of activation functions
    """
    return _activation_name
