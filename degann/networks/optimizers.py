from typing import Callable

from tensorflow import keras

_optimizers: dict = {"Adam": keras.optimizers.Adam}


def get_optimizer(name: str):
    return _optimizers.get(name)


def get_all_optimizers() -> dict[str, Callable]:
    return _optimizers
