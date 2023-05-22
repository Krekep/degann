from typing import Callable

from tensorflow import keras

_optimizers: dict = {
    "Adam": keras.optimizers.Adam,
    "AdamW": keras.optimizers.AdamW,
    "Adamax": keras.optimizers.experimental.Adamax,
    "Adadelta": keras.optimizers.experimental.Adadelta,
    "Adafactor": keras.optimizers.Adafactor,
    "Adagrad": keras.optimizers.experimental.Adagrad,
    "Ftrl": keras.optimizers.experimental.Ftrl,
    "Nadam": keras.optimizers.experimental.Nadam,
    "RMSprop": keras.optimizers.experimental.RMSprop,
    "SGD": keras.optimizers.experimental.SGD,
}


def get_optimizer(name: str):
    return _optimizers.get(name)


def get_all_optimizers() -> dict[str, Callable]:
    return _optimizers
