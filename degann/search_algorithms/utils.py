import csv
import random

import numpy.random
import tensorflow as tf

_algorithms_for_random_generator = {0: "auto_select", 1: "philox", 2: "threefry"}
tf.random.set_global_generator(numpy.random.default_rng())


def update_random_generator(curr_iter: int, cycle_size: int = 0) -> None:
    """
    Set global tensorflow random generator to random state every *cycle_size* times

    Parameters
    ----------
    curr_iter: int
        Counter showing whether it's time to update the random number generator
    cycle_size: int
        How often should we update random number generator (if not positive, then the generator does not change)

    Returns
    -------
    """
    if False and cycle_size > 0 and curr_iter % cycle_size == 0:
        new_g = tf.random.Generator.from_non_deterministic_state(
            alg=_algorithms_for_random_generator[
                random.randint(0, len(_algorithms_for_random_generator) - 1)
            ]
        )
        tf.random.set_global_generator(new_g)
    else:
        pass


def log_to_file(history: dict, fn: str):
    with open(
        f"./{fn}.csv",
        "a",
        newline="",
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerows(zip(*history.values()))
