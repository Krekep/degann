from itertools import product
from typing import List

from degann.expert.nn_code import alph_n_full, alphabet_activations
from degann.expert.search_algorithms import (
    distance_const,
    temperature_lin,
    temperature_exp,
    distance_lin,
    random_search_endless,
    simulated_annealing,
    full_search,
)

tags = {
    "type": {
        "sin",
        "lin",
        "exp",
        "log",
        "gauss",
        "hyperbol",
        "const",
        "sig",
        "unknown",
    },
    "precision": {"maximal", "median", "minimal"},
    "work time": {"long", "medium", "short"},
    "data size": {"big", "median", "small", "very small", "auto"},
}


_base_iteration_count = 5
_base_min_epoch_count = 200
_base_max_epoch_count = 700
_base_nn_max_length = 4
_base_nn_min_length = 2
_base_nn_block_size = 1
_base_block_size_offset = 8
_base_alphabet = ["".join(elem) for elem in product(alph_n_full, alphabet_activations)]
_base_random_launch = 2
_base_sim_launch = 2
_base_loss_function = "MaxAbsoluteDeviation"
_base_loss_threshold = 1
_base_optimizer = "Adam"


def suggest_parameters(
    tags: dict[str, str] = None,
) -> dict:
    if tags is None:
        tags = {
            "type": "unknown",
            "precision": "maximal",
            "work time": "long",
            "data size": "auto",
        }

    launch_count_random_search = _base_random_launch
    launch_count_simulated_annealing = _base_sim_launch

    nn_max_length = _base_nn_max_length
    nn_min_length = _base_nn_min_length

    nn_alphabet_block_size = _base_nn_block_size
    nn_alphabet_offset = _base_block_size_offset
    nn_alphabet: list[str] = _base_alphabet

    min_train_epoch = _base_min_epoch_count
    max_train_epoch = _base_max_epoch_count

    iteration_count = _base_iteration_count
    loss_threshold = _base_loss_threshold

    simulated_annealing_params = {
        # "distance_to_neighbor": [distance_const(150)],
        # "temperature_reduction_method": [temperature_lin],
        "distance_to_neighbor": "distance_const",
        "dist_offset": 150,
        "dist_scale": 0,
        "temperature_reduction_method": "temperature_lin",
        "temperature_speed": 0,
    }

    if tags["type"] in ["sin", "unknown"]:
        min_train_epoch *= 2
        nn_max_length += 1
        iteration_count += 10

        # simulated_annealing_params["distance_to_neighbor"] = [distance_const(300), distance_lin(50, 400)]
        # simulated_annealing_params["temperature_reduction_method"] = [temperature_exp(0.95), temperature_exp(0.95)]
        simulated_annealing_params["distance_to_neighbor"] = "distance_lin"
        simulated_annealing_params["dist_offset"] = 50
        simulated_annealing_params["dist_scale"] = 400
        simulated_annealing_params["temperature_reduction_method"] = "temperature_exp"
        simulated_annealing_params["temperature_speed"] = 0.95

        launch_count_random_search += 2
        launch_count_simulated_annealing = 10

    if tags["precision"] == "minimal":
        loss_threshold *= 2
    if tags["precision"] == "median":
        iteration_count = int(10 * iteration_count)
    if tags["precision"] == "maximal":
        loss_threshold /= 2
        iteration_count = int(40 * iteration_count)

    if tags["work time"] == "short":
        nn_max_length -= 1
        nn_min_length -= 1
    elif tags["work time"] == "long":
        nn_max_length += 1

    return {
        "launch_count_random_search": launch_count_random_search,
        "launch_count_simulated_annealing": launch_count_simulated_annealing,
        "nn_max_length": nn_max_length,
        "nn_min_length": nn_min_length,
        "nn_alphabet_block_size": nn_alphabet_block_size,
        "nn_alphabet_offset": nn_alphabet_offset,
        "nn_alphabet": nn_alphabet,
        "min_train_epoch": min_train_epoch,
        "max_train_epoch": max_train_epoch,
        "iteration_count": iteration_count,
        "loss_function": _base_loss_function,
        "loss_threshold": loss_threshold,
        "optimizer": _base_optimizer,
        "simulated_annealing_params": simulated_annealing_params,
    }


def execute_pipeline(
    input_size: int,
    output_size: int,
    data,
    parameters: dict,
    values: dict = None,
    additional_losses: List[str] = None,
    additional_optimizers: List[str] = None,
    val_data=None,
    **kwargs
) -> tuple[float, dict]:
    if values is None:
        values = {
            "loss": parameters["loss_function"],
            "threshold": parameters["loss_threshold"],
            "opt": parameters["optimizer"],
        }
    values["in_size"] = input_size
    values["out_size"] = output_size
    values["data"] = data
    values["val_data"] = val_data

    search_algorithm_arguments = {
        "max_iter": parameters["iteration_count"],
        "min_epoch": parameters["min_train_epoch"],
        "max_epoch": parameters["max_train_epoch"],
        "nn_min_length": parameters["nn_min_length"],
        "nn_max_length": parameters["nn_max_length"],
        "nn_alphabet": parameters["nn_alphabet"],
        "alphabet_block_size": parameters["nn_alphabet_block_size"],
        "alphabet_offset": parameters["nn_alphabet_offset"],
    }

    for i in range(parameters["launch_count_random_search"]):
        (
            train_loss,
            count_epoch,
            loss_function,
            optimizer,
            result_nn,
            last_iteration,
        ) = random_search_endless(**values, **search_algorithm_arguments, **kwargs)
        if train_loss <= values["threshold"]:
            return train_loss, result_nn

    for i in range(parameters["launch_count_simulated_annealing"]):
        (
            train_loss,
            count_epoch,
            loss_function,
            optimizer,
            result_nn,
            last_iteration,
        ) = simulated_annealing(**values, **search_algorithm_arguments, **kwargs)
        if train_loss <= values["threshold"]:
            return train_loss, result_nn

    values["loss"] = [values["loss"]] + additional_losses
    values["opt"] = [values["opt"]] + additional_optimizers
    search_algorithm_arguments.pop("max_iter")
    (
        train_loss,
        count_epoch,
        loss_function,
        optimizer,
        result_nn,
        last_iteration,
    ) = full_search(**values, **search_algorithm_arguments, **kwargs)

    return train_loss, result_nn
