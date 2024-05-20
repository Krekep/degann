from typing import List

from degann import random_search_endless, simulated_annealing, full_search


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
