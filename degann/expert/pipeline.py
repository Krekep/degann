from typing import List

from degann.search_algorithms import (
    simulated_annealing,
    grid_search,
    random_search
)
from degann.networks.topology.parameter_space import DenseNetParameterSpace


def execute_pipeline(
    input_size: int,
    output_size: int,
    data: tuple,
    parameters: dict,
    values: dict = None,
    run_grid_search: bool = False,
    additional_losses: List[str] = None,
    additional_optimizers: List[str] = None,
    val_data=None
) -> tuple[float, dict]:
    """
    This function sequentially launches algorithms for searching the topology of a neural network
    with the passed parameters and returns the resulting neural network.

    Parameters
    ----------
    input_size: int
        Feature vector size
    output_size: int
        Value vector size
    data: tuple
        Dataset
    parameters: dict
        Parameters for search algorithms
    values: dict
        Parameters for creating and training neural networks
    run_grid_search: bool
        If `True`, then if the random search and the simulated annealing method fail, the grid search will be launched
    additional_losses: list[str]
        Additional losses for grid search
    additional_optimizers: list[str]
        Additional optimizers for grid search
    val_data: tuple
        Validation dataset

    Returns
    -------
    search_result: tuple[float, dict]
        Loss value and resulting neural network
    """
    if values is None:
        values = {
            "loss": parameters["loss_function"],
            "threshold": parameters["loss_threshold"],
            "opt": parameters["optimizer"],
        }
    values["input_size"] = input_size
    values["output_size"] = output_size
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

    params = DenseNetParameterSpace(
        optimizers=values["opt"],
        loss=values["loss"],
        min_epoch=search_algorithm_arguments["min_epoch"],
        max_epoch=search_algorithm_arguments["max_epoch"],
        nn_min_length=search_algorithm_arguments["nn_min_length"],
        nn_max_length=search_algorithm_arguments["nn_max_length"],
        nn_alphabet=search_algorithm_arguments["nn_alphabet"],
        alphabet_block_size=search_algorithm_arguments["alphabet_block_size"],
        alphabet_offset=search_algorithm_arguments["alphabet_offset"]
    )

    for i in range(parameters["launch_count_random_search"]):
        (
            train_loss,
            count_epoch,
            loss_function,
            optimizer,
            result_nn,
        ) = random_search(
            input_size=values["input_size"],
            output_size=values["output_size"],
            data=values["data"],
            params=params,
            iterations=-1,
            threshold=values["threshold"],
            max_iter=search_algorithm_arguments["max_iter"],
            val_data=values["val_data"]
        )
        if train_loss <= values["threshold"]:
            return train_loss, result_nn
    print("Random search didn't find any results")

    for i in range(parameters["launch_count_simulated_annealing"]):
        (
            train_loss,
            count_epoch,
            loss_function,
            optimizer,
            result_nn,
            last_iteration,
        ) = simulated_annealing(
            input_size=values["input_size"],
            output_size=values["output_size"],
            data=values["data"],
            params=params,
            val_data=values["val_data"],
            max_iter=search_algorithm_arguments["max_iter"],
            threshold=values["threshold"]
        )
        if train_loss <= values["threshold"]:
            return train_loss, result_nn
    print("Simulated annealing didn't find any results")

    if run_grid_search:
        values["loss"] = [values["loss"]] + additional_losses
        values["opt"] = [values["opt"]] + additional_optimizers
        search_algorithm_arguments.pop("max_iter")
        (
            train_loss,
            count_epoch,
            loss_function,
            optimizer,
            result_nn
        ) = grid_search(
            input_size=values["input_size"],
            output_size=values["output_size"],
            data=values["data"],
            params=params,
            val_data=values["val_data"]
        )

        return train_loss, result_nn

    return 10**9, {}
