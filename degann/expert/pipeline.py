from degann.search_algorithms import simulated_annealing, grid_search, random_search
from degann.networks.topology.abstracts import ParameterSpace


def execute_pipeline(
    data: tuple,
    params: ParameterSpace,
    parameters: dict,
    run_grid_search: bool = False,
    val_data=None,
) -> tuple[float, dict]:
    """
    This function sequentially launches algorithms for searching the topology of a neural network
    with the passed parameters and returns the resulting neural network.

    Parameters
    ----------
    data: tuple
        Dataset
    params: ParameterSpace
        Pre-configured parameter space for the neural network architecture.
    parameters: dict
        Parameters for search algorithms (launch counts, thresholds).
    run_grid_search: bool
        If `True`, then if the random search and the simulated annealing method fail, the grid search will be launched
    val_data: tuple
        Validation dataset

    Returns
    -------
    search_result: tuple[float, dict]
        Loss value and resulting neural network
    """
    threshold = parameters.get("loss_threshold", 1.0)

    for i in range(parameters["launch_count_random_search"]):
        result = random_search(
            data=data,
            params=params,
            iterations=parameters["iteration_count"],
            val_data=val_data,
            threshold=threshold,
        )
        train_loss, result_nn = result[0], result[4]
        if train_loss <= threshold:
            return train_loss, result_nn
    print("Random search didn't find any results")

    for i in range(parameters["launch_count_simulated_annealing"]):
        result = simulated_annealing(
            data=data,
            params=params,
            val_data=val_data,
            max_iter=parameters["iteration_count"],
            threshold=threshold,
        )
        train_loss, result_nn = result[0], result[2]
        if train_loss <= threshold:
            return train_loss, result_nn
    print("Simulated annealing didn't find any results")

    if run_grid_search:
        result = grid_search(
            data=data,
            params=params,
            val_data=val_data,
        )
        return result[0], result[4]

    return 10**9, {}
