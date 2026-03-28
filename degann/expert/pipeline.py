from typing import Optional, Tuple

from degann.networks import IModel
from degann.search_algorithms import simulated_annealing, grid_search, random_search
from degann.networks.topology.abstracts import ParameterSpace


def execute_pipeline(
    data: tuple,
    params: ParameterSpace,
    parameters: dict,
    run_grid_search: bool = False,
    val_data: Optional[tuple] = None,
) -> Tuple[float, int, str, str, dict]:
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
    search_result: tuple[float, int, str, str, dict]
        Loss value, epochs, loss function, optimizer, and resulting neural network
    """
    threshold = parameters["loss_threshold"]

    for i in range(parameters["launch_count_random_search"]):
        result = random_search(
            data=data,
            params=params,
            iterations=parameters["iteration_count"],
            val_data=val_data,
            verbose=True,
        )
        if val_data is not None:
            model = IModel.from_dict(result[4])
            model.compile(optimizer=result[3], loss_func=result[2])
            val_loss = model.evaluate(val_data[0], val_data[1], verbose=0)
            if val_loss <= threshold:
                return result
        else:
            if result[0] <= threshold:
                return result
    print("Random search didn't find any results")
    for i in range(parameters["launch_count_simulated_annealing"]):
        result = simulated_annealing(
            data=data,
            params=params,
            val_data=val_data,
            max_iter=parameters["iteration_count"],
            threshold=threshold,
            verbose=True,
        )
        train_loss, best_epochs, loss_func, opt, best_net, k = result
        if val_data is not None:
            model = IModel.from_dict(best_net)
            model.compile(optimizer=opt, loss_func=loss_func)
            val_loss = model.evaluate(val_data[0], val_data[1], verbose=0)

            if val_loss <= threshold:
                return train_loss, best_epochs, loss_func, opt, best_net
        else:
            if train_loss <= threshold:
                return train_loss, best_epochs, loss_func, opt, best_net
    print("Simulated annealing didn't find any results")

    if run_grid_search:
        result = grid_search(
            data=data,
            params=params,
            val_data=val_data,
        )
        return result

    return 10**9, 0, "", "", {}
