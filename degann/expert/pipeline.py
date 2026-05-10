from typing import Optional, Tuple
from degann.networks import IModel
from degann.search_algorithms import simulated_annealing, grid_search, random_search
from degann.networks.topology.abstracts import ParameterSpace
from .config import ExpertMetaConfig


def _check_threshold(
    result: tuple, threshold: float, val_data: Optional[tuple]
) -> bool:
    """
    Checks if loss passes threshold.
    """

    train_loss = result[0]
    if val_data is None:
        return train_loss <= threshold

    model = IModel.from_dict(result[4])
    model.compile(optimizer=result[3], loss_func=result[2])
    val_loss = model.evaluate(val_data[0], val_data[1], verbose=0)

    return val_loss <= threshold


def execute_pipeline(
    data: tuple,
    params: ParameterSpace,
    config: ExpertMetaConfig,
    run_grid_search: bool = False,
    val_data: Optional[tuple] = None,
    verbose: bool = False
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
    config: ExpertMetaConfig
        Parameters for search algorithms (launch counts, thresholds).
    run_grid_search: bool
        If `True`, then if the random search and the simulated annealing method fail, the grid search will be launched
    val_data: tuple
        Validation dataset
    verbose: bool
        Execution information

    Returns
    -------
    search_result: tuple[float, int, str, str, dict]
        Loss value, epochs, loss function, optimizer, and resulting neural network
    """
    best_result = (10**9, 0, "", "", {})
    best_loss = 1e6

    for i in range(config.launch_count_random_search):
        result = random_search(
            data=data,
            params=params,
            iterations=config.iterations,
            val_data=val_data,
            verbose=verbose,
        )

        if result[0] < best_loss:
            best_loss = result[0]
            best_result = result

        if _check_threshold(result, config.threshold, val_data):
            return result
    print("Random search didn't find any results")

    for i in range(config.launch_count_simulated_annealing):
        result = simulated_annealing(
            data=data,
            params=params,
            val_data=val_data,
            max_iter=config.iterations,
            threshold=config.threshold,
            temperature_method=config.temperature_method,
            distance_method=config.distance_method,
            verbose=verbose,
        )
        train_loss, best_epochs, loss_func, opt, best_net, k = result

        if train_loss < best_loss:
            best_loss = train_loss
            best_result = (train_loss, best_epochs, loss_func, opt, best_net)

        if _check_threshold(result, config.threshold, val_data):
            return train_loss, best_epochs, loss_func, opt, best_net
    print("Simulated annealing didn't find any results")

    if run_grid_search:
        result = grid_search(
            data=data,
            params=params,
            val_data=val_data,
        )
        return result

    return best_result
