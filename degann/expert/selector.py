from itertools import product

from degann.search_algorithms.nn_code import alph_n_full, alphabet_activations

expert_system_tags = {
    "type": [
        "sin",
        "lin",
        "exp",
        "log",
        "gauss",
        "hyperbol",
        "const",
        "sig",
        "unknown",
    ],
    "precision": ["maximal", "median", "minimal"],
    "work time": ["long", "medium", "short"],
    "data size": ["very small", "small", "median", "big", "auto"],
}

base_sam_parameters = {
        "distance_to_neighbor": "distance_const",
        "dist_offset": 300,
        "dist_scale": 0,
        "temperature_reduction_method": "temperature_lin",
        "temperature_speed": 0,
    }
base_parameters = {
    "launch_count_random_search": 2,
    "launch_count_simulated_annealing": 2,
    "nn_max_length": 4,
    "nn_min_length": 1,
    "nn_alphabet_block_size": 1,
    "nn_alphabet_offset": 8,
    "nn_alphabet": ["".join(elem) for elem in product(alph_n_full, alphabet_activations)],
    "min_train_epoch": 200,
    "max_train_epoch": 500,
    "iteration_count": 5,
    "loss_function": "MaxAbsoluteDeviation",
    "loss_threshold": 1,
    "optimizer": "Adam",
    "simulated_annealing_params": base_sam_parameters,
}


def suggest_parameters(
    data: tuple = None,
    tags: dict[str, str] = None,
) -> dict:
    """
    Builds many parameters of search algorithms using labels supplied by the user,
     describing the requirements for the result and hints on the data.

    Parameters
    ----------
    data: Optional[tuple]
        Dataset
    tags: dict[str, str]
        A subset of tags described in expert_system_tags

    Returns
    -------
    parameters: dict
        Parameters for search algorithms
    """
    if tags is None:
        tags = {
            "type": "unknown",
            "precision": "maximal",
            "work time": "long",
            "data size": "auto",
        }

    parameters = base_parameters.copy()

    simulated_annealing_params = base_sam_parameters.copy()

    if tags["type"] in ["sin", "multidim", "unknown"]:
        parameters["min_train_epoch"] *= 2
        parameters["max_train_epoch"] = 700
        parameters["nn_max_length"] += 1
        parameters["iteration_count"] += 10

        # simulated_annealing_params["distance_to_neighbor"] = [distance_const(300), distance_lin(50, 400)]
        # simulated_annealing_params["temperature_reduction_method"] = [temperature_exp(0.95), temperature_exp(0.95)]
        simulated_annealing_params["distance_to_neighbor"] = "distance_lin"
        simulated_annealing_params["dist_offset"] = 50
        simulated_annealing_params["dist_scale"] = 400
        simulated_annealing_params["temperature_reduction_method"] = "temperature_exp"
        simulated_annealing_params["temperature_speed"] = 0.95

        parameters["launch_count_random_search"] += 2
        parameters["launch_count_simulated_annealing"] = 10
    elif tags["type"] in ["exp", "lin"]:
        parameters["iteration_count"] += 30

    if tags["precision"] == "minimal":
        parameters["loss_threshold"] *= 2
    if tags["precision"] == "median":
        parameters["iteration_count"] = int(10 * parameters["iteration_count"])
    if tags["precision"] == "maximal":
        parameters["loss_threshold"] /= 10
        parameters["iteration_count"] = int(40 * parameters["iteration_count"])
        parameters["max_train_epoch"] = 700

    if tags["work time"] == "short":
        parameters["nn_max_length"] -= 1
        parameters["nn_min_length"] -= 1
    elif tags["work time"] == "long":
        parameters["nn_max_length"] += 1

    if tags["data size"] == "auto":
        if data is None:
            tags["data size"] = "small"
        else:
            size = len(data[0])
            size_id = (
                0 + int(size // 100 > 0) + int(size // 300 > 0) + int(size // 900 > 0)
            )
            tags["data size"] = expert_system_tags["data size"][size_id]
    if tags["data size"] == "very small":
        parameters["min_train_epoch"] *= 2
        parameters["max_train_epoch"] = 700
        parameters["iteration_count"] += 10
        parameters["launch_count_random_search"] += 2
        parameters["launch_count_simulated_annealing"] += 2
    elif tags["data size"] == "small":
        parameters["min_train_epoch"] = int(parameters["min_train_epoch"] * 1.5)
        parameters["iteration_count"] += 10
        parameters["launch_count_random_search"] += 1
        parameters["launch_count_simulated_annealing"] += 1
    elif tags["data size"] == "median":
        parameters["min_train_epoch"] = int(parameters["min_train_epoch"] * 1.25)
        parameters["iteration_count"] += 10
        parameters["launch_count_random_search"] += 1
    elif tags["data size"] == "big":
        parameters["launch_count_random_search"] += 1

    parameters["simulated_annealing_params"] = simulated_annealing_params
    return parameters
