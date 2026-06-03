from degann.expert.tags import (
    EquationType,
    ExpertSystemTags,
    DataSize,
    ModelPredictTime,
    RequiredModelPrecision,
)
from typing import Optional, Tuple
from degann.expert.config import ExpertMetaConfig, ExpertSpaceConfig
from degann.search_algorithms.simulated_annealing_functions import (
    temperature_lin,
    temperature_exp,
    distance_const,
    distance_lin,
)


def suggest_parameters(
    data: Optional[tuple] = None,
    tags: Optional[ExpertSystemTags] = None,
) -> Tuple[ExpertMetaConfig, ExpertSpaceConfig]:
    """
    Builds many parameters of search algorithms using labels supplied by the user,
     describing the requirements for the result and hints on the data.

    Parameters
    ----------
    data: Optional[tuple]
        Dataset
    tags: Optional[ExpertSystemTags]
        A subset of tags described in expert_system_tags

    Returns
    -------
    parameters: Tuple[ExpertMetaConfig, ExpertSpaceConfig]
        Parameters for search algorithms and parameter space
    """

    if tags is None:
        tags = ExpertSystemTags()

    if tags.data_size == DataSize.AUTO:
        if data is None:
            tags.data_size = DataSize.SMALL
        else:
            size = len(data[0])
            size_id = (
                0 + int(size // 100 > 0) + int(size // 300 > 0) + int(size // 900 > 0)
            )
            match size_id:
                case 0:
                    tags.data_size = DataSize.VERY_SMALL
                case 1:
                    tags.data_size = DataSize.SMALL
                case 2:
                    tags.data_size = DataSize.MEDIAN
                case 3:
                    tags.data_size = DataSize.BIG

    meta = ExpertMetaConfig()
    space = ExpertSpaceConfig()
    space.layer_sizes = [8, 16, 32]

    if tags.equation_type in [
        EquationType.SIN,
        EquationType.MULTIDIM,
        EquationType.UNKNOWN,
    ]:
        space.min_epoch *= 2
        space.max_epoch *= 2
        space.nn_max_depth += 1
        space.nn_min_depth = max(space.nn_min_depth, 3)

        meta.iterations += 10
        meta.distance_method = distance_lin(50, 400)
        meta.temperature_method = temperature_exp(0.95)
        meta.launch_count_random_search += 2
        meta.launch_count_simulated_annealing = 10

    elif tags.equation_type in [EquationType.EXP, EquationType.LIN]:
        meta.iterations += 20

    if tags.model_precision == RequiredModelPrecision.MINIMAL:
        meta.threshold *= 2
        space.nn_min_depth = 1
        space.layer_sizes = [x // 2 for x in space.layer_sizes]

    if tags.model_precision == RequiredModelPrecision.MEDIAN:
        meta.iterations += 20
        space.nn_min_depth = max(space.nn_min_depth, 3)

    if tags.model_precision == RequiredModelPrecision.MAXIMAL:
        meta.threshold /= 10
        meta.iterations += 50
        space.nn_min_depth = max(space.nn_min_depth, 5)
        space.max_epoch = max(space.max_epoch * 2, 400)

    if tags.predict_time == ModelPredictTime.SHORT:
        space.nn_min_depth = 1
        space.nn_max_depth = max(space.nn_min_depth, space.nn_max_depth - 1)
        if len(space.layer_sizes) > 2:
            space.layer_sizes = space.layer_sizes[:-1]

    elif tags.predict_time == ModelPredictTime.LONG:
        space.nn_min_depth = max(space.nn_min_depth, 5)
        space.nn_max_depth = max(space.nn_min_depth, space.nn_max_depth + 1)
        space.layer_sizes.append(space.layer_sizes[-1] * 2)

    if tags.data_size == DataSize.VERY_SMALL:
        space.min_epoch *= 2
        space.max_epoch = max(space.max_epoch * 2, 400)
        space.nn_min_depth = max(space.nn_min_depth, 2)

        meta.iterations += 10
        meta.launch_count_random_search += 2
        meta.launch_count_simulated_annealing += 2

    elif tags.data_size == DataSize.SMALL:
        space.min_epoch = int(space.min_epoch * 1.5)
        space.max_epoch = max(space.min_epoch + 1, space.max_epoch)
        space.nn_min_depth = max(space.nn_min_depth, 3)

        meta.iterations += 10
        meta.launch_count_random_search += 1
        meta.launch_count_simulated_annealing += 1

    elif tags.data_size == DataSize.MEDIAN:
        space.min_epoch = int(space.min_epoch * 1.25)
        space.max_epoch = max(space.min_epoch + 1, space.max_epoch)

        meta.iterations += 10
        meta.launch_count_random_search += 1

    elif tags.data_size == DataSize.BIG:
        space.nn_min_depth = max(space.nn_min_depth, 5)

        meta.launch_count_random_search += 1

    return meta, space
