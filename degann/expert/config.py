from dataclasses import dataclass
from typing import List, Optional, Callable


@dataclass
class ExpertMetaConfig:
    launch_count_random_search: int = 2
    launch_count_simulated_annealing: int = 2
    iterations: int = 5
    threshold: float = 1.0
    verbose: bool = False
    temperature_method: Optional[Callable] = None
    distance_method: Optional[Callable] = None


@dataclass
class ExpertSpaceConfig:
    nn_min_depth: int = 1
    nn_max_depth: int = 4
    min_epoch: int = 200
    max_epoch: int = 500
    epoch_step: int = 100
    layer_sizes: List[int] = None
