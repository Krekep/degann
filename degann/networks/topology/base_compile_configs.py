from dataclasses import dataclass, field, InitVar
from typing import Optional

from degann.networks.topology.tuning_utils import TuningMetadata


@dataclass
class BaseCompileParams:
    """
    Base class for compilation parameters applicable to neural network topologies.

    This class is intended to serve as a foundation for more specialized compile parameter
    configurations (e.g., for single-network or GAN topologies).

    Attributes:
        (This base class does not define any fields by itself but acts as a base
        for inheritance.)
    """

    metadata: InitVar[dict | None] = None
    tuning_metadata: Optional[TuningMetadata] = field(default=None, init=False)

    def __post_init__(self, metadata=None):
        self.tuning_metadata = TuningMetadata(type(self))
        self.tuning_metadata.set_metadata(metadata)


@dataclass
class SingleNetworkCompileParams(BaseCompileParams):
    """
    Compilation parameters for a single-network topology.

    Attributes:
        rate (float): Learning rate for the optimizer.
        optimizer (str): Name of the optimizer.
        loss_func (str): Loss function to use.
        metric_funcs (List[str]): List of metric function names.
        run_eagerly (bool): Whether to run eagerly.
    """

    rate: float = 1e-2
    optimizer: str = "SGD"
    loss_func: str = "MeanSquaredError"
    metric_funcs: list[str] = field(
        default_factory=lambda: [
            "root_mean_squared_error",
        ]
    )
    run_eagerly: bool = False

    def get_losses(self):
        return [self.loss_func]

    def get_optimizers(self):
        return [self.optimizer]

    def add_eval_metric(self, metric: str):
        self.metric_funcs.append(metric)
