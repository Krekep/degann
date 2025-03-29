import tensorflow as tf

from dataclasses import dataclass, field, InitVar
from typing import Optional, Union

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

    def __post_init__(self, metadata: Optional[dict] = None):
        self.tuning_metadata = TuningMetadata(type(self))
        self.tuning_metadata.set_metadata(metadata)

    def get_losses(self) -> list[list[Union[str, tf.keras.Loss]]]:
        return []

    def get_optimizers(self) -> list[Union[str, tf.keras.Optimizer]]:
        return []

    def add_eval_metric(self, metric: str) -> None:
        return


@dataclass
class SingleNetworkCompileParams(BaseCompileParams):
    """
    Compilation parameters for a single-network topology.

    Attributes:
        rate (float): Learning rate for the optimizer.
        optimizer (str | tf.keras.Optimizer): Name of the optimizer or the optimizer itself.
        loss_func (str | tf.keras.Loss): Name of the loss function or loss function itself.
        metric_funcs (List[str]): List of metric function names.
        run_eagerly (bool): Whether to run eagerly.
    """

    rate: float = 1e-2
    optimizer: Union[str, tf.keras.Optimizer] = field(
        default="SGD", metadata={"tunable": True}
    )
    loss_func: Union[str, tf.keras.Loss] = field(
        default="MeanSquaredError", metadata={"tunable": True}
    )
    metric_funcs: list[str] = field(
        default_factory=lambda: [
            "root_mean_squared_error",
        ]
    )
    run_eagerly: bool = False

    def get_losses(self) -> list[list[Union[str, tf.keras.Loss]]]:
        # This structure can be interpreted as 1 neural network with 1 loss function
        return [[self.loss_func]]

    def get_optimizers(self) -> list[Union[str, tf.keras.Optimizer]]:
        return [self.optimizer]

    def add_eval_metric(self, metric: str) -> None:
        self.metric_funcs.append(metric)
