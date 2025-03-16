from dataclasses import dataclass, field, InitVar
from typing import Union, Any, Optional
import tensorflow as tf

from degann.networks.topology.tuning_utils import TuningMetadata


@dataclass
class BaseTopologyParams:
    """
    Base class for common neural network topology parameters.

    This class holds the core parameters that define the structure of a neural network,
    such as input size, block (hidden layer) sizes, and output size.

    Attributes:
        input_size (int): Size of the input vector.
        block_size (List[int]): List of neuron counts for each hidden layer.
        output_size (int): Size of the output vector.
        name (str): Name identifier for the network.
        net_type (str): Type identifier for the network (e.g., "DenseNet").
        is_debug (bool): Flag to enable debugging mode.
    """

    metadata: InitVar[dict | None] = None
    tuning_metadata: Optional[TuningMetadata] = field(default=None, init=False)

    input_size: int = 1
    block_size: list[int] = field(default_factory=list, metadata={"tunable": True})
    output_size: int = 1
    name: str = "net"
    net_type: str = field(default="DenseNet", init=False)
    is_debug: bool = False

    def __post_init__(self, metadata: Optional[dict] = None):
        self.tuning_metadata = TuningMetadata(type(self))
        self.tuning_metadata.set_metadata(metadata)


@dataclass
class SingleNetworkParams(BaseTopologyParams):
    """
    Parameters for a single-network topology.

    Attributes:
        activation_func (Union[str, List[str]]): The activation function(s) to use in the network.
        weight (Any): Initializer for the network's weights. (Default: RandomUniform between -1 and 1)
        biases (Any): Initializer for the network's biases. (Default: RandomUniform between -1 and 1)
    """

    activation_func: Union[str, list[str]] = field(
        default="sigmoid", metadata={"tunable": True}
    )
    weight: Any = field(
        default_factory=lambda: tf.random_uniform_initializer(minval=-1, maxval=1)
    )
    biases: Any = field(
        default_factory=lambda: tf.random_uniform_initializer(minval=-1, maxval=1)
    )
