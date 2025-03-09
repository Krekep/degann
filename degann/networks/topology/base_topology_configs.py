from dataclasses import dataclass, field
from typing import Union, Any
import tensorflow as tf


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

    input_size: int = 1
    block_size: list[int] = field(default_factory=list)
    output_size: int = 1
    name: str = "net"
    net_type: str = "DenseNet"
    is_debug: bool = False


@dataclass
class SingleNetworkParams(BaseTopologyParams):
    """
    Parameters for a single-network topology.

    Attributes:
        activation_func (Union[str, List[str]]): The activation function(s) to use in the network.
        weight (Any): Initializer for the network's weights. (Default: RandomUniform between -1 and 1)
        biases (Any): Initializer for the network's biases. (Default: RandomUniform between -1 and 1)
    """

    activation_func: Union[str, list[str]] = "sigmoid"
    weight: Any = field(
        default_factory=lambda: tf.random_uniform_initializer(minval=-1, maxval=1)
    )
    biases: Any = field(
        default_factory=lambda: tf.random_uniform_initializer(minval=-1, maxval=1)
    )
