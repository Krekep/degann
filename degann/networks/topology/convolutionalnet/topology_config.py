from dataclasses import dataclass, field, InitVar
from typing import Union, Any, Optional
import tensorflow as tf

from degann.networks.topology.base_topology_configs import SingleNetworkParams


@dataclass
class ConvolutionalNetParams(SingleNetworkParams):
    """
    Parameters for a fully-connected (dense) neural network topology.
    """

    # output_size,

    def __post_init__(self, metadata: Optional[dict] = None):
        super().__post_init__(metadata)

        self.net_type = "ConvolutionalNet"

        self.convolution_core_size: tuple[int, int] = (3, 1)
        self.padding_type: str = "same"
        self.convolution_block_types: list[str] = field(
            default_factory=list, metadata={"tunable": True}
        )
        self.convolution_block_sizes: list[int] = field(
            default_factory=list, metadata={"tunable": True}
        )
        self.chunk_size: int = 10
        self.convolutional_activation_func: str = "relu"
        self.dense_activation_func: str = "relu"
        self.is_debug: bool = False
