from dataclasses import dataclass
from typing import Optional

from degann.networks.topology.base_compile_configs import SingleNetworkCompileParams


@dataclass
class DenseNetCompileParams(SingleNetworkCompileParams):
    """
    Compile parameters for a fully-connected (dense) neural network topology.
    """

    def __post_init__(self, metadata: Optional[dict] = None):
        super().__post_init__(metadata)
