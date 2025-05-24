from dataclasses import dataclass, field
from typing import Optional, List

from degann.networks.topology.base_topology_configs import BaseTopologyParams
from degann.networks.topology.densenet.topology_config import DenseNetParams
from degann.networks.topology.pinn.virtual_loss import VirtualLoss


@dataclass
class PINNParams(BaseTopologyParams):
    """
    Parameters for a physics-informed neural network topology.
    """

    densenet_params: DenseNetParams = field(default_factory=DenseNetParams)

    def __post_init__(self, metadata: Optional[dict] = None):
        super().__post_init__(metadata)

        self.net_type = "PINN"
        self.input_size: int = self.densenet_params.input_size
        self.block_size: list[int] = self.densenet_params.block_size
        self.output_size: int = self.densenet_params.output_size
        self.name: str = self.densenet_params.name
        self.is_debug: bool = self.densenet_params.is_debug
