from dataclasses import dataclass, field
from typing import Optional, List

from degann.networks.topology.base_topology_configs import SingleNetworkParams
from degann.networks.topology.densenet.tf_densenet import TensorflowDenseNet
from degann.networks.topology.pinn.virtual_loss import VirtualLoss


@dataclass
class PINNParams(SingleNetworkParams):
    """
    Parameters for a physics-informed neural network topology.
    """

    def __post_init__(self, metadata: Optional[dict] = None):
        super().__post_init__(metadata)

        self.net_type = "PINN"
