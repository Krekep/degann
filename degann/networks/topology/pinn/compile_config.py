from dataclasses import dataclass, field
from typing import List

from degann.networks.topology.base_compile_configs import SingleNetworkCompileParams
from degann.networks.topology.pinn.virtual_loss import VirtualLoss


@dataclass
class PINNCompileParams(SingleNetworkCompileParams):
    """
    Compile parameters for a physics-informed neural network topology.
    """

    virtual_functions: List[VirtualLoss] = field(default_factory=list)
