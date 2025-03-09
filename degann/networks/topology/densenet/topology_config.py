from dataclasses import dataclass

from degann.networks.topology.base_topology_configs import SingleNetworkParams


@dataclass
class DenseNetParams(SingleNetworkParams):
    """
    Parameters for a fully-connected (dense) neural network topology.
    """

    def __post_init__(self):
        self.net_type = "DenseNet"
