from dataclasses import dataclass
from typing import Optional

from degann.networks.topology.base_topology_configs import (
    BaseTopologyParams,
)
from degann.networks.topology.densenet.topology_config import DenseNetParams


@dataclass(
    kw_only=True
)  # kw_only used to bypass the attribute organisation of @dataclass
class GANTopologyParams(BaseTopologyParams):
    """
    Parameters for a GAN (Generative Adversarial Network) topology.

    This topology consists of two neural networks:
      - A generator network
      - A discriminator network

    Attributes:
        generator_params (DenseNetParams): Configuration parameters for the generator.
        discriminator_params (DenseNetParams): Configuration parameters for the discriminator.
    """

    generator_params: DenseNetParams
    discriminator_params: DenseNetParams

    def __post_init__(self, metadata: Optional[dict] = None):
        super().__post_init__(metadata)
        # Set the overall GAN configuration based on the generator and discriminator settings.
        # The overall input size is taken from the generator.
        self.input_size = self.generator_params.input_size
        # The overall block_size is constructed by concatenating:
        #   1. The generator's block_size,
        #   2. A bridging layer equal to the discriminator's input_size,
        #   3. The discriminator's block_size.
        self.block_size = (
            self.generator_params.block_size
            + [self.discriminator_params.input_size]
            + self.discriminator_params.block_size
        )
        # The output size is defined by the discriminator's output.
        self.output_size = self.discriminator_params.output_size
        self.net_type = "GAN"
