from dataclasses import dataclass

from degann.networks.topology.base_compile_configs import (
    BaseCompileParams,
    SingleNetworkCompileParams,
)


@dataclass(kw_only=True)
class GANCompileParams(BaseCompileParams):
    """
    Compilation parameters for a GAN (Generative Adversarial Network) topology.

    This configuration includes separate compile settings for the generator and discriminator.

    Attributes:
        generator_params (SingleNetworkCompileParams): Compile parameters for the generator.
        discriminator_params (SingleNetworkCompileParams): Compile parameters for the discriminator.
    """

    generator_params: SingleNetworkCompileParams
    discriminator_params: SingleNetworkCompileParams

    def __post_init__(self, metadata=None):
        super().__post_init__(metadata)

    def get_losses(self):
        return (
            self.generator_params.get_losses() + self.discriminator_params.get_losses()
        )

    def get_optimizers(self):
        return (
            self.generator_params.get_optimizers()
            + self.discriminator_params.get_optimizers()
        )

    def add_eval_metric(self, metric: str):
        self.generator_params.add_eval_metric(metric)
