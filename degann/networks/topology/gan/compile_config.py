import tensorflow as tf

from dataclasses import dataclass
from typing import Union, Optional

from degann.networks.topology.base_compile_configs import (
    BaseCompileParams,
)
from degann.networks.topology.densenet.compile_config import DenseNetCompileParams


@dataclass(kw_only=True)
class GANCompileParams(BaseCompileParams):
    """
    Compilation parameters for a GAN (Generative Adversarial Network) topology.

    This configuration includes separate compile settings for the generator and discriminator.

    Attributes:
        generator_params (DenseNetCompileParams): Compile parameters for the generator.
        discriminator_params (DenseNetCompileParams): Compile parameters for the discriminator.
    """

    generator_params: DenseNetCompileParams
    discriminator_params: DenseNetCompileParams

    def __post_init__(self, metadata: Optional[dict] = None):
        super().__post_init__(metadata)

    def get_losses(self) -> list[list[Union[str, tf.keras.Loss]]]:
        return (
            self.generator_params.get_losses() + self.discriminator_params.get_losses()
        )

    def get_optimizers(self) -> list[Union[str, tf.keras.Optimizer]]:
        return (
            self.generator_params.get_optimizers()
            + self.discriminator_params.get_optimizers()
        )

    def add_eval_metric(self, metric: str) -> None:
        self.generator_params.add_eval_metric(metric)
