from dataclasses import dataclass, field
from typing import List, Callable
import numpy as np
import tensorflow as tf

from degann.networks.topology.base_compile_configs import BaseCompileParams
from degann.networks.topology.pinn.virtual_loss import VirtualLoss
from degann.networks.topology.densenet.compile_config import DenseNetCompileParams


@dataclass
class PINNCompileParams(BaseCompileParams):
    """
    Compile parameters for a physics-informed neural network topology.
    """

    densenet_compile_params: DenseNetCompileParams = field(
        default_factory=DenseNetCompileParams
    )
    virtual_functions: List[VirtualLoss] = field(default_factory=list)
    collocational_points_generator: Callable[
        [], tf.Tensor
    ] = lambda: tf.convert_to_tensor(
        np.linspace(0, 1, 100).reshape(-1, 1), dtype=tf.float32
    )

    def add_eval_metric(self, metric: str) -> None:
        self.densenet_compile_params.add_eval_metric(metric)
