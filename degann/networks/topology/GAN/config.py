from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from degann.networks.topology.abstracts import NetConfig
from degann.networks.topology.DenseNet.config import DenseNetConfig


@dataclass
class GANConfig(NetConfig):
    gen_config: DenseNetConfig
    disc_config: DenseNetConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gen_config": self.gen_config.to_dict(),
            "disc_config": self.disc_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(
            gen_config=DenseNetConfig.from_dict(config_dict["gen_config"]),
            disc_config=DenseNetConfig.from_dict(config_dict["disc_config"]),
        )

    @property
    def get_shape(self) -> Tuple[List[int], List[int]]:
        return self.gen_config.get_shape, self.disc_config.get_shape

    @property
    def get_input_size(self) -> Tuple[int, int]:
        return self.gen_config.get_input_size, self.disc_config.get_input_size

    @property
    def get_output_size(self) -> Tuple[int, int]:
        return self.gen_config.get_output_size, self.disc_config.get_output_size

    @property
    def get_loss_func(self) -> Tuple[str, str]:
        return self.gen_config.get_loss_func, self.disc_config.get_loss_func

    @property
    def get_optimizer(self) -> Tuple[str, str]:
        return self.gen_config.get_optimizer, self.disc_config.get_optimizer
