from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from degann.networks.topology.abstracts import NetConfig


@dataclass
class DenseNetConfig(NetConfig):
    layer_sizes: List[int]
    activation_funcs: List[str]
    optimizer: str = "SGD"
    loss_func: str = "MeanSquaredError"
    input_size: int = 1
    output_size: int = 1
    net_type: str = "DenseNet"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)

    @property
    def get_shape(self) -> List[int]:
        return self.layer_sizes if self.layer_sizes else []

    @property
    def get_activation_funcs(self) -> List[str]:
        return self.activation_funcs if self.activation_funcs else []

    @property
    def get_input_size(self) -> int:
        return self.input_size

    @property
    def get_output_size(self) -> int:
        return self.output_size

    @property
    def get_loss_func(self) -> str:
        return self.loss_func

    @property
    def get_optimizer(self) -> str:
        return self.optimizer

    @property
    def get_compile_kwargs(self) -> Dict[str, Any]:
        return {"optimizer": self.optimizer, "loss_func": self.loss_func}
