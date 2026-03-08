from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from degann.networks.topology.abstracts import NetConfig


@dataclass
class DenseNetConfig(NetConfig):
    block_size: Optional[List[int]]
    activation_func: Optional[List[str]]
    optimizer: str = "SGD"
    loss_func: str = "MeanSquaredError"
    code: Optional[str] = None
    input_size: int = 1
    output_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)

    @property
    def get_shape(self) -> List[int]:
        return self.block_size if self.block_size else []

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
