from dataclasses import dataclass, asdict
from typing import List, Optional, Dict


@dataclass
class DenseNetConfig:
    block_size: Optional[List[int]]
    activation_func: Optional[List[str]]
    optimizer: str = "SGD"
    loss_func: str = "MeanSquaredError"
    num_epoch: int = 100
    code: Optional[str] = None
    input_size: int = 1
    output_size: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict):
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


@dataclass
class GANConfig:
    gen_input_size: int = 2
    gen_output_size: int = 1
    gen_block_sizes: Optional[List[int]] = None
    gen_activation_funcs: Optional[List[str]] = None
    gen_out_activation: str = "linear"

    disc_input_size: int = 2
    disc_output_size: int = 1
    disc_block_sizes: Optional[List[int]] = None
    disc_activation_funcs: Optional[List[str]] = None
    disc_out_activation: str = "linear"

    gen_optimizer: str = "Adam"
    disc_optimizer: str = "Adam"

    gen_loss_func: str = "MeanSquaredError"
    disc_loss_func: str = "MeanSquaredError"
    num_epoch: int = 30

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(**config_dict)

    @property
    def get_shape(self) -> tuple[List[int], List[int]]:
        return (self.gen_block_sizes if self.gen_block_sizes else [],
                self.disc_block_sizes if self.disc_block_sizes else [])

    @property
    def get_input_size(self) -> tuple[int, int]:
        return self.gen_input_size, self.disc_input_size

    @property
    def get_output_size(self) -> tuple[int, int]:
        return self.gen_output_size, self.disc_output_size
