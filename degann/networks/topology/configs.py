from dataclasses import dataclass
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
