from dataclasses import dataclass
from typing import List, Optional


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
