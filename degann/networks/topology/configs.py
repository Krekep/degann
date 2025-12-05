from typing import List, Optional


class DenseNetConfig:

    def __init__(
            self,
            block_size: Optional[List[int]],
            activation_func: Optional[List[str]],
            optimizer: str = "SGD",
            loss_func: str = "MeanSquaredError",
            num_epoch: int = 100,
            code: Optional[str] = None,
            input_size: int = 1,
            output_size: int = 1
    ):
        self.block_size = block_size
        self.activation_func = activation_func
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.num_epoch = num_epoch
        self.code = code
        self.input_size = input_size
        self.output_size = output_size
