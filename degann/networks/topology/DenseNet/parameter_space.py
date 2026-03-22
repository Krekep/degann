import random
from typing import List, Tuple, Iterator
from itertools import product
from degann.search_algorithms.nn_code import alph_n_full, alphabet_activations, decode
from degann.search_algorithms.generate import generate_neighbour
from degann.networks.topology.DenseNet.config import DenseNetConfig
from degann.networks.topology.abstracts import ParameterSpace


class DenseNetParameterSpace(ParameterSpace):
    """
    Parameter space for DenseNet.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        optimizers: List[str],
        losses: List[str],
        min_epoch: int = 100,
        max_epoch: int = 700,
        epoch_step: int = 1,
        nn_min_length: int = 1,
        nn_max_length: int = 6,
        nn_alphabet: List[str] = [
            "".join(elem) for elem in product(alph_n_full, alphabet_activations)
        ],
        alphabet_block_size: int = 1,
        alphabet_offset: int = 8,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.optimizers = optimizers
        self.losses = losses
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.epoch_step = epoch_step
        self.nn_min_length = nn_min_length
        self.nn_max_length = nn_max_length
        self.nn_alphabet = nn_alphabet
        self.alphabet_block_size = alphabet_block_size
        self.alphabet_offset = alphabet_offset

    def iter_configs(self) -> Iterator[Tuple[DenseNetConfig, int]]:
        """
        Generates all possible configuration from the parameter space.

        Yields
        -------
        Tuple[DenseNetConfig, int]
            Configuration from the parameter space and number of epochs.
        """

        for i in range(self.nn_min_length, self.nn_max_length + 1):
            codes = product(self.nn_alphabet, repeat=i)
            for elem in codes:
                code = "".join(elem)
                for epoch in range(self.min_epoch, self.max_epoch + 1, self.epoch_step):
                    for opt in self.optimizers:
                        for loss_func in self.losses:
                            b, a = decode(
                                code,
                                block_size=self.alphabet_block_size,
                                offset=self.alphabet_offset,
                            )

                            config = DenseNetConfig(
                                block_size=b,
                                activation_func=a + ["linear"],
                                optimizer=opt,
                                loss_func=loss_func,
                                code=code,
                                input_size=self.input_size,
                                output_size=self.output_size,
                            )
                            yield config, epoch

    def get_random_config(self) -> Tuple[DenseNetConfig, int]:
        """
        Creates random configuration.

        Returns
        -------
        Tuple[DenseNetConfig, int]
            Random configuration and number of epochs.
        """

        block = random.randint(self.nn_min_length, self.nn_max_length)
        code = ""

        for i in range(block):
            code += self.nn_alphabet[random.randint(0, len(self.nn_alphabet) - 1)]
        epoch = random.randint(self.min_epoch, self.max_epoch)
        b, a = decode(
            code, block_size=self.alphabet_block_size, offset=self.alphabet_offset
        )
        opt = random.choice(self.optimizers)
        loss_func = random.choice(self.losses)
        config = DenseNetConfig(
            block_size=b,
            activation_func=a + ["linear"],
            optimizer=opt,
            loss_func=loss_func,
            code=code,
            input_size=self.input_size,
            output_size=self.output_size,
        )
        return config, epoch

    def generate_neighbour_config(
        self, config: DenseNetConfig, num_epochs: int, distance: float
    ) -> Tuple[DenseNetConfig, int]:
        """
        Generate neighbour configuration based on distance value.

        Parameters
        ----------
        config: DenseNetConfig
            Original configuration.
        num_epochs: int
            Number of epochs.
        distance: float
            Distance for generating neighbour configuration.

        Returns
        -------
        Tuple[DenseNetConfig, int]
            Neighbour configuration and number of epochs.
        """

        code = config.code
        parameters = (code, num_epochs)

        new_code_param, new_epoch_param = generate_neighbour(
            alphabet=self.nn_alphabet,
            parameters=parameters,
            distance=int(distance),
            min_epoch=self.min_epoch,
            max_epoch=self.max_epoch,
            min_length=self.nn_min_length,
            max_length=self.nn_max_length,
        )

        new_code = new_code_param.value()
        b, a = decode(
            new_code, block_size=self.alphabet_block_size, offset=self.alphabet_offset
        )

        neighbour_config = DenseNetConfig(
            block_size=b,
            activation_func=a + ["linear"],
            optimizer=config.optimizer,
            loss_func=config.loss_func,
            code=new_code,
            input_size=config.input_size,
            output_size=config.output_size,
        )
        return neighbour_config, new_epoch_param.value()
