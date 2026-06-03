import random
from typing import List, Tuple, Iterator
from itertools import product
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
        layer_sizes: List[int],
        activation_funcs: List[str],
        epochs: List[int],
        nn_min_depth: int = 1,
        nn_max_depth: int = 6,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.optimizers = optimizers
        self.losses = losses
        self.epochs = epochs
        self.nn_min_depth = nn_min_depth
        self.nn_max_depth = nn_max_depth
        self.layer_sizes = layer_sizes
        self.activation_funcs = activation_funcs

    def iter_configs(self) -> Iterator[Tuple[DenseNetConfig, int]]:
        """
        Generates all possible configuration from the parameter space.

        Yields
        -------
        Tuple[DenseNetConfig, int]
            Configuration from the parameter space and number of epochs.
        """

        for i in range(self.nn_min_depth, self.nn_max_depth + 1):
            for block_sizes in product(self.layer_sizes, repeat=i):
                for activations in product(self.activation_funcs, repeat=i):
                    for epoch in self.epochs:
                        for opt in self.optimizers:
                            for loss_func in self.losses:
                                config = DenseNetConfig(
                                    layer_sizes=list(block_sizes),
                                    activation_funcs=list(activations) + ["linear"],
                                    optimizer=opt,
                                    loss_func=loss_func,
                                    input_size=self.input_size,
                                    output_size=self.output_size,
                                )
                                yield config, epoch

    def get_random_config(self) -> Iterator[Tuple[DenseNetConfig, int]]:
        """
        Creates random configuration.

        Yields
        -------
        Tuple[DenseNetConfig, int]
            Random configuration and number of epochs.
        """

        block = random.randint(self.nn_min_depth, self.nn_max_depth)
        block_sizes = [random.choice(self.layer_sizes) for _ in range(block)]
        activation_funcs = [random.choice(self.activation_funcs) for _ in range(block)]
        epoch = random.choice(self.epochs)
        opt = random.choice(self.optimizers)
        loss_func = random.choice(self.losses)
        config = DenseNetConfig(
            layer_sizes=block_sizes,
            activation_funcs=activation_funcs + ["linear"],
            optimizer=opt,
            loss_func=loss_func,
            input_size=self.input_size,
            output_size=self.output_size,
        )
        yield config, epoch

    def generate_neighbour_config(
        self, config: DenseNetConfig, num_epochs: int, distance: float
    ) -> Iterator[Tuple[DenseNetConfig, int]]:
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

        Yields
        -------
        Tuple[DenseNetConfig, int]
            Neighbour configuration and number of epochs.
        """

        (
            new_block_sizes,
            new_activations,
            new_optimizer,
            new_epochs,
        ) = generate_neighbour(
            layer_sizes=config.layer_sizes,
            activation_funcs=config.activation_funcs,
            optimizer=config.optimizer,
            num_epochs=num_epochs,
            all_layers=self.layer_sizes,
            all_activations=self.activation_funcs,
            all_optimizers=self.optimizers,
            min_depth=self.nn_min_depth,
            max_depth=self.nn_max_depth,
            min_epoch=min(self.epochs),
            max_epoch=max(self.epochs),
            distance=distance,
        )

        neighbour_config = DenseNetConfig(
            layer_sizes=new_block_sizes,
            activation_funcs=new_activations,
            optimizer=new_optimizer,
            loss_func=config.loss_func,
            input_size=config.input_size,
            output_size=config.output_size,
        )
        yield neighbour_config, new_epochs
