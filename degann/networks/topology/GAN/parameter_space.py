import random
from typing import List, Iterator, Tuple
from itertools import product
from degann.search_algorithms.generate import mutate_block_sizes, mutate_activations
from degann.networks.topology.GAN.config import GANConfig
from degann.networks.topology.DenseNet.config import DenseNetConfig
from degann.networks.topology.abstracts import ParameterSpace


class GANParameterSpace(ParameterSpace):
    def __init__(
        self,
        gen_input_size: int,
        gen_output_size: int,
        gen_layer_sizes: List[int],
        gen_min_depth: int,
        gen_max_depth: int,
        gen_activation_funcs: List[str],
        gen_out_activation: str,
        disc_layer_sizes: List[int],
        disc_min_depth: int,
        disc_max_depth: int,
        disc_activation_funcs: List[str],
        gen_optimizers: List[str],
        disc_optimizers: List[str],
        gen_loss_funcs: List[str],
        disc_loss_funcs: List[str],
        epochs: List[int],
    ):
        self.gen_input_size = gen_input_size
        self.gen_output_size = gen_output_size
        self.gen_layer_sizes = gen_layer_sizes
        self.gen_min_depth = gen_min_depth
        self.gen_max_depth = gen_max_depth
        self.gen_activation_funcs = gen_activation_funcs
        self.gen_out_activation = gen_out_activation

        self.disc_layer_sizes = disc_layer_sizes
        self.disc_min_depth = disc_min_depth
        self.disc_max_depth = disc_max_depth
        self.disc_activation_funcs = disc_activation_funcs

        self.gen_optimizers = gen_optimizers
        self.disc_optimizers = disc_optimizers
        self.gen_loss_funcs = gen_loss_funcs
        self.disc_loss_funcs = disc_loss_funcs

        self.epochs = epochs

    def iter_configs(self) -> Iterator[Tuple[GANConfig, int]]:
        """
        Generates all possible configurations from the parameter space.

        Yields
        ------
        Tuple[GANConfig, int]
            Configuration from the parameter space and number of epochs.
        """

        gen_block_size_options = []
        for depth in range(self.gen_min_depth, self.gen_max_depth + 1):
            for sizes in product(self.gen_layer_sizes, repeat=depth):
                gen_block_size_options.append(list(sizes))

        disc_block_size_options = []
        for depth in range(self.disc_min_depth, self.disc_max_depth + 1):
            for sizes in product(self.disc_layer_sizes, repeat=depth):
                disc_block_size_options.append(list(sizes))

        other_params = [
            self.gen_optimizers,
            self.disc_optimizers,
            self.gen_loss_funcs,
            self.disc_loss_funcs,
        ]

        for gen_bs in gen_block_size_options:
            for disc_bs in disc_block_size_options:
                for params in product(*other_params):
                    gen_opt, disc_opt, gen_lf, disc_lf = params

                    gen_hidden_depth = len(gen_bs)
                    disc_hidden_depth = len(disc_bs)

                    gen_af_options = list(
                        product(self.gen_activation_funcs, repeat=gen_hidden_depth)
                    )
                    disc_af_options = list(
                        product(self.disc_activation_funcs, repeat=disc_hidden_depth)
                    )

                    for gen_af_tuple in gen_af_options:
                        for disc_af_tuple in disc_af_options:
                            gen_activation_funcs = list(gen_af_tuple) + [
                                self.gen_out_activation
                            ]
                            disc_activation_funcs = list(disc_af_tuple) + ["linear"]

                            for epoch in self.epochs:
                                gen_config = DenseNetConfig(
                                    block_size=gen_bs,
                                    activation_func=gen_activation_funcs,
                                    optimizer=gen_opt,
                                    loss_func=gen_lf,
                                    input_size=self.gen_input_size,
                                    output_size=self.gen_output_size,
                                )

                                disc_config = DenseNetConfig(
                                    block_size=disc_bs,
                                    activation_func=disc_activation_funcs,
                                    optimizer=disc_opt,
                                    loss_func=disc_lf,
                                    input_size=self.gen_input_size
                                    + self.gen_output_size,
                                    output_size=1,
                                )

                                config = GANConfig(
                                    gen_config=gen_config,
                                    disc_config=disc_config,
                                )
                                yield config, epoch

    def get_random_config(self) -> Tuple[GANConfig, int]:
        """
        Creates random configuration.

        Returns
        -------
        Tuple[GANConfig, int]
            Random configuration and number of epochs.
        """
        gen_depth = random.randint(self.gen_min_depth, self.gen_max_depth)
        disc_depth = random.randint(self.disc_min_depth, self.disc_max_depth)

        gen_block_sizes = [
            random.choice(self.gen_layer_sizes) for _ in range(gen_depth)
        ]
        disc_block_sizes = [
            random.choice(self.disc_layer_sizes) for _ in range(disc_depth)
        ]

        gen_activation_funcs = [
            random.choice(self.gen_activation_funcs) for _ in range(gen_depth)
        ] + [self.gen_out_activation]
        disc_activation_funcs = [
            random.choice(self.disc_activation_funcs) for _ in range(disc_depth)
        ] + ["linear"]

        epoch = random.choice(self.epochs)

        gen_config = DenseNetConfig(
            block_size=gen_block_sizes,
            activation_func=gen_activation_funcs,
            optimizer=random.choice(self.gen_optimizers),
            loss_func=random.choice(self.gen_loss_funcs),
            input_size=self.gen_input_size,
            output_size=self.gen_output_size,
        )

        disc_config = DenseNetConfig(
            block_size=disc_block_sizes,
            activation_func=disc_activation_funcs,
            optimizer=random.choice(self.disc_optimizers),
            loss_func=random.choice(self.disc_loss_funcs),
            input_size=self.gen_input_size + self.gen_output_size,
            output_size=1,
        )

        config = GANConfig(
            gen_config=gen_config,
            disc_config=disc_config,
        )
        return config, epoch

    def generate_neighbour_config(
        self,
        config: GANConfig,
        num_epochs: int,
        distance: float,
    ) -> Tuple[GANConfig, int]:
        """
        Generate a neighbour configuration for GANConfig.

        Parameters
        ----------
        config: GANConfig
            Original configuration.
        num_epochs: int
            Number of epochs.
        distance: float
            A proxy for mutation strength (higher -> more changes).

        Returns
        -------
        Tuple[GANConfig, int]
            New neighbour configuration and number of epochs.
        """
        mutation_prob = min(0.5, distance / 100.0)

        new_gen_block_sizes = mutate_block_sizes(
            block_sizes=config.gen_config.block_size,
            layer_sizes=self.gen_layer_sizes,
            min_depth=self.gen_min_depth,
            max_depth=self.gen_max_depth,
            mutation_prob=mutation_prob,
        )
        new_gen_activations = mutate_activations(
            activations=config.gen_config.activation_func[:-1]
            + [config.gen_config.activation_func[-1]],
            all_activations=self.gen_activation_funcs,
            mutation_prob=mutation_prob,
        )

        if len(new_gen_activations) > 1:
            new_gen_activations[-1] = config.gen_config.activation_func[-1]
        else:
            new_gen_activations = [config.gen_config.activation_func[-1]]

        new_disc_block_sizes = mutate_block_sizes(
            block_sizes=config.disc_config.block_size,
            layer_sizes=self.disc_layer_sizes,
            min_depth=self.disc_min_depth,
            max_depth=self.disc_max_depth,
            mutation_prob=mutation_prob,
        )
        new_disc_activations = mutate_activations(
            activations=config.disc_config.activation_func[:-1] + ["linear"],
            all_activations=self.disc_activation_funcs,
            mutation_prob=mutation_prob,
        )

        if len(new_disc_activations) > 1:
            new_disc_activations[-1] = "linear"
        else:
            new_disc_activations = ["linear"]

        new_gen_optimizer = config.gen_config.optimizer
        new_disc_optimizer = config.disc_config.optimizer
        new_gen_loss_func = config.gen_config.loss_func
        new_disc_loss_func = config.disc_config.loss_func
        new_num_epoch = num_epochs

        if random.random() < mutation_prob:
            new_gen_optimizer = random.choice(self.gen_optimizers)
        if random.random() < mutation_prob:
            new_disc_optimizer = random.choice(self.disc_optimizers)
        if random.random() < mutation_prob:
            new_gen_loss_func = random.choice(self.gen_loss_funcs)
        if random.random() < mutation_prob:
            new_disc_loss_func = random.choice(self.disc_loss_funcs)
        if random.random() < mutation_prob:
            new_num_epoch = random.choice(self.epochs)

        new_gen_config = DenseNetConfig(
            block_size=new_gen_block_sizes,
            activation_func=new_gen_activations,
            optimizer=new_gen_optimizer,
            loss_func=new_gen_loss_func,
            input_size=config.gen_config.input_size,
            output_size=config.gen_config.output_size,
        )

        new_disc_config = DenseNetConfig(
            block_size=new_disc_block_sizes,
            activation_func=new_disc_activations,
            optimizer=new_disc_optimizer,
            loss_func=new_disc_loss_func,
            input_size=config.disc_config.input_size,
            output_size=config.disc_config.output_size,
        )

        neighbour_config = GANConfig(
            gen_config=new_gen_config,
            disc_config=new_disc_config,
        )

        return neighbour_config, new_num_epoch
