from typing import List, Iterator, Tuple
from degann.networks.topology.GAN.config import GANConfig
from degann.networks.topology.DenseNet.config import DenseNetConfig
from degann.networks.topology.DenseNet.parameter_space import DenseNetParameterSpace
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
        self.gen_out_activation = gen_out_activation
        self.epochs = epochs

        self.gen_space = DenseNetParameterSpace(
            input_size=gen_input_size,
            output_size=gen_output_size,
            optimizers=gen_optimizers,
            losses=gen_loss_funcs,
            layer_sizes=gen_layer_sizes,
            activation_funcs=gen_activation_funcs,
            epochs=epochs,
            nn_min_depth=gen_min_depth,
            nn_max_depth=gen_max_depth,
        )

        self.disc_space = DenseNetParameterSpace(
            input_size=gen_input_size + gen_output_size,
            output_size=1,
            optimizers=disc_optimizers,
            losses=disc_loss_funcs,
            layer_sizes=disc_layer_sizes,
            activation_funcs=disc_activation_funcs,
            epochs=epochs,
            nn_min_depth=disc_min_depth,
            nn_max_depth=disc_max_depth,
        )

    def iter_configs(self) -> Iterator[Tuple[GANConfig, int]]:
        """
        Generates all possible configurations from the parameter space.

        Yields
        ------
        Tuple[GANConfig, int]
            Configuration from the parameter space and number of epochs.
        """
        for epoch in self.epochs:
            for gen_config, _ in self.gen_space.iter_configs():
                for disc_config, _ in self.disc_space.iter_configs():
                    gen_activation_funcs = gen_config.activation_funcs[:-1] + [
                        self.gen_out_activation
                    ]
                    gen_config = DenseNetConfig(
                        layer_sizes=gen_config.layer_sizes,
                        activation_funcs=gen_activation_funcs,
                        optimizer=gen_config.optimizer,
                        loss_func=gen_config.loss_func,
                        input_size=gen_config.input_size,
                        output_size=gen_config.output_size,
                    )

                    disc_activation_funcs = disc_config.activation_funcs[:-1] + [
                        "linear"
                    ]
                    disc_config = DenseNetConfig(
                        layer_sizes=disc_config.layer_sizes,
                        activation_funcs=disc_activation_funcs,
                        optimizer=disc_config.optimizer,
                        loss_func=disc_config.loss_func,
                        input_size=self.gen_input_size + self.gen_output_size,
                        output_size=1,
                    )

                    config = GANConfig(
                        gen_config=gen_config,
                        disc_config=disc_config,
                    )
                    yield config, epoch

    def get_random_config(self) -> Iterator[Tuple[GANConfig, int]]:
        """
        Creates random configuration.

        Yields
        -------
        Tuple[GANConfig, int]
            Random configuration and number of epochs.
        """
        gen_config, _ = next(self.gen_space.get_random_config())
        disc_config, epoch = next(self.disc_space.get_random_config())

        gen_activation_funcs = gen_config.activation_funcs[:-1] + [
            self.gen_out_activation
        ]
        gen_config = DenseNetConfig(
            layer_sizes=gen_config.layer_sizes,
            activation_funcs=gen_activation_funcs,
            optimizer=gen_config.optimizer,
            loss_func=gen_config.loss_func,
            input_size=gen_config.input_size,
            output_size=gen_config.output_size,
        )

        disc_activation_funcs = disc_config.activation_funcs[:-1] + ["linear"]
        disc_config = DenseNetConfig(
            layer_sizes=disc_config.layer_sizes,
            activation_funcs=disc_activation_funcs,
            optimizer=disc_config.optimizer,
            loss_func=disc_config.loss_func,
            input_size=self.gen_input_size + self.gen_output_size,
            output_size=1,
        )

        config = GANConfig(
            gen_config=gen_config,
            disc_config=disc_config,
        )
        yield config, epoch

    def generate_neighbour_config(
        self,
        config: GANConfig,
        num_epochs: int,
        distance: float,
    ) -> Iterator[Tuple[GANConfig, int]]:
        """
        Generate a neighbour configuration for GANConfig.

        Parameters
        ----------
        config: GANConfig
            Original configuration.
        num_epochs: int
            Number of epochs.
        distance: float
            A proxy for mutation strength.

        Yields
        -------
        Tuple[GANConfig, int]
            New neighbour configuration and number of epochs.
        """

        new_gen_config, _ = next(
            self.gen_space.generate_neighbour_config(
                config.gen_config, num_epochs, distance
            )
        )

        new_disc_config, new_epoch = next(
            self.disc_space.generate_neighbour_config(
                config.disc_config, num_epochs, distance
            )
        )

        new_gen_activation_funcs = new_gen_config.activation_funcs[:-1] + [
            self.gen_out_activation
        ]
        new_gen_config = DenseNetConfig(
            layer_sizes=new_gen_config.layer_sizes,
            activation_funcs=new_gen_activation_funcs,
            optimizer=new_gen_config.optimizer,
            loss_func=new_gen_config.loss_func,
            input_size=new_gen_config.input_size,
            output_size=new_gen_config.output_size,
        )

        new_disc_activation_funcs = new_disc_config.activation_funcs[:-1] + ["linear"]
        new_disc_config = DenseNetConfig(
            layer_sizes=new_disc_config.layer_sizes,
            activation_funcs=new_disc_activation_funcs,
            optimizer=new_disc_config.optimizer,
            loss_func=new_disc_config.loss_func,
            input_size=self.gen_input_size + self.gen_output_size,
            output_size=1,
        )

        neighbour_config = GANConfig(
            gen_config=new_gen_config,
            disc_config=new_disc_config,
        )

        yield neighbour_config, new_epoch
