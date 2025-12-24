import random
from abc import ABC, abstractmethod
from typing import List, Tuple
from itertools import product
from degann.search_algorithms.nn_code import alph_n_full, alphabet_activations, decode, encode
from degann.search_algorithms.utils import update_random_generator, log_to_file
from degann.networks.imodel import IModel
from degann.search_algorithms.generate import generate_neighbour, mutate_block_sizes, mutate_activations
from degann.networks.topology.configs import DenseNetConfig, GANConfig


class ParameterSpace(ABC):
    """
    Abstract class for parameter space of different neural network architectures.
    """

    @abstractmethod
    def create_parameter_space(self) -> List:
        """
        Abstract method for creating parameter space dict.
        """
        pass

    @abstractmethod
    def get_random_config(self):
        """
        Abstract method that creates and returns a random config.
        """
        pass

    @abstractmethod
    def generate_neighbour_config(self, config, distance: float):
        """
        Abstract method that generates neighbour config.
        """
        pass

    @abstractmethod
    def train(self, config, *args, **kwargs) -> Tuple[float, float, dict]:
        """
        Abstract method for training and evaluating model.
        """
        pass


class DenseNetParameterSpace(ParameterSpace):
    """
    Parameter space for DenseNet.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            optimizers: list[str],
            loss: list[str],
            min_epoch: int = 100,
            max_epoch: int = 700,
            epoch_step: int = 1,
            nn_min_length: int = 1,
            nn_max_length: int = 6,
            nn_alphabet: list[str] = [
                "".join(elem) for elem in product(alph_n_full, alphabet_activations)
            ],
            alphabet_block_size: int = 1,
            alphabet_offset: int = 8,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.optimizers = optimizers
        self.loss = loss
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.epoch_step = epoch_step
        self.nn_min_length = nn_min_length
        self.nn_max_length = nn_max_length
        self.nn_alphabet = nn_alphabet
        self.alphabet_block_size = alphabet_block_size
        self.alphabet_offset = alphabet_offset

    def create_parameter_space(self) -> List[DenseNetConfig]:
        """
        Creates parameter space with all possible configurations.

        Returns
        -------
        configs: List[DenseNetConfig]
            List of all possible configurations in the parameter space.
        """

        configs = []
        for i in range(self.nn_min_length, self.nn_max_length + 1):
            codes = product(self.nn_alphabet, repeat=i)
            for elem in codes:
                code = "".join(elem)
                for epoch in range(self.min_epoch, self.max_epoch + 1, self.epoch_step):
                    for opt in self.optimizers:
                        for loss_func in self.loss:
                            b, a = decode(code, block_size=self.alphabet_block_size, offset=self.alphabet_offset)
                            config = DenseNetConfig(
                                block_size=b,
                                activation_func=a + ["linear"],
                                optimizer=opt,
                                loss_func=loss_func,
                                num_epoch=epoch,
                                code=code,
                                input_size=self.input_size,
                                output_size=self.output_size
                            )
                            configs.append(config)
        return configs

    def get_random_config(self) -> DenseNetConfig:
        """
        Creates random configuration.

        Returns
        -------
        config: DenseNetConfig
            Random configuration.
        """

        block = random.randint(self.nn_min_length, self.nn_max_length)
        code = ""

        for i in range(block):
            code += self.nn_alphabet[random.randint(0, len(self.nn_alphabet) - 1)]
        epoch = random.randint(self.min_epoch, self.max_epoch)
        b, a = decode(code, block_size=self.alphabet_block_size, offset=self.alphabet_offset)
        opt = random.choice(self.optimizers)
        loss_func = random.choice(self.loss)
        config = DenseNetConfig(
            block_size=b,
            activation_func=a + ["linear"],
            optimizer=opt,
            loss_func=loss_func,
            num_epoch=epoch,
            code=code,
            input_size=self.input_size,
            output_size=self.output_size
        )
        return config

    def generate_neighbour_config(self, config: DenseNetConfig, distance: float) -> DenseNetConfig:
        """
        Generate neighbour configuration based on distance value.

        Parameters
        ----------
        config: DenseNetConfig
            Original configuration.
        distance: float
            Distance for generating neighbour configuration.

        Returns
        -------
        neighbour_config: DenseNetConfig
            Neighbour configuration.
        """

        code = config.code
        epoch = config.num_epoch
        parameters = (code, epoch)

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
            new_code,
            block_size=self.alphabet_block_size,
            offset=self.alphabet_offset
        )

        neighbour_config = DenseNetConfig(
            block_size=b,
            activation_func=a + ["linear"],
            optimizer=config.optimizer,
            loss_func=config.loss_func,
            num_epoch=new_epoch_param.value(),
            code=new_code,
            input_size=config.input_size,
            output_size=config.output_size
        )
        return neighbour_config

    def train(
            self,
            config: DenseNetConfig = None,
            data: tuple = None,
            repeat: int = 1,
            update_gen_cycle: int = 0,
            val_data: tuple = None,
            logging: bool = False,
            file_name: str = "",
            callbacks: list = None,
    ) -> tuple[float, float, dict]:
        """
        Train and evaluate model with given configuration.

        Parameters
        ----------
        config: DenseNetConfig
            Configuration for training the model.
        data: Tuple[Any, Any]
            Training data.
        repeat: int
            Number of training repetitions.
        update_gen_cycle: int
            Cycle size for random generator update.
        val_data: Tuple[Any, Any]
            Validation data.
        logging: bool
            Flag to enable logging.
        file_name: str
            Name for log files.
        callbacks: List[Any]
            List of training callbacks.

        Returns
        -------
        best_loss: float
            Best training loss achieved.
        best_val_loss: float
            Best validation loss achieved.
        best_net: dict
            Dictionary representation of the best network.
        """

        best_net = None
        best_loss = 1e6
        best_val_loss = 1e6
        for i in range(repeat):
            update_random_generator(i, cycle_size=update_gen_cycle)
            history = dict()
            nn = IModel(
                config=config,
                name="net",
                net_type="DenseNet"
            )
            nn.compile(optimizer=config.optimizer, loss_func=config.loss_func)
            temp_his = nn.train(
                data[0], data[1], epochs=config.num_epoch, verbose=0, callbacks=callbacks
            )

            history["shapes"] = [nn.get_shape]
            history["activations"] = [config.activation_func]
            history["code"] = [encode(nn)]
            history["epoch"] = [config.num_epoch]
            history["optimizer"] = [config.optimizer]
            history["loss function"] = [config.loss_func]
            history["loss"] = [temp_his.history["loss"][-1]]
            history["validation loss"] = (
                [nn.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)["loss"]]
                if val_data is not None
                else [None]
            )
            history["train_time"] = [nn.network.trained_time["train_time"]]

            if logging:
                fn = f"{file_name}_{len(data[0])}_{config.num_epoch}_{config.loss_func}_{config.optimizer}"
                log_to_file(history, fn)
            if history["loss"][0] < best_loss:
                best_loss = history["loss"][0]
                best_val_loss = history["validation loss"][0]
                best_net = nn.to_dict()
        return best_loss, best_val_loss, best_net


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

    def create_parameter_space(self) -> List[GANConfig]:
        configs = []
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

                    gen_af_options = list(product(self.gen_activation_funcs, repeat=gen_hidden_depth))
                    disc_af_options = list(product(self.disc_activation_funcs, repeat=disc_hidden_depth))

                    for gen_af_tuple in gen_af_options:
                        for disc_af_tuple in disc_af_options:
                            gen_activation_funcs = list(gen_af_tuple) + [self.gen_out_activation]
                            disc_activation_funcs = list(disc_af_tuple) + ["linear"]

                            for epoch in self.epochs:
                                config = GANConfig(
                                    gen_input_size=self.gen_input_size,
                                    gen_output_size=self.gen_output_size,
                                    gen_block_sizes=gen_bs,
                                    gen_activation_funcs=gen_activation_funcs,
                                    gen_out_activation=self.gen_out_activation,
                                    disc_block_sizes=disc_bs,
                                    disc_activation_funcs=disc_activation_funcs,
                                    gen_optimizer=gen_opt,
                                    disc_optimizer=disc_opt,
                                    gen_loss_func=gen_lf,
                                    disc_loss_func=disc_lf,
                                    num_epoch=epoch,
                                )
                                configs.append(config)
        return configs

    def get_random_config(self):
        gen_depth = random.randint(self.gen_min_depth, self.gen_max_depth)
        disc_depth = random.randint(self.disc_min_depth, self.disc_max_depth)

        gen_block_sizes = [random.choice(self.gen_layer_sizes) for _ in range(gen_depth)]
        disc_block_sizes = [random.choice(self.disc_layer_sizes) for _ in range(disc_depth)]

        gen_activation_funcs = [random.choice(self.gen_activation_funcs) for _ in range(gen_depth)] + [self.gen_out_activation]
        disc_activation_funcs = [random.choice(self.disc_activation_funcs) for _ in range(disc_depth)] + ["linear"]

        epoch = random.choice(self.epochs)

        config = GANConfig(
            gen_input_size=self.gen_input_size,
            gen_output_size=self.gen_output_size,
            gen_block_sizes=gen_block_sizes,
            gen_activation_funcs=gen_activation_funcs,
            gen_out_activation=self.gen_out_activation,
            disc_block_sizes=disc_block_sizes,
            disc_activation_funcs=disc_activation_funcs,
            gen_optimizer=random.choice(self.gen_optimizers),
            disc_optimizer=random.choice(self.disc_optimizers),
            gen_loss_func=random.choice(self.gen_loss_funcs),
            disc_loss_func=random.choice(self.disc_loss_funcs),
            num_epoch=epoch,
        )
        return config

    def generate_neighbour_config(
            self,
            config: GANConfig,
            distance: float,
    ) -> GANConfig:
        """
        Generate a neighbour configuration for GANConfig.

        Parameters
        ----------
        config : GANConfig
            Original configuration.
        distance : float
            A proxy for mutation strength (higher -> more changes).

        Returns
        -------
        GANConfig
            New neighbour configuration.
        """
        mutation_prob = min(0.5, distance / 100.0)

        new_gen_block_sizes = mutate_block_sizes(
            block_sizes=config.gen_block_sizes,
            layer_sizes=self.gen_layer_sizes,
            min_depth=self.gen_min_depth,
            max_depth=self.gen_max_depth,
            mutation_prob=mutation_prob,
        )
        new_gen_activations = mutate_activations(
            activations=config.gen_activation_funcs[:-1] + [config.gen_out_activation],
            all_activations=self.gen_activation_funcs,
            mutation_prob=mutation_prob,
        )

        if len(new_gen_activations) > 1:
            new_gen_activations[-1] = config.gen_out_activation
        else:
            new_gen_activations = [config.gen_out_activation]

        new_disc_block_sizes = mutate_block_sizes(
            block_sizes=config.disc_block_sizes,
            layer_sizes=self.disc_layer_sizes,
            min_depth=self.disc_min_depth,
            max_depth=self.disc_max_depth,
            mutation_prob=mutation_prob,
        )
        new_disc_activations = mutate_activations(
            activations=config.disc_activation_funcs[:-1] + ["linear"],
            all_activations=self.disc_activation_funcs,
            mutation_prob=mutation_prob,
        )

        if len(new_disc_activations) > 1:
            new_disc_activations[-1] = "linear"
        else:
            new_disc_activations = ["linear"]

        new_gen_optimizer = config.gen_optimizer
        new_disc_optimizer = config.disc_optimizer
        new_gen_loss_func = config.gen_loss_func
        new_disc_loss_func = config.disc_loss_func
        new_num_epoch = config.num_epoch

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

        neighbour_config = GANConfig(
            gen_input_size=self.gen_input_size,
            gen_output_size=self.gen_output_size,
            gen_block_sizes=new_gen_block_sizes,
            gen_activation_funcs=new_gen_activations,
            gen_out_activation=config.gen_out_activation,
            disc_block_sizes=new_disc_block_sizes,
            disc_activation_funcs=new_disc_activations,
            gen_optimizer=new_gen_optimizer,
            disc_optimizer=new_disc_optimizer,
            gen_loss_func=new_gen_loss_func,
            disc_loss_func=new_disc_loss_func,
            num_epoch=new_num_epoch,
        )
        return neighbour_config

    def train(
            self,
            config: GANConfig = None,
            data: tuple = None,
            repeat: int = 1,
            val_data: tuple = None,
            logging: bool = False,
            file_name: str = "",
            callbacks: list = None,
            verbose: int = 0,
            mini_batch_size: int = 32,
    ) -> tuple[float, float, dict]:
        best_net = None
        best_loss = float('inf')
        best_val_loss = float('inf')

        for i in range(repeat):
            nn = IModel(config=config, net_type="GAN")
            nn.compile(
                gen_optimizer=config.gen_optimizer,
                disc_optimizer=config.disc_optimizer,
                gen_loss_func=config.gen_loss_func,
                disc_loss_func=config.disc_loss_func,
            )

            history = nn.train(
                x_data=data[0],
                y_data=data[1],
                validation_data=val_data,
                epochs=config.num_epoch,
                mini_batch_size=mini_batch_size,
                callbacks=callbacks,
                verbose=verbose
            )

            last_gen_loss = history.history['gen_loss'][-1]
            last_disc_loss = history.history['disc_loss'][-1]
            current_loss = (last_disc_loss + last_gen_loss) / 2.0

            current_val_loss = current_loss
            if val_data is not None:
                pass

            if logging:
                fn = f"{file_name}_gan_{len(data[0])}_{config.num_epoch}_{config.gen_loss_func}_{config.disc_loss_func}_{config.gen_optimizer}_{config.disc_optimizer}"
                log_to_file(history, fn)

            if current_loss < best_loss:
                best_loss = current_loss
                best_val_loss = current_val_loss
                best_net = nn.to_dict()

        return best_loss, best_val_loss, best_net
