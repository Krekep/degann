import random
from typing import List, Tuple, Optional, Iterator
from itertools import product
from degann.search_algorithms.nn_code import (
    alph_n_full,
    alphabet_activations,
    decode,
    encode,
)
from degann.search_algorithms.utils import update_random_generator, log_to_file
from degann.networks.imodel import IModel
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

    def train(
        self,
        config: DenseNetConfig,
        num_epochs: int,
        data: tuple,
        repeat: int = 1,
        update_gen_cycle: int = 0,
        val_data: Optional[tuple] = None,
        logging: bool = False,
        file_name: str = "",
        callbacks: Optional[list] = None,
    ) -> Tuple[float, float, dict]:
        """
        Train and evaluate model with given configuration.

        Parameters
        ----------
        config: DenseNetConfig
            Configuration for training the model.
        num_epochs: int
            Number of training epochs.
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
            nn = IModel(config=config, name="net", net_type="DenseNet")
            nn.compile(optimizer=config.optimizer, loss_func=config.loss_func)
            temp_his = nn.train(
                data[0],
                data[1],
                epochs=num_epochs,
                verbose=0,
                callbacks=callbacks,
            )

            history["shapes"] = [nn.get_shape]
            history["activations"] = [config.activation_func]
            history["code"] = [encode(nn)]
            history["epoch"] = [num_epochs]
            history["optimizer"] = [config.optimizer]
            history["loss function"] = [config.loss_func]
            history["loss"] = [temp_his.history["loss"][-1]]
            history["validation loss"] = (
                [
                    nn.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)[
                        "loss"
                    ]
                ]
                if val_data is not None
                else [None]
            )
            history["train_time"] = [nn.network.trained_time["train_time"]]

            if logging:
                fn = f"{file_name}_{len(data[0])}_{num_epochs}_{config.loss_func}_{config.optimizer}"
                log_to_file(history, fn)
            if history["loss"][0] < best_loss:
                best_loss = history["loss"][0]
                best_val_loss = history["validation loss"][0]
                best_net = nn.to_dict()
        return best_loss, best_val_loss, best_net
