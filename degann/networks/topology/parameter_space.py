import random
from abc import ABC, abstractmethod
from typing import Dict, Any
from itertools import product
from degann.search_algorithms.nn_code import alph_n_full, alphabet_activations, decode, encode
from degann.search_algorithms.utils import update_random_generator, log_to_file
# from .tf_densenet import TensorflowDenseNet
from degann.networks.imodel import IModel


class ParameterSpace(ABC):
    """
    Abstract class for parameter space of different neural network architectures.
    """

    @abstractmethod
    def _create_parameter_space(self) -> list:
        """
        Abstract method for creating parameter space dict.
        """
        pass


class DenseNetParameterSpace(ParameterSpace):
    """
    Parameter space for DenseNet.
    """

    def __init__(
            self,
            optimizers: list[str],
            loss: list[str],
            min_epoch: int = 100,
            max_epoch: int = 700,
            epoch_step: int = 50,
            nn_min_length: int = 1,
            nn_max_length: int = 6,
            nn_alphabet: list[str] = [
                "".join(elem) for elem in product(alph_n_full, alphabet_activations)
            ],
            alphabet_block_size: int = 1,
            alphabet_offset: int = 8,
    ):
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

        self.configs = self._create_parameter_space()

    def _create_parameter_space(self) -> list:
        configs = []
        for i in range(self.nn_min_length, self.nn_max_length + 1):
            codes = product(self.nn_alphabet, repeat=i)
            for elem in codes:
                code = "".join(elem)
                for epoch in range(self.min_epoch, self.max_epoch + 1, self.epoch_step):
                    for opt in self.optimizers:
                        for loss_func in self.loss:
                            b, a = decode(code, block_size=self.alphabet_block_size, offset=self.alphabet_offset)
                            config = {
                                "block_size": b,
                                "activation_func": a,
                                "optimizer": opt,
                                "loss_func": loss_func,
                                "num_epoch": epoch
                            }
                            configs.append(config)
        return configs

    def get_random_config(self) -> dict[str: Any]:
        return random.choice(self.configs)

    def get_configs(self) -> list:
        return self.configs

    @staticmethod
    def train(
            config: Dict[str, Any],
            input_size: int,
            output_size: int,
            data: tuple,
            repeat: int = 1,
            update_gen_cycle: int = 0,
            val_data: tuple = None,
            logging: bool = False,
            file_name: str = "",
            callbacks: list = None,
    ) -> tuple[float, float, dict]:
        best_net = None
        best_loss = 1e6
        best_val_loss = 1e6
        for i in range(repeat):
            update_random_generator(i, cycle_size=update_gen_cycle)
            history = dict()
            nn = IModel(
                input_size=input_size,
                block_size=config["block_size"],
                output_size=output_size,
                activation_func=config["activation_func"],
            )
            nn.compile(optimizer=config["optimizer"], loss_func=config["loss_func"])
            temp_his = nn.train(
                data[0], data[1], epochs=config["num_epoch"], verbose=0, callbacks=callbacks
            )

            history["shapes"] = [nn.get_shape]
            history["activations"] = [config["activation_func"]]
            history["code"] = [encode(nn)]
            history["epoch"] = [config["num_epoch"]]
            history["optimizer"] = [config["optimizer"]]
            history["loss function"] = [config["loss_func"]]
            history["loss"] = [temp_his.history["loss"][-1]]
            history["validation loss"] = (
                [nn.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)["loss"]]
                if val_data is not None
                else [None]
            )
            history["train_time"] = [nn.network.trained_time["train_time"]]

            if logging:
                fn = f"{file_name}_{len(data[0])}_{config['num_epoch']}_{config['loss_func']}_{config['optimizer']}"
                log_to_file(history, fn)
            if history["loss"][0] < best_loss:
                best_loss = history["loss"][0]
                best_val_loss = history["validation loss"][0]
                best_net = nn.to_dict()
        return best_loss, best_val_loss, best_net
