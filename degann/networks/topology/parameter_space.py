import random
from abc import ABC, abstractmethod
from typing import Dict, Any
from itertools import product
from degann.search_algorithms.nn_code import alph_n_full, alphabet_activations, decode, encode
from degann.search_algorithms.utils import update_random_generator, log_to_file
from degann.networks.imodel import IModel
from degann.search_algorithms.generate import generate_neighbour


class ParameterSpace(ABC):
    """
    Abstract class for parameter space of different neural network architectures.
    """

    @abstractmethod
    def create_parameter_space(self) -> list:
        """
        Abstract method for creating parameter space dict.
        """
        pass

    @abstractmethod
    def get_random_config(self) -> Dict[str, Any]:
        """
        Abstract method that creates and returns a random config.
        """
        pass

    @abstractmethod
    def generate_neighbour_config(self, config: Dict[str, Any], distance: float) -> Dict[str, Any]:
        """
        Abstract method that generates neighbour config.
        """
        pass

    @abstractmethod
    def train(config: Dict[str, Any], *args, **kwargs) -> tuple[float, float, dict]:
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

    def create_parameter_space(self) -> list:
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
                                "code": code,
                                "block_size": b,
                                "activation_func": a + ["linear"],
                                "optimizer": opt,
                                "loss_func": loss_func,
                                "num_epoch": epoch
                            }
                            configs.append(config)
        return configs

    def get_random_config(self) -> dict[str: Any]:
        block = random.randint(self.nn_min_length, self.nn_max_length)
        code = ""

        for i in range(block):
            code += self.nn_alphabet[random.randint(0, len(self.nn_alphabet) - 1)]
        epoch = random.randint(self.min_epoch, self.max_epoch)
        b, a = decode(code, block_size=self.alphabet_block_size, offset=self.alphabet_offset)
        opt = random.choice(self.optimizers)
        loss_func = random.choice(self.loss)
        config = {
            "code": code,
            "block_size": b,
            "activation_func": a + ["linear"],
            "optimizer": opt,
            "loss_func": loss_func,
            "num_epoch": epoch
        }
        return config

    def generate_neighbour_config(self, config: dict, distance: float) -> dict:
        code = config["code"]
        epoch = config["num_epoch"]
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

        neighbour_config = {
            "code": new_code,
            "block_size": b,
            "activation_func": a + ["linear"],
            "optimizer": config["optimizer"],
            "loss_func": config["loss_func"],
            "num_epoch": new_epoch_param.value()
        }
        return neighbour_config

    @staticmethod
    def train(
            config: Dict[str, Any] = None,
            input_size: int = 1,
            output_size: int = 1,
            data: tuple = None,
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
                config=config,
                input_size=input_size,
                output_size=output_size,
                net_type="DenseNet"
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
