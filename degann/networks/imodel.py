import json
from collections import defaultdict
from typing import List, Optional, Dict, Union, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from degann.networks.config_format import HEADER_OF_APG_FILE
from degann.networks.topology.DenseNet.tf_densenet import TensorflowDenseNet
from degann.networks.topology.GAN.tf_gan import TensorflowGAN
from degann.networks.topology.abstracts import NetConfig
from degann.networks.topology.DenseNet.config import DenseNetConfig
from degann.networks.topology.GAN.config import GANConfig


class IModel(object):
    """
    Interface class for working with neural topology
    """

    def __init__(
        self,
        config: NetConfig,
        net_type: str,
        name: str = "net",
        is_debug: bool = False,
        **kwargs,
    ):
        self.network = _create_functions[net_type](
            config,
            is_debug=is_debug,
            **kwargs,
        )

        self._input_size = config.get_input_size
        self._output_size = config.get_output_size
        self._shape = config.get_shape

        self._name = name
        self._is_debug = is_debug
        self.set_name(name)
        self._evaluate_history = None

    def compile(self, **kwargs) -> None:
        """
        Configures the model for training
        """
        self.network.custom_compile(**kwargs)

    def feedforward(self, inputs: np.ndarray) -> tf.Tensor:
        """
        Return network answer for passed input by network __call__()

        Parameters
        ----------
        inputs: np.ndarray
            Input activation vector

        Returns
        -------
        outputs: tf.Tensor
            Network answer
        """
        return self.network(inputs, training=False)

    def train(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        validation_split: float = 0.0,
        validation_data: Optional[tuple] = None,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[List] = None,
        verbose: Union[int, str] = "auto",
    ) -> keras.callbacks.History:
        """
        Train network on passed dataset and return training history
        """
        if self._is_debug:
            if callbacks is not None:
                callbacks.append(
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                )
            else:
                callbacks = [
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                ]
        temp = self.network.fit(
            x_data,
            y_data,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            epochs=epochs,
            verbose=verbose,
        )
        return temp

    def evaluate(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        batch_size: Optional[int] = None,
        callbacks: Optional[List] = None,
        verbose: Union[int, str] = "auto",
        **kwargs,
    ) -> Union[float, List[float]]:
        """
        Evaluate network on passed dataset and return evaluate history
        """
        if self._is_debug:
            if callbacks is not None:
                callbacks.append(
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                )
            else:
                callbacks = [
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                ]
        self._evaluate_history = self.network.evaluate(
            x_data,
            y_data,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs,
        )
        return self._evaluate_history

    def clear_history(self):
        del self.network.history
        del self._evaluate_history

    def export_to_cpp(
        self,
        path: str,
        array_type: str = "[]",
        path_to_compiler: str = None,
        vectorized_level: str = "none",
        **kwargs,
    ) -> None:
        """
        Export neural network as feedforward function on c++
        """
        self.network.export_to_cpp(
            path,
            array_type,
            path_to_compiler,
            vectorized_level=vectorized_level,
            **kwargs,
        )

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """
        Export neural network to dictionary
        """
        return self.network.to_dict(**kwargs)

    def export_to_file(self, path: str, **kwargs) -> None:
        """
        Export neural network as parameters to file
        """
        config = self.to_dict(**kwargs)
        with open(path + ".apg", "w") as f:
            f.write(HEADER_OF_APG_FILE + json.dumps(config, indent=2))

    @classmethod
    def from_dict(cls, config: dict, **kwargs):
        """
        Create neural network instance from dictionary.

        Parameters
        ----------
        config : dict
            Dictionary with network parameters.

        Returns
        -------
        IModel
            New instance of the network.
        """
        net_type = config["net_type"]
        name = config["name"]

        config_class = _create_config[net_type]
        net_config = config_class.from_dict(config["config"])

        instance = cls(config=net_config, net_type=net_type, name=name, **kwargs)
        instance.network.from_dict(config, **kwargs)
        return instance

    @classmethod
    def from_file(cls, path: str, **kwargs):
        """
        Create neural network instance from file.

        Parameters
        ----------
        path : str
            Path to file (without extension).

        Returns
        -------
        IModel
            New instance of the network.
        """
        with open(path + ".apg", "r") as f:
            for header in range(HEADER_OF_APG_FILE.count("\n")):
                _ = f.readline()
            config_str = ""
            for line in f:
                config_str += line
            config = json.loads(config_str)
        return cls.from_dict(config, **kwargs)

    def set_name(self, name: str) -> None:
        """
        Set network name
        """
        self.network.set_name(name)
        self._name = name

    @property
    def get_name(self) -> str:
        return self._name

    @property
    def get_shape(self) -> Any:
        """
        Get shape for current network
        """
        return self._shape

    @property
    def get_input_size(self) -> Any:
        """
        Get input vector size for current network
        """
        return self._input_size

    @property
    def get_output_size(self) -> Any:
        """
        Get output vector size for current network
        """
        return self._output_size

    @property
    def get_activations(self) -> Union[List[str], Tuple[List[str], List[str]]]:
        """
        Get list of activations for each layer
        """
        return self.network.get_activations

    def __str__(self) -> str:
        """
        Get a string representation of the neural network
        """
        return str(self.network)


_create_functions = defaultdict(lambda: TensorflowDenseNet)
_create_functions["DenseNet"] = TensorflowDenseNet
_create_functions["GAN"] = TensorflowGAN

_create_config = defaultdict(lambda: NetConfig)
_create_config["DenseNet"] = DenseNetConfig
_create_config["GAN"] = GANConfig
