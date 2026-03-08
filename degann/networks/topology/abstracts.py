from abc import ABC, abstractmethod
from typing import Tuple, Any, Iterator, Dict


class NetConfig(ABC):
    """
    Abstract base class for neural network configuration.
    Defines the interface that all config classes must implement.
    """

    @property
    @abstractmethod
    def get_shape(self) -> Any:
        """
        Get the shape of the network configuration.
        """
        pass

    @property
    @abstractmethod
    def get_input_size(self) -> Any:
        """
        Get the input size of the network.
        """
        pass

    @property
    @abstractmethod
    def get_output_size(self) -> Any:
        """
        Get the output size of the network.
        """
        pass

    @property
    @abstractmethod
    def get_loss_func(self) -> Any:
        """
        Get the loss function name.
        """
        pass

    @property
    @abstractmethod
    def get_optimizer(self) -> Any:
        """
        Get the optimizer name.
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Create a configuration instance from a dictionary.
        """
        pass


class ParameterSpace(ABC):
    """
    Abstract class for parameter space of different neural network architectures.
    """

    @abstractmethod
    def iter_configs(self) -> Iterator[Tuple[NetConfig, int]]:
        """
        Abstract method for generating configs from the parameter space.
        """
        pass

    @abstractmethod
    def get_random_config(self) -> Tuple[NetConfig, int]:
        """
        Abstract method that creates and returns a random config.
        """
        pass

    @abstractmethod
    def generate_neighbour_config(
        self, config: NetConfig, num_epochs: int, distance: float
    ) -> NetConfig:
        """
        Abstract method that generates neighbour config.
        """
        pass

    @abstractmethod
    def train(
        self, config: NetConfig, num_epochs: int, data: tuple, *args, **kwargs
    ) -> Tuple[float, float, dict]:
        """
        Abstract method for training and evaluating model.
        """
        pass
