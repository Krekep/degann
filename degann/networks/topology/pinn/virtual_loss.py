import tensorflow as tf
from abc import ABC, abstractmethod

class VirtualLoss(ABC):

    @abstractmethod
    def __call__(self, model: tf.keras.Model, tape: tf.GradientTape, x: tf.Tensor) -> float | int | tf.Tensor:
        ...
    
    @property
    def weight(self) -> float:
        return 1