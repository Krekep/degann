from typing import Optional, List, Dict, Union
from tensorflow.keras.callbacks import Callback

# from tensorflow.keras.src.utils import traceback_utils
import tensorflow as tf
import numpy as np
import math


from degann.networks import layer_creator, losses, metrics, optimizers
from degann.networks.topology.pinn.compile_config import PINNCompileParams
from degann.networks.topology.pinn.topology_config import PINNParams
from degann.networks.topology.pinn.virtual_loss import VirtualLoss
from degann.networks.topology.densenet.tf_densenet import TensorflowDenseNet


class PhysicsInformedNet(tf.keras.Model):
    def __init__(self, config: Optional[PINNParams] = None, **kwargs):
        super().__init__()
        if config is None:
            config = PINNParams()
        self._name = "PINN"

        self.network = TensorflowDenseNet(config.densenet_params)
        self.virtual_functions: List[VirtualLoss] = []

    def custom_compile(self, config: Optional[PINNCompileParams]) -> None:
        """
        Configures the model for training

        Parameters
        ----------
        config: DenseNetCompileParams
            parameters for compilation containing learning rate, optimizer,
            loss function and metrics

        Returns
        -------

        """
        if config is None:
            config = PINNCompileParams()
        self.virtual_functions = config.virtual_functions
        self.network.custom_compile(config.densenet_compile_params)
        self.collocational_points_generator = config.collocational_points_generator

    def call(self, inputs, training=None, mask=None):
        """
        Obtaining a neural network response on the input data vector
        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """
        self.network.call(inputs=inputs, training=training, mask=mask)

    def train_step(self, data: tuple[tf.Tensor, tf.Tensor] | None):  # type: ignore
        """
        Custom train step with physics and
        boundary losses implementation
        Parameters
        ----------
        data: tuple
            Pair of x and y (or dataset)
        Returns
        -------
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x = self.collocational_points_generator()
        if data is not None:
            # print(data)
            x, y = data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y_pred: tf.Tensor = self.network(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            total_loss = None
            if data is not None:
                total_loss = self.network.compute_loss(y=y, y_pred=y_pred)
            if total_loss is None:
                total_loss = tf.constant(0, dtype=tf.float32)
            for virtual_function in self.virtual_functions:
                virtual_deviation = virtual_function(self.network, tape, x)
                virtual_loss = self.network.compiled_loss(
                    tf.zeros_like(virtual_deviation), virtual_deviation
                )
                total_loss += virtual_loss * virtual_function.weight

        # Compute gradients
        trainable_vars = self.network.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weight
        if self.network.optimizer is None:
            raise RuntimeError("compile or custom_compile must be called before train")
        self.network.optimizer.apply_gradients(zip(gradients, trainable_vars))
        del tape
        # Update metrics (includes the metric that tracks the loss)
        loss_metric = None
        for metric in self.network.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
                loss_metric = metric
            elif data is not None:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        if data is not None:
            return {m.name: m.result() for m in self.network.metrics}
        return {loss_metric.name: loss_metric.result()}

    def set_name(self, new_name):
        self._name = new_name

    def __str__(self):
        res = f"IModel {self.name}\n"
        for layer in self.network.blocks:
            res += str(layer)
        res += str(self.network.out_layer)
        return res

    def to_dict(self, **kwargs):
        """
        Export neural network to dictionary

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        res = self.network.to_dict(**kwargs)
        res["net_type"] = "MyPINN"
        return

    def from_dict(self, config: dict, **kwargs):
        """
        Restore neural network from dictionary of params

        Parameters
        ----------
        config: dict
            Model parameters

        """
        self.network.from_dict(config["densenet_params"])

    @property
    def get_activations(self) -> List:
        """
        Get list of activations functions for each layer
        Returns
        -------
        activation: list
        """
        return self.network.get_activations

    # @traceback_utils.filter_traceback
    def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None):
        return self.network.predict(
            x=x,
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
        )
