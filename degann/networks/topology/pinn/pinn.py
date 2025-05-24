from typing import Optional, List, Union
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Optimizer
from keras.src.trainers.data_adapters import data_adapter_utils

import tensorflow as tf
import numpy as np

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
        self.with_data = True

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

    def train(
        self,
        x_data: Union[np.ndarray, tf.Tensor, None] = None,
        y_data: Union[np.ndarray, tf.Tensor, None] = None,
        validation_split=0.0,
        validation_data=None,
        epochs=10,
        batch_size: Optional[int] = None,
        callbacks: Optional[List[Callback] | tf.keras.callbacks.CallbackList] = None,
        verbose="auto",
        sample_weight=None,
        initial_epoch=0,
    ):
        """
        Custom training method that internally uses train_step and test_step

        Args:
            x_data: Input data
            y_data: Target data
            validation_split: Fraction of data to use for validation
            validation_data: Tuple (x_val, y_val) for validation
            epochs: Number of training epochs
            mini_batch_size: Size of mini-batches (None for full batch)
            callbacks: List of keras callbacks
            verbose: Verbosity mode ("auto", 0, 1, or 2)
        """
        if x_data is not None and y_data is not None:
            return self.network.fit(
                x_data,
                y_data,
                batch_size=batch_size,
                callbacks=callbacks,
                validation_split=validation_split,
                validation_data=validation_data,
                epochs=epochs,
                verbose=verbose,
                sample_weight=sample_weight,
            )
        self.network._assert_compile_called("train")
        self.network._eval_epoch_iterator = None

        num_samples = 1
        self.network._maybe_symbolic_build()

        # Initialize callbacks
        if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
            callbacks = tf.keras.callbacks.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=1,  # num_samples,
                model=self.network,
            )
        self.network.stop_training = False
        self.make_train_function()
        callbacks.on_train_begin()
        training_logs = None
        logs = {}
        initial_epoch = self.network._initial_epoch or initial_epoch

        # Epoch loop
        for epoch in range(initial_epoch, epochs):
            self.network.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            callbacks.on_train_batch_begin(1)

            logs = self.train_function([None])
            callbacks.on_train_batch_end(1, logs)
            callbacks.on_epoch_end(epoch, logs)

            training_logs = logs
            if self.network.stop_training:
                break

            # TODO: custom evaluate to validate network without data

        if isinstance(self.network.optimizer, Optimizer) and epochs > 0:
            self.network.optimizer.finalize_variable_values(
                self.network.trainable_weights
            )

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self.network, "_eval_epoch_iterator", None) is not None:
            del self.network._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.network.history

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
            x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y_pred: tf.Tensor = self.network(x, training=True)
            total_loss = None
            if data is not None:
                total_loss = self.network.compute_loss(
                    x=x,
                    y=y,
                    y_pred=y_pred,
                    sample_weight=sample_weight,
                    training=True,
                )

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
            return self.compute_metrics(
                x, y, y_pred, sample_weight=sample_weight
            )  # {m.name: m.result() for m in self.network.metrics}
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
