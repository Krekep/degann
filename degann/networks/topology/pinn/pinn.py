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

    def train(
        self,
        x_data: Union[np.ndarray, tf.Tensor, None] = None,
        y_data: Union[np.ndarray, tf.Tensor, None] = None,
        validation_split=0.0,
        validation_data=None,
        epochs=10,
        batch_size=None,
        callbacks: Optional[List[Callback] | tf.keras.callbacks.CallbackList] = None,
        verbose="auto",
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
        with_data = True
        if x_data is None or y_data is None:
            with_data = False
        # Handle validation data
        if validation_data is not None:
            x_val, y_val = validation_data
        elif validation_split > 0 and with_data:
            split = int(len(x_data) * (1 - validation_split))
            x_train, y_train = x_data[:split], y_data[:split]
            x_val, y_val = x_data[split:], y_data[split:]
            x_data, y_data = x_train, y_train
        else:
            x_val, y_val = None, None

        if with_data:
            # Convert data to tensors if needed
            if not isinstance(x_data, tf.Tensor):
                x_data = tf.convert_to_tensor(x_data)
            if not isinstance(y_data, tf.Tensor):
                y_data = tf.convert_to_tensor(y_data)
            if x_val is not None and not isinstance(x_val, tf.Tensor):
                x_val = tf.convert_to_tensor(x_val)
            if y_val is not None and not isinstance(y_val, tf.Tensor):
                y_val = tf.convert_to_tensor(y_val)

        num_samples = None
        num_batches = None
        if with_data:
            num_samples = len(x_data) if x_data is not None else 0
            batch_size = batch_size if batch_size is not None else num_samples
            num_batches = math.ceil(num_samples / batch_size) if num_samples > 0 else 0

        if callbacks is None:
            callbacks = []

        # Initialize callbacks
        if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
            callbacks = tf.keras.callbacks.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=num_samples if with_data else 1,
                model=self.network,
            )

        # Callback hooks
        training_logs = {}
        callbacks.on_train_begin()

        # Epoch loop
        for epoch in range(epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}

            self._jax_state_synced = True

            range_end = num_samples
            range_step = batch_size

            if not with_data:
                range_end = 1
                range_step = 1

            # Batch training
            epoch_losses = []
            for step in range(0, range_end, range_step):
                callbacks.on_train_batch_begin(step)
                x_batch, y_batch = 0, 0
                if with_data:
                    batch_end = min(step + batch_size, num_samples)
                    x_batch = x_data[step:batch_end]
                    y_batch = y_data[step:batch_end]

                    # Use train_step
                    train_logs = self.train_step((x_batch, y_batch))
                else:
                    train_logs = self.train_step(None)
                loss = train_logs["loss"]
                epoch_losses.append(loss.numpy())

                # Callback hooks for batch
                batch_logs = {
                    "batch": step // batch_size if batch_size is not None else 0,
                    "size": len(x_batch) if with_data else 0,
                    "loss": loss.numpy(),
                }
                batch_logs.update(train_logs)
                callbacks.on_train_batch_end(step, batch_logs)

                if self.network.stop_training:
                    # Stop training if a callback has set
                    # this flag in on_(train_)batch_end.
                    break

            # Epoch metrics
            epoch_loss = np.mean(epoch_losses)
            # history["loss"].append(epoch_loss)
            epoch_logs["loss"] = epoch_loss

            # Validation (using test_step)
            if x_val is not None and y_val is not None:
                val_logs = self.test_step((x_val, y_val))
                val_loss = val_logs["loss"]
                # history["val_loss"].append(val_loss.numpy())
                epoch_logs["val_loss"] = val_loss

                # Add other metrics if available
                for k, v in val_logs.items():
                    if k not in ["loss", "val_loss"]:
                        epoch_logs[k] = v

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.network.stop_training:
                break
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
