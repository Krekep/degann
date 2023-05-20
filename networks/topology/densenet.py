from typing import List, Optional, Dict

import tensorflow as tf
from keras.utils.tf_utils import ListWrapper
from tensorflow import keras

from networks import layer_creator, optimizers, losses, metrics
from networks.layers.dense import MyDense


class DenseNet(tf.keras.Model):
    def __init__(
        self,
        input_size: int = 2,
        block_size: list = None,
        output_size: int = 10,
        activation_func: str = "linear",
        weight=keras.initializers.get("ones"),
        biases=keras.initializers.get("zeros"),
        is_debug: bool = False,
        **kwargs,
    ):
        decorator_params: List[Optional[Dict]] = [None]
        if "decorator_params" in kwargs.keys():
            decorator_params = kwargs.get("decorator_params")
            kwargs.pop("decorator_params")
        else:
            decorator_params = [None]

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is None
            or decorator_params is None
        ):
            decorator_params = [None] * (len(block_size) + 1)

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is not None
        ):
            decorator_params = decorator_params * (len(block_size) + 1)

        super(DenseNet, self).__init__(**kwargs)
        self.blocks = []

        if not isinstance(activation_func, list):
            activation_func = [activation_func] * (len(block_size) + 1)
        if len(block_size) != 0:
            self.blocks.append(
                layer_creator.create_dense(
                    input_size,
                    block_size[0],
                    activation=activation_func[0],
                    weight=weight,
                    bias=biases,
                    is_debug=is_debug,
                    name=f"MyDense0",
                    decorator_params=decorator_params[0],
                )
            )
            for i in range(1, len(block_size)):
                self.blocks.append(
                    layer_creator.create_dense(
                        block_size[i - 1],
                        block_size[i],
                        activation=activation_func[i],
                        weight=weight,
                        bias=biases,
                        is_debug=is_debug,
                        name=f"MyDense{i}",
                        decorator_params=decorator_params[i],
                    )
                )
        else:
            block_size = [input_size]

        self.out_layer = layer_creator.create_dense(
            block_size[-1],
            output_size,
            activation=activation_func[-1],
            weight=weight,
            bias=biases,
            is_debug=is_debug,
            name=f"OutLayerMyDense",
            decorator_params=decorator_params[-1],
        )

        self.activation_funcs = activation_func
        self.weight_initializer = weight
        self.bias_initializer = biases
        self.input_size = input_size
        self.block_size = block_size
        self.output_size = output_size
        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}

    def custom_compile(
        self,
        rate=1e-2,
        optimizer="SGD",
        loss_func="MeanSquaredError",
        metric_funcs=None,
        run_eagerly=False,
    ):
        """
        Configures the model for training

        Parameters
        ----------
        rate: float
            learning rate for optimizer
        optimizer: str
            name of optimizer
        loss_func: str
            name of loss function
        metric_funcs: list[str]
            list with metric function names
        run_eagerly: bool

        Returns
        -------

        """
        opt = optimizers.get_optimizer(optimizer)(learning_rate=rate)
        loss = losses.get_loss(loss_func)
        m = [metrics.get_metric(metric) for metric in metric_funcs]
        self.compile(
            optimizer=opt,
            loss=loss,
            metrics=m,
            run_eagerly=run_eagerly,
        )

    def call(self, inputs, **kwargs):
        """
        Obtaining a neural network response on the input data vector
        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """
        x = inputs
        for layer in self.blocks:
            x = layer(x, **kwargs)
        return self.out_layer(x, **kwargs)

    def train_step(self, data):
        """
        Custom train step from tensorflow tutorial

        Parameters
        ----------
        data: tuple
            Pair of x and y (or dataset)
        Returns
        -------

        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def set_name(self, new_name):
        self._name = new_name

    def __str__(self):
        res = f"IModel {self.name}\n"
        for layer in self.blocks:
            res += str(layer)
        res += str(self.out_layer)
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
        res = {
            "net_type": "MyDense",
            "name": self._name,
            "input_size": self.input_size,
            "block_size": self.block_size,
            "output_size": self.output_size,
            "layer": [],
            "out_layer": self.out_layer.to_dict(),
        }

        for i, layer in enumerate(self.blocks):
            res["layer"].append(layer.to_dict())

        return res

    @classmethod
    def from_layers(
        cls,
        input_size: int,
        block_size: List[int],
        output_size: int,
        layers: List[MyDense],
        **kwargs,
    ):
        """
        Restore neural network from list of layers
        Parameters
        ----------
        input_size
        block_size
        output_size
        layers
        kwargs

        Returns
        -------

        """
        res = cls(
            input_size=input_size,
            block_size=block_size,
            output_size=output_size,
            **kwargs,
        )

        for layer_num in range(len(res.blocks)):
            res.blocks[layer_num] = layers[layer_num]

        return res

    def from_dict(self, config, **kwargs):
        """
        Restore neural network from dictionary of params
        Parameters
        ----------
        config
        kwargs

        Returns
        -------

        """
        input_size = config["input_size"]
        block_size = config["block_size"]
        output_size = config["output_size"]

        layers: List[MyDense] = []
        for layer_config in config["layer"]:
            layers.append(layer_creator.from_dict(layer_config))

        for layer_num in range(len(self.blocks)):
            self.blocks[layer_num] = layers[layer_num]

        self.out_layer = layer_creator.from_dict(config["out_layer"])

    @property
    def get_activations(self) -> List:
        """
        Get list of activations functions for each layer

        Returns
        -------
        activation: list
        """
        return self.activation_funcs.copy()
