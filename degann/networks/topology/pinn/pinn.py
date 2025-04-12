from typing import Optional, List, Dict
import tensorflow as tf

from degann.networks import layer_creator, losses, metrics, optimizers
from degann.networks.topology.pinn.compile_config import PINNCompileParams
from degann.networks.topology.pinn.topology_config import PINNParams
from degann.networks.topology.pinn.virtual_loss import VirtualLoss



class PhysicsInformedNet(tf.keras.Model):
    def __init__(self, config: Optional[PINNParams] = None, **kwargs):
        if config is None:
            config = PINNParams()
        
        decorator_params: List[Optional[Dict]] = [None]
        if "decorator_params" in kwargs.keys():
            value = kwargs.get("decorator_params")
            if isinstance(value, list) and all(
                isinstance(item, dict) for item in value
            ):
                decorator_params = value
            kwargs.pop("decorator_params")
        else:
            decorator_params = [None]
        super(PhysicsInformedNet, self).__init__(**kwargs)

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is None
            or decorator_params is None
        ):
            decorator_params = [None] * (len(config.block_size) + 1)

        if (
            isinstance(decorator_params, list)
            and len(decorator_params) == 1
            and decorator_params[0] is not None
        ):
            decorator_params = decorator_params * (len(config.block_size) + 1)

        self.blocks: List[tf.keras.layers.Layer] = []

        if not isinstance(config.activation_func, list):
            activation_func_list = [config.activation_func] * (
                len(config.block_size) + 1
            )
        else:
            activation_func_list = config.activation_func.copy()

        if len(config.block_size) != 0:
            self.blocks.append(
                layer_creator.create_dense(
                    config.input_size,
                    config.block_size[0],
                    activation=activation_func_list[0],
                    weight=config.weight,
                    bias=config.biases,
                    is_debug=config.is_debug,
                    name=f"PINN0",
                    decorator_params=decorator_params[0],
                )
            )
            for i in range(1, len(config.block_size)):
                self.blocks.append(
                    layer_creator.create_dense(
                        config.block_size[i - 1],
                        config.block_size[i],
                        activation=activation_func_list[i],
                        weight=config.weight,
                        bias=config.biases,
                        is_debug=config.is_debug,
                        name=f"PINN{i}",
                        decorator_params=decorator_params[i],
                    )
                )
            last_block_size = config.block_size[-1]
        else:
            last_block_size = config.input_size

        self.out_layer = layer_creator.create_dense(
            last_block_size,
            config.output_size,
            activation=activation_func_list[-1],
            weight=config.weight,
            bias=config.biases,
            is_debug=config.is_debug,
            name=f"OutLayerPINN",
            decorator_params=decorator_params[-1],
        )

        self.activation_funcs = activation_func_list
        self.weight_initializer = config.weight
        self.bias_initializer = config.biases
        self.input_size = config.input_size
        self.block_size = config.block_size
        self.output_size = config.output_size
        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}
        self.virtual_functions: Optional[List[VirtualLoss]] = None

    def custom_compile(
        self, config: Optional[PINNCompileParams]
    ) -> None:
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
        opt = (
            optimizers.get_optimizer(config.optimizer)(learning_rate=config.rate)
            if isinstance(config.optimizer, str)
            else config.optimizer
        )
        loss = (
            losses.get_loss(config.loss_func)
            if isinstance(config.loss_func, str)
            else config.loss_func
        )
        m = [metrics.get_metric(metric) for metric in config.metric_funcs]
        
        self.virtual_functions = config.virtual_functions
        self.compile(
            optimizer=opt,
            loss=loss,
            metrics=m,
            run_eagerly=config.run_eagerly,
        )

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
        x = inputs
        if training is None:
            training = False
        for layer in self.blocks:
            x = layer(x, training=training, mask=mask)
        return self.out_layer(x, training=training, mask=mask)


    def train_step(self, data: tuple[tf.Tensor, tf.Tensor]):  # type: ignore
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
        print("1111111111111111", data)
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y_pred: tf.Tensor = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            total_loss = self.compute_loss(y=y, y_pred=y_pred)
            if total_loss is None:
                total_loss = tf.constant(0, dtype=tf.float32)
            for virtual_function in  self.virtual_functions:
                virtual_deviation = virtual_function(self, tape, x)
                virtual_loss = self.compiled_loss(
                    tf.zeros_like(virtual_deviation), virtual_deviation
                )
                total_loss += virtual_loss * virtual_function.weight

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        # Update weights
        if self.optimizer is None:
            raise RuntimeError("compile or custom_compile must be called before train")
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        del tape
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def train(self,)

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
            "net_type": "MyPINN",
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
        layers: List[tf.keras.layers.Layer],
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

    def from_dict(self, config: dict, **kwargs):
        """
        Restore neural network from dictionary of params

        Parameters
        ----------
        config: dict
            Model parameters

        """
        input_size = config["input_size"]
        block_size = config["block_size"]
        output_size = config["output_size"]

        self.block_size = list(block_size)
        self.input_size = input_size
        self.output_size = output_size

        layers: List[tf.keras.layers.Layer] = []
        for layer_config in config["layer"]:
            layers.append(layer_creator.from_dict(layer_config))

        self.blocks.clear()
        for layer_num in range(len(layers)):
            self.blocks.append(layers[layer_num])

        self.out_layer = layer_creator.from_dict(config["out_layer"])

    @property
    def get_activations(self) -> List:
        """
        Get list of activations functions for each layer
        Returns
        -------
        activation: list
        """
        return [layer.get_activation for layer in self.blocks]
