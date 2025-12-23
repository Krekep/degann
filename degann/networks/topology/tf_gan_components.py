from typing import List
import tensorflow as tf
from tensorflow import keras

from degann.networks import layer_creator


class TensorflowGenerator(tf.keras.Model):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        block_sizes: List[int],
        activation_funcs: List[str],
        out_activation: str,
        weight_init,
        bias_init,
        is_debug: bool = False,
        name: str = "TensorflowGenerator",
        **kwargs,
    ):
        super(TensorflowGenerator, self).__init__(name=name, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.block_sizes = block_sizes
        self.activation_funcs = activation_funcs
        self.is_debug = is_debug

        self.blocks = []
        prev_size = input_size
        for i, (size, act_func) in enumerate(zip(block_sizes, activation_funcs)):
            layer = layer_creator.create_dense(
                inp_size=prev_size,
                shape=size,
                activation=act_func,
                weight=weight_init,
                bias=bias_init,
                is_debug=is_debug,
                name=f"GenDense{i}",
            )
            self.blocks.append(layer)
            prev_size = size

        self.out_layer = layer_creator.create_dense(
            inp_size=prev_size,
            shape=output_size,
            activation=out_activation,
            weight=weight_init,
            bias=bias_init,
            is_debug=is_debug,
            name="GenOutputLayer",
        )

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.blocks:
            x = layer(x, **kwargs)
        x = self.out_layer(x, **kwargs)
        return x

    def to_dict(self, **kwargs):
        res = {
            "net_type": "TFGenerator",
            "name": self.name,
            "input_size": self.input_size,
            "block_sizes": self.block_sizes,
            "output_size": self.output_size,
            "activation_funcs": self.activation_funcs,
            "out_activation": self.out_layer.get_activation,
            "layer": [],
            "out_layer": self.out_layer.to_dict(),
        }
        for i, layer in enumerate(self.blocks):
            res["layer"].append(layer.to_dict())
        return res

    def from_dict(self, config):
        layers: List = []
        for layer_config in config["layer"]:
            layers.append(layer_creator.from_dict(layer_config))

        self.blocks = layers
        self.out_layer.from_dict(config["out_layer"])

    @property
    def get_activations(self):
        inner_acts = [layer.get_activation for layer in self.blocks]
        outer_act = self.out_layer.get_activation
        return inner_acts + [outer_act]

    def __str__(self) -> str:
        res = f"Generator {self.name}\n"
        res += f"  Input size: {self.input_size}\n"
        res += f"  Output size: {self.output_size}\n"
        res += f"  Block sizes: {self.block_sizes}\n"
        res += f"  Activation funcs: {self.activation_funcs}\n"
        res += f"  Output activation: {self.out_layer.get_activation}\n"
        res += f"  Number of internal layers: {len(self.blocks)}\n"
        return res


class TensorflowDiscriminator(tf.keras.Model):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        block_sizes: List[int],
        activation_funcs: List[str],
        weight_init,
        bias_init,
        is_debug: bool = False,
        name: str = "TensorflowDiscriminator",
        **kwargs,
    ):
        super(TensorflowDiscriminator, self).__init__(name=name, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.block_sizes = block_sizes
        self.activation_funcs = activation_funcs
        self.is_debug = is_debug

        self.blocks = []
        prev_size = input_size
        for i, (size, act_func) in enumerate(zip(block_sizes, activation_funcs)):
            layer = layer_creator.create_dense(
                inp_size=prev_size,
                shape=size,
                activation=act_func,
                weight=weight_init,
                bias=bias_init,
                is_debug=is_debug,
                name=f"DiscDense{i}",
            )
            self.blocks.append(layer)
            prev_size = size

        self.out_layer = layer_creator.create_dense(
            inp_size=prev_size,
            shape=output_size,
            activation="linear",
            weight=weight_init,
            bias=bias_init,
            is_debug=is_debug,
            name="DiscOutputLayer",
        )

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.blocks:
            x = layer(x, **kwargs)
        x = self.out_layer(x, **kwargs)
        return x

    def to_dict(self, **kwargs):
        res = {
            "net_type": "TFDiscriminator",
            "name": self.name,
            "input_size": self.input_size,
            "block_sizes": self.block_sizes,
            "output_size": self.output_size,
            "activation_funcs": self.activation_funcs,
            "out_activation": "linear",
            "layer": [],
            "out_layer": self.out_layer.to_dict(),
        }
        for i, layer in enumerate(self.blocks):
            res["layer"].append(layer.to_dict())
        return res

    def from_dict(self, config):
        layers: List = []
        for layer_config in config["layer"]:
            layers.append(layer_creator.from_dict(layer_config))

        self.blocks = layers
        self.out_layer.from_dict(config["out_layer"])

    @property
    def get_activations(self):
        inner_acts = [layer.get_activation for layer in self.blocks]
        outer_act = self.out_layer.get_activation
        return inner_acts + [outer_act]

    def __str__(self) -> str:
        res = f"Discriminator {self.name}\n"
        res += f"  Input size: {self.input_size}\n"
        res += f"  Output size: {self.output_size}\n"
        res += f"  Block sizes: {self.block_sizes}\n"
        res += f"  Activation funcs: {self.activation_funcs}\n"
        res += f"  Output activation: {self.out_layer.get_activation}\n"
        res += f"  Number of internal layers: {len(self.blocks)}\n"
        return res
