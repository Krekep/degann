import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

import os
from typing import List, Optional, Dict, Callable

from degann.networks.config_format import LAYER_DICT_NAMES
from degann.networks import layer_creator, losses, metrics, cpp_utils
from degann.networks import optimizers
from degann.networks.layers.tf_dense import TensorflowDense
from degann.networks.topology.convolutionalnet.topology_config import (
    ConvolutionalNetParams,
)
from degann.networks.topology.convolutionalnet.compile_config import (
    ConvolutionalNetCompileParams,
)


class TensorflowConvolutionNet(tf.keras.Model):
    def __init__(
        self,
        config: ConvolutionalNetParams = ConvolutionalNetParams(),
        **kwargs,
    ):
        # input data validation
        print(config.convolution_block_sizes)
        print(config.convolution_block_types)
        print(config.block_size)
        assert len(config.convolution_block_types) == len(
            config.convolution_block_sizes
        ), "Sizes of convolutional types array and convolutional sizes array must be the same"

        # model initialisation
        super(TensorflowConvolutionNet, self).__init__()
        self.blocks = []
        self.output_size = config.output_size
        self.chunk_size = config.chunk_size
        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}

        for conv_layer_number in range(len(config.convolution_block_types)):
            match config.convolution_block_types[conv_layer_number]:
                case "maxPooling":
                    self.blocks.append(
                        layers.MaxPooling2D(
                            (config.convolution_block_sizes[conv_layer_number], 1)
                        )
                    )
                case "conv":
                    self.blocks.append(
                        layers.Conv2D(
                            config.convolution_block_sizes[conv_layer_number],
                            config.convolution_core_size,
                            activation=config.convolutional_activation_func,
                            padding=config.padding_type,
                        )
                    )
        self.blocks.append(layers.Flatten())
        for dense_layer_number in range(len(config.block_size)):
            self.blocks.append(
                layers.Dense(
                    config.block_size[dense_layer_number],
                    activation=config.dense_activation_func,
                ),
            )
        self.out_layer = layers.Dense(config.output_size, activation="linear")

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

    def set_name(self, new_name):
        self._name = new_name

    def split_data(self, x, y):
        """
        функция принимает на вод данные и разделяет
        их на чанки для обучения CNN
        """
        x_data = np.array(
            [x[i : i + self.chunk_size] for i in range(0, len(x), self.chunk_size)]
        )[..., tf.newaxis]
        y_data = np.array(
            [y[i : i + self.chunk_size] for i in range(0, len(y), self.chunk_size)]
        )
        return [x_data, y_data]

    def fit(self, x_data, y_data, *args, **kwargs):
        x_data, y_data = self.split_data(x_data, y_data)
        super().fit(x_data, y_data, *args, **kwargs)

    def predict(self, input, *args, **kwargs):
        input, _ = self.split_data(input, np.array([]))
        return super().predict(input, *args, **kwargs)

    def __str__(self):
        res = f"IModel {self.name}\n"
        for layer in self.blocks:
            res += str(layer)
        res += str(self.out_layer)
        return res

    def custom_compile(
        self, config: ConvolutionalNetCompileParams = ConvolutionalNetCompileParams()
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
        self.compile(
            optimizer=opt,
            loss=loss,
            metrics=m,
            run_eagerly=config.run_eagerly,
        )
