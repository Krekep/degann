from concurrent.futures import ProcessPoolExecutor
import datetime
import math
import os
import random
import time
from typing import List, Optional, Dict, Callable

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import mlflow

from degann.geometry import RectangleDomain, Decomposition, Block
from degann.networks.config_format import LAYER_DICT_NAMES
from degann.networks import layer_creator, losses, metrics, cpp_utils
from degann.networks import optimizers
from degann.networks.layers.tf_dense import TensorflowDense
from degann.networks.topology import TensorflowDenseNet
from degann.networks.topology.pinn import PhysicsInformedNet
from examples.fbpinn_tests.plot_functions import plot_each_submodel, plot_model


class TensorflowFBPINN(tf.keras.Model):
    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 10,
        activation_func: str | list[str] = "linear",
        weight=keras.initializers.RandomUniform(minval=-1, maxval=1),
        biases=keras.initializers.RandomUniform(minval=-1, maxval=1),
        is_debug: bool = False,
        models_size=[10],
        domain=RectangleDomain([0], [1]),
        block_size=0.2,
        overlap=0.05,
        physic_loss=None,
        boundary_loss: list[tuple[Callable, list[float]]] = None,
        offset=False,
        points_per_block=50,
        summary_writer=None,
        time_input: bool = False,
        end_time: float = 1.0,
        time_step: float = 1.0,
        losses_weight: list = None,
        **kwargs,
    ):
        super(TensorflowFBPINN, self).__init__(**kwargs)
        self.networks: list

        domain: RectangleDomain = domain
        block_size: float = block_size
        overlap: float = overlap
        self.physic_loss: Callable = physic_loss

        self.decomposition = Decomposition(
            domain=domain,
            block_size=block_size,
            overlap=overlap,
            offset=offset,
            points_per_block=points_per_block,
        )
        number_of_networks = len(self.decomposition.blocks)
        if losses_weight is None:
            losses_weight = [1.0] * (len(boundary_loss) + 1)
        for block in self.decomposition.blocks:
            block.set_losses(
                boundary_loss + [(physic_loss, [None])], time_input=time_input
            )
        self.networks = []
        self.phys_losses = list(map(lambda x: x[0], boundary_loss)) + [physic_loss]
        for i, block in enumerate(self.decomposition.blocks):
            # nn = TensorflowDenseNet(input_size=input_size, block_size=[16, 16], output_size=output_size, activation_func=["tanh", "tanh", "linear"])
            nn = PhysicsInformedNet(
                input_size=input_size,
                block_size=models_size,
                output_size=output_size,
                phys_func=block.losses[-1],
                boundary_func=block.losses[:-1],
                activation_func=activation_func,
                weight=keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
                biases=keras.initializers.Zeros(),
                # activation_func=["swish", "relu", "linear"],
                # decorator_params=[None, {"negative_slope": -0.5, "max_value": 10}, None]
                domain_block=block,
            )
            self.networks.append(nn)

        self.blocks: list[tuple[PhysicsInformedNet, Block]] = list(
            zip(self.networks, self.decomposition.blocks)
        )
        self.input_size = input_size
        self.output_size = output_size
        self.summary_writer = summary_writer

        # [(Loss, [index Model1, index Model2, ...]), (Other loss, [index Model1, index Model5, ...])]
        self.model_per_loss = []
        for i, (loss, point) in enumerate(boundary_loss + [(physic_loss, None)]):
            if point is None:
                self.model_per_loss.append(
                    (loss, list(range(0, len(self.blocks))), losses_weight[i])
                )
            else:
                models = []
                for j, (nn, block) in enumerate(self.blocks):
                    if time_input:
                        left_down_corner = [-math.inf] + block.left_down_corner
                        right_up_corner = [math.inf] + block.right_up_corner
                    else:
                        left_down_corner = block.left_down_corner
                        right_up_corner = block.right_up_corner
                    fl = False
                    for lc, var_b, rc in zip(left_down_corner, point, right_up_corner):
                        if var_b is not None:
                            for b in var_b:
                                if lc <= b and b <= rc:
                                    fl = True
                                    break
                    if fl:
                        models.append(j)
                self.model_per_loss.append((loss, models, losses_weight[i]))

        self.time_input = time_input
        self.time_step = time_step
        self.end_time = end_time
        self.build_data()
        self.build_time_vectors()

    def custom_compile(
        self,
        rate: float = 1e-2,
        optimizer: str | tf.keras.optimizers.Optimizer = "SGD",
        inner_loss_func: str | tf.keras.losses.Loss = "MeanSquaredError",
        loss_func: str | tf.keras.losses.Loss = "MeanSquaredError",
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
        inner_loss_func: str
            name of loss function for each network in blocks
        loss_func: str
            name of loss function
        metric_funcs: list[str]
            list with metric function names
        run_eagerly: bool

        Returns
        -------

        """
        for nn in self.networks:
            nn.custom_compile(
                optimizer=optimizer,
                rate=rate,
                loss_func=inner_loss_func,
                metric_funcs=metric_funcs,
                run_eagerly=run_eagerly,
            )
        opt = optimizers.get_optimizer(optimizer)(learning_rate=rate)
        loss = losses.get_loss(loss_func)
        m = (
            [metrics.get_metric(metric) for metric in metric_funcs]
            if metric_funcs is not None
            else None
        )
        self.optimizer_ = optimizer
        self.rate_ = rate
        self.loss_func_ = loss_func
        self.metric_funcs_ = metric_funcs
        self.compile(
            optimizer=opt,
            loss=loss,
            metrics=m,
            run_eagerly=run_eagerly,
        )

    def call(self, x, active_models: list = None, **kwargs):
        """
        Obtaining a neural network response on the input data vector
        Parameters
        ----------
        inputs
        kwargs

        Returns
        -------

        """
        fbpinn_predict: tf.Tensor = tf.zeros(shape=self.output_size)

        active_models = active_models if active_models is not None else self.blocks
        for nn, block in active_models:
            x_norm = block.normalization(x)
            predicted = nn(x_norm)
            predicted_unnorm: tf.Tensor = block.unnormalization(predicted)
            if self.time_input:
                windowed = block.window_function(x[:, 1:])
            else:
                windowed = block.window_function(x)
            result = windowed * predicted_unnorm
            fbpinn_predict = fbpinn_predict + result

        return fbpinn_predict

    def build_data(self):
        temp = []
        for i, block in enumerate(self.decomposition.blocks):
            block.data = tf.reshape(
                tf.convert_to_tensor(block.get_data(), dtype=tf.float32),
                shape=block.data.shape,
            )
            temp.append(block.data)
        self.data = tf.concat(temp, axis=0)

    def build_time_vectors(self):
        self.time_data = []
        if self.time_input:
            t = 0
            while t <= self.end_time:
                self.time_data.append(
                    tf.constant(t, shape=self.data.shape, dtype=tf.float32)
                )
                t += self.time_step
        else:
            self.time_data = [None]

    def load_model(self, model_type, model_state, trainable, block):
        nn = model_type.from_dict_cls(model_state, block=block)
        # nn.block = block

        nn.trainable = trainable
        nn.custom_compile(
            optimizer=self.optimizer_,
            rate=self.rate_,
            loss_func=self.loss_func_,
            metric_funcs=self.metric_funcs_,
            run_eagerly=self.run_eagerly,
        )
        return nn

    def split_to_batches(self, input_data, batch_size):
        # input_data = tf.random.shuffle(input_data)
        # n = input_data.shape[0]
        # k = batch_size
        # num_batches = n // k
        # remainder = n % k

        # # Полные батчи
        # if num_batches > 0:
        #     full_batches = tf.split(input_data[: num_batches * k], num_batches)
        #     if remainder != 0:
        #         full_batches.append(input_data[num_batches * k :])

        #     return full_batches
        return [input_data]

    def log_weights(self, epoch, model):
        with self.summary_writer.as_default():
            for layer in model.layers:
                if hasattr(layer, "w"):
                    # Логирование весов (kernel)
                    tf.summary.histogram(
                        f"{model.name}/{layer.name}/w", layer.w, step=epoch
                    )
                if hasattr(layer, "b"):
                    # Логирование смещений (bias)
                    tf.summary.histogram(
                        f"{model.name}/{layer.name}/b", layer.b, step=epoch
                    )

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        y_pred = self(x)
        square = tf.math.square(y - y_pred)
        return tf.math.reduce_mean(square)

    @tf.function
    def get_val_score(self, val_input, val_function):
        y_pred = self.call(val_input)
        y_true = val_function(val_input)
        val_mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        val_mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        val_l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred)) / tf.reduce_mean(
            tf.abs(y_true)
        )
        return y_pred, y_true, val_mse_loss, val_mae_loss, val_l1_loss

    def log_validation_metrics(self, t, val_input, ode, epoch, loss, epoch_time):
        if t[0] is not None:
            t_val = tf.fill([5000, 1], t[0])
            x = tf.concat([t_val, val_input], axis=1)
        else:
            x = val_input
        (
            y_pred,
            y_true,
            val_mse_loss,
            val_mae_loss,
            val_l1_loss,
        ) = self.get_val_score(x, ode.solution)
        mlflow.log_metric("Validation MSE loss", val_mse_loss, step=epoch)
        mlflow.log_metric("Validation MAE loss", val_mae_loss, step=epoch)
        mlflow.log_metric("Validation Relative L1Loss", val_l1_loss, step=epoch)
        mlflow.log_metric("Epoch time", epoch_time, step=epoch)
        mlflow.log_metric("Loss", loss, step=epoch)
        mlflow.log_metric("Learning rate", self.optimizer.learning_rate, step=epoch)

    def log_graphics(self, t, val_input, ode, epoch, loss, png_salt):
        if t[0] is not None:
            t_val = tf.fill([5000, 1], t[0])
            x = tf.concat([t_val, val_input], axis=1)
        else:
            x = val_input
        (
            y_pred,
            y_true,
            val_mse_loss,
            val_mae_loss,
            val_l1_loss,
        ) = self.get_val_score(x, ode.solution)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        plot_each_submodel(x, val_input[:, 1], y_true, self, axes[0])
        plot_model(x, val_input[:, 1], y_true, self, axes[1])
        plt.savefig(
            f"FBPINN_{png_salt}_{epoch}_t{t[0]}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        for nn, block in self.blocks:
            self.log_weights(epoch, nn)

        with tf.GradientTape() as tape:
            tape.watch(x)
            u = self(x)
            u_x = tape.gradient(u, x)
            original_dx = ode.first_der("x", val_input[:, 0], val_input[:, 1])
            original_dt = ode.first_der("t", val_input[:, 0], val_input[:, 1])
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
            axes[0].plot(
                val_input[:, 1], original_dx, label="Truth x derivative", color="red"
            )
            axes[0].plot(
                val_input[:, 0], original_dt, label="Truth t derivative", color="black"
            )
            axes[0].plot(
                val_input[:, 1], u_x[:, 1], label="Model x derivative", color="green"
            )
            axes[0].plot(
                val_input[:, 0], u_x[:, 0], label="Model t derivative", color="purple"
            )
            axes[1].plot(
                val_input[:, 1],
                original_dx - u_x[:, 1],
                label="Difference x",
                color="blue",
            )
            axes[1].plot(
                val_input[:, 0],
                original_dt - u_x[:, 0],
                label="Difference t",
                color="orange",
            )
            axes[0].grid()
            axes[1].grid()
            axes[1].legend()
            axes[0].legend()
            plt.savefig(
                f"FBPINN_{png_salt}_d_{epoch}_t{t[0]}.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)

        print(
            f"Epoch {epoch}, loss {loss}, val mse loss {val_mse_loss}, mae loss {val_mae_loss}, rel l1loss {val_l1_loss}. {datetime.datetime.now()}"
        )

    def train(
        self,
        epochs,
        verbose,
        callbacks,
        ode,
        patience=300,
        log_interval=100,
        eval_interval=1,
        batch_size=10,
        mode="sequence",
        val_input=None,
        png_salt="",
        noise=None,
        blocks_per_layer=None,
        epoch_before_increase=None,
    ):
        self.build_data()
        self.build_time_vectors()
        if noise is None:
            noise = tf.constant(0, shape=self.data.shape, dtype=tf.float32)
        if mode == "sequence":
            self.sequence_train(
                epochs,
                verbose,
                callbacks,
                ode,
                patience,
                log_interval,
                eval_interval,
                batch_size,
            )
        elif mode == "full":
            self.full_train(
                epochs,
                verbose,
                callbacks=callbacks,
                ode=ode,
                patience=patience,
                log_interval=log_interval,
                eval_interval=eval_interval,
                batch_size=batch_size,
                val_input=val_input,
                png_salt=png_salt,
                noise=noise,
            )
        elif mode == "layer":
            self.layer_train(
                epochs,
                verbose,
                callbacks=callbacks,
                ode=ode,
                patience=patience,
                log_interval=log_interval,
                eval_interval=eval_interval,
                batch_size=batch_size,
                val_input=val_input,
                png_salt=png_salt,
                noise=noise,
                epoch_before_increase=epoch_before_increase,
                blocks_per_layer=blocks_per_layer,
            )
        else:
            raise ValueError("Unsupported train mode")

    def layer_train(
        self,
        epochs,
        verbose,
        val_input,
        ode,
        patience=300,
        callbacks=None,
        log_interval=100,
        eval_interval=1,
        batch_size=10,
        png_salt="",
        epoch_before_increase=2000,
        noise=0,
        blocks_per_layer=None,
    ):
        curr_patience = 0
        best_loss = 1e6
        best_val_loss = 1e6
        if noise is None:
            noise = tf.constant(0, shape=self.data.shape, dtype=tf.float32)

        if blocks_per_layer is None:
            blocks_per_axis = self.decomposition.blocks_per_axis
            blocks_per_layer = sum(blocks_per_axis[:-1])
        layer_start = 0
        layer_end = blocks_per_layer
        counter = 0
        print(f"Epoch {0}, layer start {layer_start}, layer end {layer_end}")
        for epoch in range(epochs):
            start_time = time.perf_counter()
            data = []
            for i in range(layer_start, layer_end):
                data.append(self.blocks[i][1].get_data())
            data = tf.concat(data, axis=0)

            blocks_data = data
            # blocks_data = self.data
            for t in self.time_data:
                log_t = [None]
                if t is not None:
                    data = tf.concat([t, blocks_data], axis=1)
                    log_t = t
                else:
                    data = blocks_data
                # batches = self.split_to_batches(data, batch_size)
                loss = 0
                # for batch in batches:
                if epoch == 0:
                    metrics = self.train_step(data, noise, 0, len(self.blocks))
                else:
                    metrics = self.train_step(data, noise, layer_start, layer_end)
                loss += metrics["loss"]
                if loss < best_loss:
                    best_loss = loss
                    curr_patience = 0
                curr_patience += 1
                if curr_patience > patience:
                    break
            end_time = time.perf_counter()
            if epoch == 0:
                end_time = start_time
            if epoch % eval_interval == 0:
                self.log_validation_metrics(
                    log_t, val_input, ode, epoch, loss, end_time - start_time
                )
            if epoch % log_interval == 0:
                self.log_graphics(log_t, val_input, ode, epoch, loss, png_salt)
                print(f"Epoch {epoch}, data length {len(blocks_data)}")
            if (epoch + 1) % epoch_before_increase == 0:
                if layer_end < len(self.blocks) - 1:
                    if counter == 0:
                        counter += 1
                    else:
                        layer_start = layer_start + blocks_per_layer
                    layer_end = layer_end + blocks_per_layer
                    print(
                        f"Epoch {epoch}, layer start {layer_start}, layer end {layer_end}"
                    )
                else:
                    layer_start = 0
                    print(
                        f"Epoch {epoch}, layer start {layer_start}, layer end {layer_end}"
                    )
            if loss < best_loss:
                best_loss = loss
                curr_patience = 0
            curr_patience += 1
            if curr_patience > patience:
                break
        self.log_graphics(log_t, val_input, ode, epoch, loss, png_salt)

    def full_train(
        self,
        epochs,
        verbose,
        val_input,
        ode,
        patience=300,
        callbacks=None,
        log_interval=100,
        eval_interval=1,
        batch_size=10,
        png_salt="",
        noise=None,
        **kwargs,
    ):
        curr_patience = 0
        best_loss = 1e6
        self.ode = ode
        if noise is None:
            noise = tf.constant(0, shape=self.data.shape, dtype=tf.float32)

        for epoch in range(epochs):
            start_time = time.perf_counter()
            # blocks_data = []
            # for nn, block in self.blocks:
            #     blocks_data.append(block.get_data())
            # blocks_data = tf.concat(blocks_data, axis=0)
            blocks_data = self.data
            for t in self.time_data:
                log_t = [None]
                if t is not None:
                    data = tf.concat([t, blocks_data], axis=1)
                    log_t = t
                else:
                    data = blocks_data
                batches = self.split_to_batches(data, batch_size)
                loss = 0
                for batch in batches:
                    metrics = self.train_step(batch, noise, 0, len(self.blocks))
                    loss += metrics["loss"]
                if loss < best_loss:
                    best_loss = loss
                    curr_patience = 0
                curr_patience += 1
                if curr_patience > patience:
                    break
            end_time = time.perf_counter()
            if epoch == 0:
                end_time = start_time
            if epoch % eval_interval == 0:
                self.log_validation_metrics(
                    log_t, val_input, ode, epoch, loss, end_time - start_time
                )
            if epoch % log_interval == 0:
                self.log_graphics(log_t, val_input, ode, epoch, loss, png_salt)
        self.log_graphics(log_t, val_input, ode, epoch, loss, png_salt)

    # @tf.function
    def train_step(self, data, noise, layer_start, layer_end):
        """
        Custom train step from tensorflow tutorial

        Parameters
        ----------
        data: tuple
            Pair of x and y (or dataset)
        Returns
        -------

        """

        def get_active_variables(active_models):
            variables = []
            for nn, block in active_models:
                variables.extend(nn.trainable_variables)
            # variables.extend(self.trainable_variables)
            return variables

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        with tf.GradientTape(persistent=True) as tape:
            with tf.device("/GPU:0"):
                x = tf.identity(data)
            loss: tf.Tensor = tf.zeros(shape=1)
            for loss_func, models, loss_weigth in self.model_per_loss:
                active_models = []
                for model in models:
                    if layer_start <= model <= layer_end:
                        active_models.append(self.blocks[model])
                if len(active_models) > 0:
                    loss += (
                        loss_func(
                            self, tape, x, noise=noise, active_models=active_models
                        )
                        * loss_weigth
                    )

        # Compute gradients
        trainable_vars = get_active_variables(active_models)
        gradients = tape.gradient(loss, trainable_vars)
        # # Update weights
        # self.optimizer.apply(gradients, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            # elif y is not None:
            #     metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def set_name(self, new_name):
        raise NotImplementedError("This method not implemented")

    def __str__(self):
        raise NotImplementedError("This method not implemented")

    def to_dict(self, **kwargs):
        """
        Export neural network to dictionary

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        raise NotImplementedError("This method not implemented")

    @classmethod
    def from_layers(
        cls,
        input_size: int,
        block_size: List[int],
        output_size: int,
        layers: List[TensorflowDense],
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
        raise NotImplementedError("This method not implemented")

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
        raise NotImplementedError("This method not implemented")

    def export_to_cpp(
        self,
        path: str,
        array_type: str = "[]",
        path_to_compiler: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Export neural network as feedforward function on c++

        Parameters
        ----------
        path: str
            path to file with name, without extension
        array_type: str
            c-style or cpp-style ("[]" or "vector")
        path_to_compiler: str
            path to c/c++ compiler
        kwargs

        Returns
        -------

        """
        raise NotImplementedError("This method not implemented")

    @property
    def get_activations(self) -> List:
        """
        Get list of activations functions for each layer

        Returns
        -------
        activation: list
        """
        raise NotImplementedError("This method not implemented")
