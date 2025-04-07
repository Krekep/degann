from concurrent.futures import ProcessPoolExecutor
import datetime
import os
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
        for block in self.decomposition.blocks:
            block.set_losses(boundary_loss + [(physic_loss, [None])])
        self.networks = []
        self.phys_losses = list(map(lambda x: x[0], boundary_loss)) + [physic_loss]
        temp = []
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
            block.data = tf.reshape(
                tf.convert_to_tensor(block.get_data(), dtype=tf.float32), shape=(-1, 1)
            )
            temp.append(block.data)
        self.data = tf.concat(temp, axis=0)

        self.blocks: list[tuple[PhysicsInformedNet, Block]] = list(
            zip(self.networks, self.decomposition.blocks)
        )
        self.input_size = input_size
        self.output_size = output_size
        self.summary_writer = summary_writer

        # [(Loss, [Model1, Model2, ...]), (Other loss, [Model1, Model5, ...])]
        self.model_per_loss = []
        for loss, point in boundary_loss + [(physic_loss, None)]:
            if point is None:
                self.model_per_loss.append((loss, [None]))
            else:
                models = []
                for i, (nn, block) in enumerate(self.blocks):
                    if (
                        block.left_down_corner <= point
                        and point <= block.right_up_corner
                    ):
                        models.append(nn)
                self.model_per_loss.append((loss, models))

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
        # outputs = [self.process_single_block(inputs, nn, block) for nn, block in self.blocks]
        # return tf.add_n(outputs)

        fbpinn_predict: tf.Tensor = tf.zeros(shape=self.output_size)
        for nn, block in self.blocks:
            x_norm = block.normalization(inputs)
            predicted = nn(x_norm)
            predicted_unnorm: tf.Tensor = block.unnormalization(predicted)
            windowed = block.window_function(inputs)
            result = windowed * predicted_unnorm
            fbpinn_predict = fbpinn_predict + result

        return fbpinn_predict

    @tf.function
    def process_single_block(self, inputs, nn, block):
        x_norm = block.normalization(inputs)
        predicted = nn(x_norm)
        predicted_unnorm = block.unnormalization(predicted)
        windowed = block.window_function(inputs)
        return windowed * predicted_unnorm

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
        input_data = tf.random.shuffle(input_data)
        n = input_data.shape[0]
        k = batch_size
        num_batches = n // k
        remainder = n % k

        # Полные батчи
        if num_batches > 0:
            full_batches = tf.split(input_data[: num_batches * k], num_batches)
            if remainder != 0:
                full_batches.append(input_data[num_batches * k :])

            return full_batches
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

    def train(
        self,
        epochs,
        verbose,
        callbacks,
        val_function,
        patience=300,
        log_interval=100,
        eval_interval=1,
        batch_size=10,
        mode="sequence",
        val_input=None,
        png_salt="",
    ):
        if mode == "sequence":
            self.sequence_train(
                epochs,
                verbose,
                callbacks,
                val_function,
                patience,
                log_interval,
                eval_interval,
                batch_size,
            )
        elif mode == "all":
            self.all_train(
                epochs,
                verbose,
                callbacks,
                val_function,
                patience,
                log_interval,
                eval_interval,
                batch_size,
            )
        elif mode == "compose":
            self.compose_train(
                epochs,
                verbose,
                callbacks,
                val_function,
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
                val_function=val_function,
                patience=patience,
                log_interval=log_interval,
                eval_interval=eval_interval,
                batch_size=batch_size,
                val_input=val_input,
                png_salt=png_salt,
            )
        else:
            raise ValueError("Unsupported train mode")

    def sequence_train(
        self,
        epochs,
        verbose,
        callbacks,
        val_function,
        patience=300,
        log_interval=100,
        eval_interval=1,
        batch_size=50,
    ):
        i = 0
        while i < len(self.blocks):
            nn, block = self.blocks[i]
            inputs = block.data

            best_loss = 1e6
            best_val_loss = 1e6
            best_weights = nn.get_weights()
            curr_patience = 0
            batches = self.split_to_batches(inputs, batch_size)
            for epoch in range(epochs):
                train_loss = 0
                for batch in batches:
                    if i == 0:
                        loss = nn.custom_train_step(batch, block, None, None)
                    else:
                        loss = nn.custom_train_step(
                            batch,
                            block=block,
                            prev_model=self.blocks[i - 1][0],
                            prev_block=self.blocks[i - 1][1],
                        )
                    train_loss += loss
                train_loss /= len(batches)

                if epoch % eval_interval == 0:
                    if i == 0:
                        val_loss = nn.get_val_score(
                            inputs, block, None, None, val_function
                        )
                    else:
                        val_loss = nn.get_val_score(
                            inputs,
                            block,
                            self.blocks[i - 1][0],
                            self.blocks[i - 1][1],
                            val_function,
                        )
                    mlflow.log_metric(
                        f"Validation loss model {i}", val_loss, step=epoch
                    )
                    mlflow.log_metric(f"Train loss model {i}", train_loss, step=epoch)

                    with self.summary_writer.as_default():
                        tf.summary.scalar(f"{nn.name}/loss", loss, step=epoch)
                        tf.summary.scalar(f"{nn.name}/val_loss", val_loss, step=epoch)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        mlflow.log_metric(
                            f"Best validation loss model {i}", best_val_loss, step=epoch
                        )
                        # best_weights = nn.get_weights()

                if epoch % log_interval == 0:
                    print(
                        f"Model {i}, epoch {epoch}, last loss {loss}, val_loss {val_loss}, best val {best_val_loss}"
                    )
                if loss < best_loss:
                    best_loss = loss
                    curr_patience = 0
                curr_patience += 1
                if curr_patience == patience:
                    break

            if i == 0:
                val_loss = nn.get_val_score(inputs, block, None, None, val_function)
            else:
                val_loss = nn.get_val_score(
                    inputs,
                    block,
                    self.blocks[i - 1][0],
                    self.blocks[i - 1][1],
                    val_function,
                )
            mlflow.log_metric(f"Validation loss model {i}", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mlflow.log_metric(
                    f"Best validation loss model {i}", best_val_loss, step=epoch
                )
                # best_weights = nn.get_weights()

            print(
                f"Model {i}, epoch {epoch}, last loss {loss}, val_loss {val_loss}, best val {best_val_loss}"
            )
            print()
            i += 1

    def all_train(
        self,
        epochs,
        verbose,
        callbacks,
        val_function,
        patience=300,
        log_interval=100,
        eval_interval=1,
        batch_size=10,
        epoch_before_increase=2000,
    ):
        curr_patience = 0
        best_loss = 1e6
        best_val_loss = 1e6
        best_states = [nn.to_dict() for i, (nn, block) in enumerate(self.blocks)]
        for i, (nn, block) in enumerate(self.blocks):
            inputs = block.data
            batches = self.split_to_batches(inputs, batch_size)
            batches_per_model.append(batches)

        last_train_index = 1
        for epoch in range(epochs):
            acc_loss = 0
            acc_val_loss = 0
            batches_per_model = []

            for i, (nn, block) in enumerate(self.blocks):
                if i < last_train_index:
                    batches = batches_per_model[i]
                    for j, batch in enumerate(batches):
                        if i == 0:
                            loss = nn.custom_train_step(
                                batch,
                                block,
                                None,
                                None,
                            )
                        else:
                            loss = nn.custom_train_step(
                                batch,
                                block=block,
                                prev_model=self.blocks[i - 1][0],
                                prev_block=self.blocks[i - 1][1],
                            )
                        mlflow.log_metric(
                            "Batch loss", loss, step=j + len(batches) * epoch
                        )

                    if epoch % eval_interval == 0:
                        if i == 0:
                            val_loss = nn.get_val_score(
                                inputs, block, None, None, val_function
                            )
                        else:
                            val_loss = nn.get_val_score(
                                inputs,
                                block,
                                self.blocks[i - 1][0],
                                self.blocks[i - 1][1],
                                val_function,
                            )
                        acc_val_loss += val_loss
                        mlflow.log_metric(
                            f"Validation loss model {i}", val_loss, step=epoch
                        )
                        with self.summary_writer.as_default():
                            tf.summary.scalar("loss", loss, step=epoch)
                            tf.summary.scalar("val_loss", val_loss, step=epoch)
                    acc_loss += loss
            acc_val_loss /= len(self.blocks)
            acc_loss /= len(self.blocks)
            mlflow.log_metric("Validation loss", acc_val_loss, step=epoch)
            mlflow.log_metric("Train loss", acc_loss, step=epoch)

            if epoch % patience == 0:
                print(
                    f"Epoch {epoch}, last loss {loss}, val_loss {val_loss}, best val_loss {best_val_loss}"
                )
            if acc_loss < best_loss:
                curr_patience = 0
                best_loss = acc_loss
                mlflow.log_metric("Best train loss", best_loss, step=epoch)
            if (epoch + 1) % epoch_before_increase == 0 and last_train_index < len(
                self.blocks
            ):
                last_train_index += 1
            if best_val_loss > acc_val_loss:
                best_val_loss = acc_val_loss
                mlflow.log_metric("Best val loss", best_val_loss, step=epoch)
                for i, (nn, block) in enumerate(self.blocks):
                    best_states[i] = nn.to_dict()
            curr_patience += 1
            if curr_patience == patience:
                break

    def compose_train(
        self,
        epochs,
        verbose,
        callbacks,
        val_function,
        patience=300,
        log_interval=100,
        eval_interval=1,
        batch_size=10,
    ):
        self.sequence_train(
            epochs // 2,
            verbose,
            callbacks,
            val_function,
            patience,
            log_interval,
            eval_interval,
            batch_size,
        )
        i = 0
        while i < len(self.blocks):
            nn = self.blocks[i][0]
            nn.trainable = True
            nn.custom_compile(
                optimizer=self.optimizer_,
                rate=self.rate_,
                loss_func=self.loss_func_,
                metric_funcs=self.metric_funcs_,
                run_eagerly=self.run_eagerly,
            )
            i += 1
        self.all_train(
            epochs // 2,
            verbose,
            callbacks,
            val_function,
            patience,
            log_interval,
            eval_interval,
            batch_size,
        )

    def full_train(
        self,
        epochs,
        verbose,
        val_input,
        val_function,
        patience=300,
        callbacks=None,
        log_interval=100,
        eval_interval=1,
        batch_size=10,
        png_salt="",
        **kwargs,
    ):
        curr_patience = 0
        best_loss = 1e6

        for epoch in range(epochs):
            batches = self.split_to_batches(self.data, batch_size)
            loss = 0
            for batch in batches:
                metrics = self.train_step(batch)
                loss += metrics["loss"]
            if epoch % eval_interval == 0:
                (
                    y_pred,
                    y_true,
                    val_mse_loss,
                    val_mae_loss,
                    val_l1_loss,
                ) = self.get_val_score(val_input, val_function)
                mlflow.log_metric("Validation MSE loss", val_mse_loss, step=epoch)
                mlflow.log_metric("Validation MAE loss", val_mae_loss, step=epoch)
                mlflow.log_metric("Validation Relative L1Loss", val_l1_loss, step=epoch)
                mlflow.log_metric("Loss", loss, step=epoch)
                mlflow.log_metric(
                    "Learning rate", self.optimizer.learning_rate.numpy(), step=epoch
                )
            if epoch % log_interval == 0:
                (
                    y_pred,
                    y_true,
                    val_mse_loss,
                    val_mae_loss,
                    val_l1_loss,
                ) = self.get_val_score(val_input, val_function)

                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
                plot_each_submodel(val_input, y_true, self, axes[0])
                plot_model(val_input, y_true, self, axes[1])
                plt.savefig(
                    f"FBPINN_{png_salt}_{epoch}.png", dpi=300, bbox_inches="tight"
                )

                for nn, block in self.blocks:
                    self.log_weights(epoch, nn)

                print(
                    f"Epoch {epoch}, loss {loss}, val mse loss {val_mse_loss}, mae loss {val_mae_loss}, rel l1loss {val_l1_loss}. {datetime.datetime.now()}"
                )
            if loss < best_loss:
                best_loss = loss
                curr_patience = 0
            curr_patience += 1
            if curr_patience > patience:
                break
        y_pred, y_true, val_mse_loss, val_mae_loss, val_l1_loss = self.get_val_score(
            val_input, val_function
        )

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        plot_each_submodel(val_input, y_true, self, axes[0])
        plot_model(val_input, y_true, self, axes[1])
        plt.savefig(f"FBPINN_{png_salt}_{epoch}.png", dpi=300, bbox_inches="tight")

        print(
            f"Epoch {epoch}, loss {loss}, val mse loss {val_mse_loss}, mae loss {val_mae_loss}, rel l1loss {val_l1_loss}. {datetime.datetime.now()}"
        )

    @tf.function
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
        with tf.GradientTape(persistent=True) as tape:
            x = tf.identity(data)
            loss: tf.Tensor = tf.zeros(shape=1)
            for loss_func, models in self.model_per_loss:
                for model in models:
                    if model is not None:
                        loss += loss_func(model, tape, x)
                    else:
                        loss += loss_func(self, tape, x)

        # Compute gradients
        trainable_vars = self.trainable_variables
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
