import os
from typing import List, Optional, Dict, Callable

import tensorflow as tf
from tensorflow import keras

from degann.geometry import RectangleDomain, Decomposition, Block
from degann.networks.config_format import LAYER_DICT_NAMES
from degann.networks import layer_creator, losses, metrics, cpp_utils
from degann.networks import optimizers
from degann.networks.layers.tf_dense import TensorflowDense
from degann.networks.topology import TensorflowDenseNet
from degann.networks.topology.pinn import PhysicsInformedNet


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
        fbpinn_predict: tf.Tensor = tf.zeros(shape=self.output_size)
        for nn, block in self.blocks:
            x_norm = block.normalization(inputs)
            predicted = nn(x_norm)
            predicted_unnorm: tf.Tensor = block.unnormalization(predicted)
            windowed = block.window_function(inputs)
            result = windowed * predicted_unnorm
            fbpinn_predict = fbpinn_predict + result

        return fbpinn_predict

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

    def split_to_batches(self, input, batch_size):
        input = tf.random.shuffle(input)
        n = input.shape[0]
        k = batch_size
        num_batches = n // k
        remainder = n % k

        # Полные батчи
        if num_batches > 0:
            full_batches = tf.split(input[: num_batches * k], num_batches)
            if remainder != 0:
                full_batches.append(input[num_batches * k :])

            return full_batches
        return [input]

    def get_val_score(self, inputs, i, val_function):
        nn, block = self.blocks[i]
        x_norm = block.normalization(inputs)
        predicted = nn(x_norm)
        predicted_unnorm = block.unnormalization(predicted)
        if i == 0:
            y_pred = block.window_function(inputs) * predicted_unnorm
        else:
            left_block = self.blocks[i - 1][1]
            left_model = self.blocks[i - 1][0]
            x_norm_left = left_block.normalization(inputs)
            predicted_left = left_model(x_norm_left)
            predicted_unnorm_left = left_block.unnormalization(predicted_left)
            y_pred_left = left_block.window_function(inputs) * predicted_unnorm_left
            y_pred = block.window_function(inputs) * predicted_unnorm + y_pred_left
        y_true = val_function(inputs)
        val_loss = nn.loss(y_true, y_pred)
        return val_loss

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
            inputs = tf.reshape(
                tf.convert_to_tensor(block.get_data(), dtype=tf.float32), shape=(-1, 1)
            )

            best_loss = 1e6
            best_val_loss = 1e6
            best_weights = nn.get_weights()
            curr_patience = 0
            batches = self.split_to_batches(inputs, batch_size)
            for epoch in range(epochs):
                # loss: tf.Tensor = tf.Variable(0, dtype=tf.float32)
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
                # loss /= len(batches)

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

                    with self.summary_writer.as_default():
                        tf.summary.scalar(f"{nn.name}/loss", loss, step=epoch)
                        tf.summary.scalar(f"{nn.name}/val_loss", val_loss, step=epoch)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = nn.get_weights()

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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = nn.get_weights()

            print(
                f"Model {i}, epoch {epoch}, last loss {loss}, val_loss {val_loss}, best val {best_val_loss}"
            )
            print()

            nn.set_weights(best_weights)
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
    ):
        curr_patience = 0
        best_loss = 1e6
        best_val_loss = 1e6
        best_states = [nn.to_dict() for i, (nn, block) in enumerate(self.blocks)]
        for epoch in range(epochs):
            acc_loss = 0
            acc_val_loss = 0
            for i, (nn, block) in enumerate(self.blocks):
                inputs = tf.convert_to_tensor(block.data, dtype=tf.float32)
                if i == 0:
                    loss, _ = nn.custom_train_step(
                        inputs,
                        nn,
                        block,
                        None,
                        None,
                        log_interval=log_interval,
                        epoch=epoch,
                        summary_writer=self.summary_writer,
                    )
                else:
                    loss, _ = nn.custom_train_step(
                        inputs,
                        nn,
                        block=block,
                        prev_model=self.blocks[i - 1][0],
                        prev_block=self.blocks[i - 1][1],
                        log_interval=log_interval,
                        epoch=epoch,
                        summary_writer=self.summary_writer,
                    )

                if epoch % eval_interval == 0:
                    val_loss = self.get_val_score(inputs, i, val_function)
                    acc_val_loss += val_loss
                    with self.summary_writer.as_default():
                        tf.summary.scalar("loss", loss, step=epoch)
                        tf.summary.scalar("val_loss", val_loss, step=epoch)
                acc_loss += loss
            acc_val_loss /= len(self.blocks)
            acc_loss /= len(self.blocks)

            if epoch % patience == 0:
                print(
                    f"Epoch {epoch}, last loss {loss}, val_loss {val_loss}, best val_loss {best_val_loss}"
                )
            if epoch % log_interval:
                self.log_weights(epoch)
            if acc_loss < best_loss:
                curr_patience = 0
                best_loss = acc_loss

            if best_val_loss > acc_val_loss:
                best_val_loss = acc_val_loss
                for i, (nn, block) in enumerate(self.blocks):
                    best_states[i] = nn.to_dict()
            curr_patience += 1
            if curr_patience == patience:
                break
        i = 0
        while i < len(self.blocks):
            block = self.blocks[i][1]
            nn = self.load_model(
                model_type=PhysicsInformedNet,
                model_state=best_states[i],
                trainable=False,
                block=block,
            )
            self.blocks[i] = (nn, block)
            i += 1

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
        x = data
        with tf.GradientTape(persistent=True) as tape:
            loss: tf.Tensor = tf.zeros(shape=1)
            # data = []
            # for i, (nn, block) in enumerate(self.blocks):
            # inputs = tf.convert_to_tensor(block.data, dtype=tf.float32)
            # tape.watch(inputs)
            # for loss_func in block.losses:
            #     if i == 0:
            #         loss += loss_func(nn, tape, inputs, training=True, block=block, prev_model=None, prev_block=None)
            #     else:
            #         loss += loss_func(nn, tape, inputs, training=True, block=block, prev_model=None, prev_block=None)
            # data.append(inputs)
            # input = tf.concat(data, axis=0)
            tape.watch(x)
            for loss_func in self.phys_losses:
                loss += loss_func(
                    self,
                    tape,
                    x,
                    training=True,
                    block=None,
                    prev_model=None,
                    prev_block=None,
                    make_shit=False,
                )
            # y_pred = self(x)  # for metrics
            # Compute the loss value
            # (the loss function is configured in `compile()`)

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
