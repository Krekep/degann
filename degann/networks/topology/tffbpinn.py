import datetime
import math
import time
from typing import List, Optional, Dict, Callable

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import mlflow

from degann.geometry import RectangleDomain, Decomposition, Block
from degann.networks import losses, metrics, optimizers
from degann.networks.layers.tf_dense import TensorflowDense
from degann.networks.topology.pinn import PhysicsInformedNet
from examples.fbpinn_tests.plot_functions import plot_each_submodel, plot_model


class LayerScheduler:
    def __init__(
        self,
        n: int,
        left_bound_step: int,
        right_bound_step: int,
        left_bound_schedule: int,
        right_bound_schedule: int,
        start_left_bound: int,
        start_right_bound: int,
    ):
        self.n = n
        self.left_bound_step = left_bound_step
        self.right_bound_step = right_bound_step
        self.left_bound_schedule = left_bound_schedule
        self.right_bound_schedule = right_bound_schedule
        self.current_step = 0
        self.current_left_bound = start_left_bound
        self.current_right_bound = start_right_bound
        self.current_indices = list(range(start_left_bound, start_right_bound))
        self.is_changes = True

    def on_epoch_start(self) -> list[int]:
        res = self.current_indices

        if self.is_changes:
            range_changed = False
            if self.current_step % self.right_bound_schedule == 0:
                if self.current_right_bound == self.n:
                    self.current_left_bound = 0
                    self.is_changes = False
                self.current_right_bound = min(
                    self.n, self.current_right_bound + self.right_bound_step
                )
                range_changed = True

            if self.is_changes and self.current_step % self.left_bound_schedule == 0:
                self.current_left_bound += self.left_bound_step
                range_changed = True
            if range_changed:
                self.current_indices = list(
                    range(self.current_left_bound, self.current_right_bound)
                )
        return res

    def step(self):
        self.current_step += 1


class LossScheduler:
    """
    Simple loss scheduler. First k epochs it returns only boundary and initial loss indices, after returns all loss indices
    """

    def __init__(
        self,
        k: int,
        boundary_indices: list[int],
    ):
        self.k = k
        self.current_step = 0
        self.boundary_indices = boundary_indices
        self.all_indices = boundary_indices + [len(boundary_indices)]

    def on_epoch_start(self) -> list[int]:
        if self.current_step <= self.k:
            return self.boundary_indices
        return self.all_indices

    def step(self):
        self.current_step += 1


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
        block_size=[0.2],
        overlap=[0.05],
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
        block_size: list[float] = block_size
        overlap: list[float] = overlap
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
                    fl = True
                    for lc, var_b, rc in zip(left_down_corner, point, right_up_corner):
                        if var_b is not None:
                            for b in var_b:
                                if lc > b or b > rc:
                                    fl = False
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

    # @tf.function
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
            result = block.forward(nn, x)
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
        if self.summary_writer is not None:
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

    # @tf.function
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

        for i, (nn, block) in enumerate(self.blocks):
            data = block.get_data()
            result = block.forward(nn, data)
            t = ode.solution(data)
            nn_mae_loss = tf.reduce_mean(tf.abs(result, t))
            nn_mse_loss = tf.reduce_mean(tf.square(result, t))
            nn_rel_loss = tf.reduce_mean(tf.abs(t - result)) / tf.reduce_mean(tf.abs(t))
            mlflow.log_metric(f"Validation MSE loss model{i}", nn_mae_loss, step=epoch)
            mlflow.log_metric(f"Validation MAE loss model{i}", nn_mse_loss, step=epoch)
            mlflow.log_metric(
                f"Validation Relative L1Loss model{i}", nn_rel_loss, step=epoch
            )

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
        # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        # plot_each_submodel(x, val_input, y_true, self, axes[0])
        # plot_model(x, val_input, y_true, self, axes[1])
        plt.savefig(
            f"FBPINN_{png_salt}_{epoch}_t{t[0]}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        # for nn, block in self.blocks:
        #     self.log_weights(epoch, nn)

        # with tf.GradientTape() as tape:
        #     tape.watch(x)
        #     u = self(x)
        #     u_x = tape.gradient(u, x)
        #     original_dx = ode.first_der("x", val_input[:, 0], val_input[:, 1])
        #     original_dt = ode.first_der("t", val_input[:, 0], val_input[:, 1])
        #     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        #     axes[0].plot(
        #         val_input[:, 1], original_dx, label="Truth x derivative", color="red"
        #     )
        #     axes[0].plot(
        #         val_input[:, 0], original_dt, label="Truth t derivative", color="black"
        #     )
        #     axes[0].plot(
        #         val_input[:, 1], u_x[:, 1], label="Model x derivative", color="green"
        #     )
        #     axes[0].plot(
        #         val_input[:, 0], u_x[:, 0], label="Model t derivative", color="purple"
        #     )
        #     axes[1].plot(
        #         val_input[:, 1],
        #         original_dx - u_x[:, 1],
        #         label="Difference x",
        #         color="blue",
        #     )
        #     axes[1].plot(
        #         val_input[:, 0],
        #         original_dt - u_x[:, 0],
        #         label="Difference t",
        #         color="orange",
        #     )
        #     axes[0].grid()
        #     axes[1].grid()
        #     axes[1].legend()
        #     axes[0].legend()
        #     plt.savefig(
        #         f"FBPINN_{png_salt}_d_{epoch}_t{t[0]}.png", dpi=300, bbox_inches="tight"
        #     )
        #     plt.close(fig)

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
        layer_scheduler=None,
        loss_scheduler=None,
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
                val_input,
                png_salt,
            )
        elif mode == "all":
            self.all_train(
                epochs,
                verbose,
                callbacks,
                ode,
                patience,
                log_interval,
                eval_interval,
                batch_size,
                val_input,
                png_salt,
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
                layer_scheduler=layer_scheduler,
                loss_scheduler=loss_scheduler,
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
        layer_scheduler: LayerScheduler = None,
        loss_scheduler: LayerScheduler = None,
    ):
        curr_patience = 0
        best_loss = 1e6

        for epoch in range(epochs):
            start_time = time.perf_counter()
            layer_scheduler.step()
            loss_scheduler.step()
            layer_indices = layer_scheduler.on_epoch_start()
            loss_indices = loss_scheduler.on_epoch_start()
            # layer_end = min(layer_end, len(self.blocks))
            data = []
            for i in layer_indices:
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
                    _, step_loss, losses = self.train_step(
                        data,
                        list(range(0, len(self.blocks))),
                        list(range(0, len(self.model_per_loss))),
                    )
                else:
                    _, step_loss, losses = self.train_step(
                        data, layer_indices, loss_indices
                    )
                loss += step_loss

            end_time = time.perf_counter()
            if epoch % eval_interval == 0:
                self.log_validation_metrics(
                    log_t, val_input, ode, epoch, loss, end_time - start_time
                )
            if epoch % log_interval == 0:
                self.log_graphics(log_t, val_input, ode, epoch, loss, png_salt)
                print(f"Epoch {epoch}, layer indices {layer_indices}")
                print(f"Epoch {epoch}, loss indices {loss_indices}")
                print(f"Epoch {epoch}, data length {len(blocks_data)}")
                tf.print(losses)
                self.save_weights(f"fbpinn{png_salt}_{epoch}.weights.h5")
            if loss < best_loss:
                best_loss = loss
                curr_patience = 0
            curr_patience += 1
            if curr_patience > patience:
                print(
                    f"Too long, Current patience is {curr_patience}, all patience is {patience}, best loss {best_loss}"
                )
                break
        self.log_graphics(log_t, val_input, ode, epoch, loss, png_salt)

    @tf.function
    def train_step(self, data, layer_indices, loss_indices):
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
                # temp = nn.trainable_variables
                variables.extend(nn.trainable_variables)
            # variables.extend(self.trainable_variables)
            return variables

        # adf = self.trainable_variables
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        all_active_models_idx = set()
        losses = []
        with tf.GradientTape() as tape:
            x = tf.identity(data)
            loss: tf.Tensor = tf.zeros(shape=1)
            for i, (loss_func, models, loss_weigth) in enumerate(self.model_per_loss):
                if i in loss_indices:
                    active_models = [
                        self.blocks[model] for model in models if model in layer_indices
                    ]
                    # for model in models:
                    #     if layer_start <= model <= layer_end:
                    #         active_models.append(self.blocks[model])
                    #         all_active_models_idx.add(model)
                    all_active_models_idx.update(active_models)
                    if len(active_models) > 0:
                        temp_loss = (
                            loss_func(self, tape, x, active_models=active_models)
                            * loss_weigth
                        )
                        losses.append(temp_loss)
                        loss += temp_loss
        # all_active_models = []
        # for idx in all_active_models_idx:
        #     all_active_models.append(self.blocks[idx])

        # Compute gradients
        trainable_vars = get_active_variables(all_active_models_idx)
        gradients = tape.gradient(loss, trainable_vars)
        # # Update weights
        # self.optimizer.apply(gradients, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        del tape
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            # elif y is not None:
            #     metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}, loss, losses

    def sequence_train(
        self,
        epochs,
        verbose,
        callbacks,
        ode,
        patience=300,
        log_interval=100,
        eval_interval=1,
        batch_size=50,
        val_input=None,
        png_salt="",
    ):
        i = 0
        while i < len(self.blocks):
            nn, block = self.blocks[i]
            inputs = block.get_data()

            best_loss = 1e6
            best_val_loss = 1e6
            curr_patience = 0
            for epoch in range(epochs):
                start_time = time.perf_counter()
                if i == 0:
                    loss, losses = nn.custom_train_step(inputs, block, None, None)
                else:
                    loss, losses = nn.custom_train_step(
                        inputs,
                        block=block,
                        prev_model=self.blocks[i - 1][0],
                        prev_block=self.blocks[i - 1][1],
                    )

                end_time = time.perf_counter()
                if epoch % eval_interval == 0:
                    if i == 0:
                        val_mse_loss, val_mae_loss, val_l1_loss = nn.get_val_score(
                            inputs, block, None, None, ode.solution
                        )
                    else:
                        val_mse_loss, val_mae_loss, val_l1_loss = nn.get_val_score(
                            inputs,
                            block,
                            self.blocks[i - 1][0],
                            self.blocks[i - 1][1],
                            ode.solution,
                        )
                    mlflow.log_metric(
                        f"Validation MSE loss model {i}", val_mse_loss, step=epoch
                    )
                    mlflow.log_metric(
                        f"Validation MAE loss model {i}", val_mae_loss, step=epoch
                    )
                    mlflow.log_metric(
                        f"Validation Relative L1Loss model {i}", val_l1_loss, step=epoch
                    )
                    mlflow.log_metric(f"Train loss model {i}", loss, step=epoch)
                    mlflow.log_metric(
                        f"Epoch time model {i}", end_time - start_time, step=epoch
                    )
                if epoch % log_interval == 0:
                    print(
                        f"Model {i}, epoch {epoch}, last loss {loss}, val l1 loss {val_l1_loss}, best val {best_val_loss}"
                    )
                    tf.print(losses)
                    log_t = [None]
                    self.log_graphics(log_t, val_input, ode, epoch, loss, png_salt)
                if loss < best_loss:
                    best_loss = loss
                    curr_patience = 0
                curr_patience += 1
                if curr_patience == patience:
                    break

            if i == 0:
                val_mse_loss, val_mae_loss, val_l1_loss = nn.get_val_score(
                    inputs, block, None, None, ode.solution
                )
            else:
                val_mse_loss, val_mae_loss, val_l1_loss = nn.get_val_score(
                    inputs,
                    block,
                    self.blocks[i - 1][0],
                    self.blocks[i - 1][1],
                    ode.solution,
                )
            mlflow.log_metric(
                f"Validation MSE loss model {i}", val_mse_loss, step=epoch
            )
            mlflow.log_metric(
                f"Validation MAE loss model {i}", val_mae_loss, step=epoch
            )
            mlflow.log_metric(
                f"Validation Relative L1Loss model {i}", val_l1_loss, step=epoch
            )
            mlflow.log_metric(f"Train loss model {i}", loss, step=epoch)

            print(
                f"Model {i}, epoch {epoch}, last loss {loss}, val l1 loss {val_l1_loss}, best val {best_val_loss}"
            )
            print()

            i += 1

    def all_train(
        self,
        epochs,
        verbose,
        callbacks,
        ode,
        patience=300,
        log_interval=100,
        eval_interval=1,
        batch_size=10,
        val_input=None,
        png_salt="",
    ):
        curr_patience = 0
        best_loss = 1e6
        best_val_loss = 1e6
        inputs_per_model = []
        for i, (nn, block) in enumerate(self.blocks):
            inputs = block.get_data()
            inputs_per_model.append(inputs)

        for epoch in range(epochs):
            acc_loss = 0
            acc_val_mse = 0
            acc_val_mae = 0
            acc_val_l1 = 0

            for i, (nn, block) in enumerate(self.blocks):
                start_time = time.perf_counter()
                inputs = inputs_per_model[i]
                if i == 0:
                    loss, losses = nn.custom_train_step(
                        inputs,
                        block,
                        None,
                        None,
                    )
                else:
                    loss, losses = nn.custom_train_step(
                        inputs,
                        block=block,
                        prev_model=self.blocks[i - 1][0],
                        prev_block=self.blocks[i - 1][1],
                    )
                end_time = time.perf_counter()
                if epoch % eval_interval == 0:
                    if i == 0:
                        val_mse_loss, val_mae_loss, val_l1_loss = nn.get_val_score(
                            inputs, block, None, None, ode.solution
                        )
                    else:
                        val_mse_loss, val_mae_loss, val_l1_loss = nn.get_val_score(
                            inputs,
                            block,
                            self.blocks[i - 1][0],
                            self.blocks[i - 1][1],
                            ode.solution,
                        )
                    mlflow.log_metric(
                        f"Validation MSE loss model {i}", val_mse_loss, step=epoch
                    )
                    mlflow.log_metric(
                        f"Validation MAE loss model {i}", val_mae_loss, step=epoch
                    )
                    mlflow.log_metric(
                        f"Validation Relative L1Loss model {i}", val_l1_loss, step=epoch
                    )
                    mlflow.log_metric(f"Train loss model {i}", loss, step=epoch)
                    mlflow.log_metric(
                        f"Epoch time model {i}", end_time - start_time, step=epoch
                    )
                    acc_val_mse += val_mse_loss
                    acc_val_mae += val_mae_loss
                    acc_val_l1 += val_l1_loss
                acc_loss += loss
            acc_val_mse /= len(self.blocks)
            acc_val_mae /= len(self.blocks)
            acc_val_l1 /= len(self.blocks)
            acc_loss /= len(self.blocks)
            mlflow.log_metric("Validation MSE loss", acc_val_mse, step=epoch)
            mlflow.log_metric("Validation MAE loss", acc_val_mae, step=epoch)
            mlflow.log_metric("Validation Relative L1Loss loss", acc_val_l1, step=epoch)
            mlflow.log_metric("Train loss", acc_loss, step=epoch)

            if epoch % log_interval == 0:
                print(
                    f"Model {i}, epoch {epoch}, last loss {loss}, val l1 loss {acc_val_l1}"
                )
                tf.print(losses)
                log_t = [None]
                self.log_graphics(log_t, val_input, ode, epoch, loss, png_salt)
            if loss < best_loss:
                best_loss = loss
                curr_patience = 0
            curr_patience += 1
            if curr_patience == patience:
                break

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
        res = dict()
        for i, (nn, block) in enumerate(self.blocks):
            res[f"block_{i}"] = {
                block.left_down_corner,
                block.right_up_corner,
                # TODO: Window function
            }
            # TODO: export phus losses or create method for run time adding them
            # TODO: export models
            # "discriminator": self.discriminator.to_dict(
            #     **kwargs.get("discriminator", dict())
            # ),
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
