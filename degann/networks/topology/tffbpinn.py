import datetime
import math
import time
from typing import List, Optional, Callable

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import mlflow

from degann.geometry import RectangleDomain, Decomposition, Block
from degann.networks import losses, metrics, optimizers
from degann.networks.layers.tf_dense import TensorflowDense
from degann.networks.topology.layer_scheduler import (
    SequenceLayerScheduler,
    BaseLayerScheduler,
)
from degann.networks.topology.loss_scheduler import LossScheduler
from degann.networks.topology.pinn import PhysicsInformedNet
from examples.fbpinn_tests.plot_functions import plot_each_submodel, plot_model


class TensorflowFBPINN(tf.keras.Model):
    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 10,
        activation_func: str | list[str] = "linear",
        weight=keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
        biases=keras.initializers.Zeros(),
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
                weight=weight,
                biases=biases,
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
        self.model_per_loss: list[tuple[Callable, list[int]]] = []
        for i, (loss, point) in enumerate(boundary_loss + [(physic_loss, None)]):
            if point is None:
                self.model_per_loss.append((loss, list(range(0, len(self.blocks)))))
            else:
                models: list[int] = []
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
                self.model_per_loss.append((loss, models))

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
            nn.trainable = False
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

        nn.trainable = trainable
        nn.custom_compile(
            optimizer=self.optimizer_,
            rate=self.rate_,
            loss_func=self.loss_func_,
            metric_funcs=self.metric_funcs_,
            run_eagerly=self.run_eagerly,
        )
        return nn

    def freeze_model(self, nn):
        nn.trainable = False

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

    def log_validation_metrics(self, val_mse, val_mae, val_l1, epoch, loss, epoch_time):
        mlflow.log_metric("Validation MSE loss", val_mse, step=epoch)
        mlflow.log_metric("Validation MAE loss", val_mae, step=epoch)
        mlflow.log_metric("Validation Relative L1Loss", val_l1, step=epoch)
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

        # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        # plot_each_submodel(x, val_input[:, 1], y_true, self, axes[0])
        # plot_model(x, val_input[:, 1], y_true, self, axes[1])
        # plt.savefig(
        #     f"FBPINN_{png_salt}_{epoch}_t{t[0]}.png", dpi=300, bbox_inches="tight"
        # )
        # plt.close(fig)

        print(
            f"Epoch {epoch}, loss {loss}, val mse loss {val_mse_loss}, mae loss {val_mae_loss}, rel l1loss {val_l1_loss}. {datetime.datetime.now()}"
        )

    def train(
        self,
        epochs: int,
        verbose: bool,
        callbacks: list,
        ode,
        patience: int = 300,
        log_interval: int = 100,
        eval_interval: int = 1,
        batch_size: int = 10,
        mode: str = "sequence",  # TODO: Rewrite to Enum
        val_input=None,
        png_salt: str = "",
        layer_scheduler: BaseLayerScheduler = None,
        loss_scheduler: LossScheduler = None,
        need_export_weights: bool = False,
    ):
        self.build_data()
        self.build_time_vectors()
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
                need_export_weights=need_export_weights,
            )
        else:
            raise ValueError("Unsupported train mode")

    def _get_trainable_variables_for_active_models(
        self, active_indices, frozen_indices
    ):
        """Возвращает список обучаемых переменных для активных моделей"""
        trainable_vars = []

        # Собираем переменные только из активных и незамороженных моделей
        for i, (nn, block) in enumerate(self.blocks):
            if i in active_indices and i not in frozen_indices:
                trainable_vars.extend(nn.trainable_variables)

        return trainable_vars

    def layer_train(
        self,
        epochs: int,
        verbose: bool,
        val_input,
        ode,
        patience: int = 300,
        callbacks=None,
        log_interval: int = 100,
        eval_interval: int = 1,
        batch_size: int = 10,
        png_salt: str = "",
        layer_scheduler: BaseLayerScheduler = None,
        loss_scheduler: LossScheduler = None,
        need_export_weights: bool = False,
    ):
        curr_patience = 0
        best_loss = 1e6

        train_step_func = None

        max_w_update = 1e6
        old_frozen_indices = set()
        prev_layer_indices = set()
        for epoch in range(epochs):
            start_time = time.perf_counter()
            layer_scheduler.step()
            loss_scheduler.step()
            layer_indices = layer_scheduler.on_epoch_start()
            frozen_indices = layer_scheduler.get_frozen_indices()

            is_changed = False
            for i in layer_indices:
                if i not in prev_layer_indices:
                    prev_layer_indices.add(i)
                    self.blocks[i][0].trainable = True
                    is_changed = True

            for i in frozen_indices:
                if i not in old_frozen_indices:
                    old_frozen_indices.add(i)
                    nn = self.blocks[i][0]
                    self.freeze_model(nn)
                    is_changed = True

            if is_changed:
                self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.rate_)
                trainable_vars = self._get_trainable_variables_for_active_models(
                    layer_indices, frozen_indices
                )
                self.optimizer.build(trainable_vars)
                self.compile(
                    optimizer=self.optimizer,
                    loss=self.loss_func_,
                    metrics=self.metric_funcs_,
                    run_eagerly=False,
                )

                train_step_func = tf.function(self.train_step)
                # Логирование изменений
                print(f"Epoch {epoch}: Active models changed")
                print(f"  Active indices: {layer_indices}")
                print(f"  Frozen indices: {frozen_indices}")
                # Проверка переменных оптимизатора
                print(
                    f"Epoch {epoch}: Trainable vars={len(trainable_vars)}, "
                    f"Optimizer vars={len(self.optimizer.variables)}, "
                    f"Expected optimizer vars={2 * len(trainable_vars) + 2}"
                )

            if train_step_func is None:
                train_step_func = tf.function(self.train_step)

            loss_indices, loss_weights = loss_scheduler.on_epoch_start(
                curr_max_w_update=max_w_update
            )
            data = []
            for i in layer_indices:
                if i not in old_frozen_indices:
                    data.append(self.blocks[i][1].get_data())
            data = tf.concat(data, axis=0)

            blocks_data = data
            for t in self.time_data:
                log_t = [None]
                loss = 0
                if epoch == 0:
                    _, step_loss, losses, max_w_update = train_step_func(
                        data,
                        layer_indices,
                        frozen_indices,
                        list(range(0, len(self.model_per_loss))),
                        loss_weights,
                    )
                else:
                    _, step_loss, losses, max_w_update = train_step_func(
                        data, layer_indices, frozen_indices, loss_indices, loss_weights
                    )
                loss += step_loss

            end_time = time.perf_counter()
            if epoch % eval_interval == 0:
                (
                    y_pred,
                    y_true,
                    val_mse_loss,
                    val_mae_loss,
                    val_l1_loss,
                ) = self.get_val_score(val_input, ode.solution)
                mlflow.log_metric(
                    f"Max layer weights gradient", max_w_update, step=epoch
                )
                self.log_validation_metrics(
                    val_mse_loss,
                    val_mae_loss,
                    val_l1_loss,
                    epoch,
                    loss,
                    end_time - start_time,
                )
            if epoch % log_interval == 0:
                self.log_graphics(log_t, val_input, ode, epoch, loss, png_salt)
                print(f"Epoch {epoch}, layer indices {layer_indices}")
                print(f"Epoch {epoch}, loss indices {loss_indices}")
                print(f"Epoch {epoch}, data length {len(blocks_data)}")
                tf.print(losses)
                if need_export_weights:
                    self.save_weights(f"fbpinn{png_salt}_{epoch}.weights.h5")
                active_vars = self._get_trainable_variables_for_active_models(
                    layer_indices, frozen_indices
                )
                print(
                    f"Epoch {epoch}: Trainable vars={len(self.trainable_variables)}, "
                    f"Optimizer vars={len(self.optimizer.variables)}, "
                    f"Expected optimizer vars={2 * len(self.trainable_variables) + 2}"
                )
                mlflow.log_metric("Trainable variables", len(active_vars), step=epoch)
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

    # @tf.function
    def train_step(
        self, data, layer_indices, frozen_indices, loss_indices, loss_weights
    ):
        """
        Custom train step from tensorflow tutorial
        """
        all_active_models_idx = set()
        losses = []
        with tf.GradientTape() as tape:
            x = tf.identity(data)
            loss: tf.Tensor = tf.zeros(shape=1)
            for i, (loss_func, models) in enumerate(self.model_per_loss):
                if i in loss_indices:
                    active_models = [
                        self.blocks[model] for model in models if model in layer_indices
                    ]
                    all_active_models_idx.update(active_models)
                    if len(active_models) > 0:
                        temp_loss = (
                            loss_func(self, tape, x, active_models=active_models)
                            * loss_weights[i]
                        )
                        losses.append(temp_loss)
                        loss += temp_loss

        # Compute gradients
        trainable_vars = self._get_trainable_variables_for_active_models(
            layer_indices, frozen_indices
        )
        gradients = tape.gradient(loss, trainable_vars)

        max_grad = tf.reduce_max(
            [tf.reduce_max(tf.abs(g)) for g in gradients if g is not None]
        )

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        del tape
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
        return (
            {m.name: m.result() for m in self.metrics},
            loss,
            losses,
            max_grad,
        )

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
        for i in range(len(self.blocks)):
            nn = self.blocks[i][0]
            nn.trainable = True

        i = 0
        while i < len(self.blocks):
            nn, block = self.blocks[i]

            best_loss = 1e6
            best_val_loss = 1e6
            curr_patience = 0
            for epoch in range(epochs):
                inputs = block.get_data()
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
                if epoch % log_interval == 0:
                    print(f"Model {i}, epoch {epoch}, last loss {loss}")
                    tf.print(losses)
                    log_t = [None]
                    self.log_graphics(log_t, val_input, ode, epoch, loss, png_salt)
                if loss < best_loss:
                    best_loss = loss
                    curr_patience = 0
                curr_patience += 1
                if curr_patience == patience:
                    break
            print(f"Model {i}, epoch {epoch}, last loss {loss}")
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
        for i in range(len(self.blocks)):
            nn = self.blocks[i][0]
            nn.trainable = True

        curr_patience = 0
        best_loss = 1e6
        best_val_loss = 1e6

        for epoch in range(epochs):
            inputs_per_model = []
            for i, (nn, block) in enumerate(self.blocks):
                inputs = block.get_data()
                inputs_per_model.append(inputs)
            acc_loss = 0
            acc_val_mse = 0
            acc_val_mae = 0
            acc_val_l1 = 0

            start_time = time.perf_counter()
            for i, (nn, block) in enumerate(self.blocks):
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
                acc_loss += loss
            end_time = time.perf_counter()
            acc_loss /= len(self.blocks)
            if epoch % eval_interval == 0:
                (
                    y_pred,
                    y_true,
                    val_mse_loss,
                    val_mae_loss,
                    val_l1_loss,
                ) = self.get_val_score(val_input, ode.solution)
                self.log_validation_metrics(
                    val_mse_loss,
                    val_mae_loss,
                    val_l1_loss,
                    epoch,
                    acc_loss,
                    end_time - start_time,
                )

            if epoch % log_interval == 0:
                print(f"Epoch {epoch}, last loss {acc_loss}, val l1 loss {acc_val_l1}")
                tf.print(losses)
                log_t = [None]
                # self.log_graphics(log_t, val_input, ode, epoch, acc_loss, png_salt)
            if acc_loss < best_loss:
                best_loss = acc_loss
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
