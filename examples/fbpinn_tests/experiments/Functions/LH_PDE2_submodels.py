import math
import tensorflow as tf

from examples.fbpinn_tests.phys_losses import PhysLoss


class LH_PDE2(PhysLoss):
    def __init__(self, description: str = "", a: float = 0.1, **kwargs):
        description = (
            "du/dt = a * d^2u/dx^2, u(t, 0) = 0, u(t, 1) = 0, u(0, x) = sin(pi * x)"
        )
        self.equation_class = "Parabolic PDE"
        self.a = tf.constant(a, dtype=tf.float32)
        self.time_input = True
        super().__init__(description)

        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (self.boundary_loss_1, [(0.0, 0.0), None]),
            (self.boundary_loss_2, [None, (0.0, 0.0)]),
            (self.boundary_loss_3, [None, (1.0, 1.0)]),
        ]

    @tf.function
    def phys_loss(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        block,
        prev_model,
        prev_block,
        **kwargs
    ):
        """du/dt = a * d^2u/dx^2"""
        t, x = x_in[:, 0:1], x_in[:, 1:2]
        # tape.watch(t)
        # tape.watch(x)

        with tf.GradientTape() as outer_tape:
            outer_tape.watch(t)
            outer_tape.watch(x)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(t)
                inner_tape.watch(x)
                x_ = tf.concat([t, x], axis=1)
                x_norm = block.normalization(x_)
                predicted = model(x_norm)
                predicted_unnorm: tf.Tensor = block.unnormalization(predicted)

                windowed = block.window_function(x_)
                if prev_model is not None:
                    x_left_norm = prev_block.normalization(x_)
                    predicted_left = prev_model(x_left_norm)
                    predicted_unnorm_left: tf.Tensor = prev_block.unnormalization(
                        predicted_left
                    )
                    windowed_left = prev_block.window_function(x_)

                    u = (
                        windowed * predicted_unnorm
                        + windowed_left * predicted_unnorm_left
                    )
                else:
                    u = windowed * predicted_unnorm
                # assert_equal(u.shape.rank, 1, "Model() have len(shape) != 1")
                # if u.shape.rank > 1 and u.shape[-1] == 1:
                u = tf.squeeze(u, axis=-1)
                du_dt, du_dx = inner_tape.gradient(u, [t, x])  # Форма (n,)

        u_xx = outer_tape.gradient(du_dx, x)  # Форма (n,)

        # Добавьте в конец функции перед return:
        # tf.debugging.check_numerics(u_tt, "Invalid u_tt")
        # tf.debugging.check_numerics(u_xx, "Invalid u_xx")

        del inner_tape, outer_tape

        u_model = du_dt - self.a * u_xx

        u_true = tf.zeros_like(u_model)
        diff = u_true - u_model
        phys_loss = tf.reduce_mean(tf.square(diff))

        return phys_loss

    @tf.function
    def boundary_loss_1(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        block,
        prev_model,
        prev_block,
        **kwargs
    ):
        """u(0, x) = sin(pi * x)"""
        x_wout_t = x_in[:, 1]
        t = tf.zeros_like(x_wout_t)
        x = tf.stack([t, x_wout_t], axis=1)

        x_norm = block.normalization(x)
        predicted = model(x_norm)
        predicted_unnorm: tf.Tensor = block.unnormalization(predicted)

        windowed = block.window_function(x)
        if prev_model is not None:
            x_left_norm = prev_block.normalization(x)
            predicted_left = prev_model(x_left_norm)
            predicted_unnorm_left: tf.Tensor = prev_block.unnormalization(
                predicted_left
            )
            windowed_left = prev_block.window_function(x)

            u_model = (
                windowed * predicted_unnorm + windowed_left * predicted_unnorm_left
            )
        else:
            u_model = windowed * predicted_unnorm
        u_model = tf.squeeze(u_model, axis=-1)

        u_true = tf.sin(math.pi * x_wout_t)
        diff = u_true - u_model
        phys_loss = tf.reduce_mean(tf.square(diff))

        return phys_loss

    @tf.function
    def boundary_loss_2(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        block,
        prev_model,
        prev_block,
        **kwargs
    ):
        """u(t, 0) = 0"""
        t_wout_x = x_in[:, 0]
        x_zeros = tf.zeros_like(t_wout_x)
        x = tf.stack([t_wout_x, x_zeros], axis=1)

        x_norm = block.normalization(x)
        predicted = model(x_norm)
        predicted_unnorm: tf.Tensor = block.unnormalization(predicted)

        windowed = block.window_function(x)
        if prev_model is not None:
            x_left_norm = prev_block.normalization(x)
            predicted_left = prev_model(x_left_norm)
            predicted_unnorm_left: tf.Tensor = prev_block.unnormalization(
                predicted_left
            )
            windowed_left = prev_block.window_function(x)

            u_model = (
                windowed * predicted_unnorm + windowed_left * predicted_unnorm_left
            )
        else:
            u_model = windowed * predicted_unnorm
        u_model = tf.squeeze(u_model, axis=-1)

        u_true = tf.constant([0.0], shape=u_model.shape, dtype=tf.float32)
        diff = u_true - u_model
        phys_loss = tf.reduce_mean(tf.square(diff))

        return phys_loss

    @tf.function
    def boundary_loss_3(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        block,
        prev_model,
        prev_block,
        **kwargs
    ):
        """u(t, 1) = 0"""
        t_wout_x = x_in[:, 0]
        x_zeros = tf.ones_like(t_wout_x)
        x = tf.stack([t_wout_x, x_zeros], axis=1)

        x_norm = block.normalization(x)
        predicted = model(x_norm)
        predicted_unnorm: tf.Tensor = block.unnormalization(predicted)

        windowed = block.window_function(x)
        if prev_model is not None:
            x_left_norm = prev_block.normalization(x)
            predicted_left = prev_model(x_left_norm)
            predicted_unnorm_left: tf.Tensor = prev_block.unnormalization(
                predicted_left
            )
            windowed_left = prev_block.window_function(x)

            u_model = (
                windowed * predicted_unnorm + windowed_left * predicted_unnorm_left
            )
        else:
            u_model = windowed * predicted_unnorm
        u_model = tf.squeeze(u_model, axis=-1)

        u_true = tf.constant([0.0], shape=u_model.shape, dtype=tf.float32)
        diff = u_true - u_model
        phys_loss = tf.reduce_mean(tf.square(diff))

        return phys_loss

    @tf.function
    def solution(self, x_in):
        t = x_in[:, 0]
        x = x_in[:, 1]
        res = tf.sin(math.pi * x) * tf.math.exp(-self.a * math.pi * math.pi * t)
        res = tf.expand_dims(res, axis=-1)
        return res

    def first_der(self, var, t, x):
        if var == "x":
            return self.first_der_x(t, x)
        elif var == "t":
            return self.first_der_t(t, x)
        else:
            raise NotImplementedError()

    @tf.function
    def first_der_x(self, t, x):
        return 2 * x

    @tf.function
    def first_der_t(self, t, x):
        return 2 * t
