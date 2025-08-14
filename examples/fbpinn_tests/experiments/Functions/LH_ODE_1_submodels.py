import tensorflow as tf
from examples.fbpinn_tests.phys_losses import PhysLoss


class LH_ODE_1(PhysLoss):
    def __init__(self, description: str = "", **kwargs):
        description = "y'' + 100 y = 0, y(0) = 0"
        super().__init__(description)

        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (self.boundary_loss_1, [(0.0, 0.0)]),
        ]

    @tf.function
    def phys_loss(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x,
        block,
        prev_model,
        prev_block,
        **kwargs
    ):
        """y'' + 100 y = 0"""
        tape.watch(x)

        with tf.GradientTape() as outer_tape:
            outer_tape.watch(x)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(x)
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

                    u = (
                        windowed * predicted_unnorm
                        + windowed_left * predicted_unnorm_left
                    )
                else:
                    u = windowed * predicted_unnorm

                u = tf.squeeze(u, axis=-1)
                u_x = inner_tape.gradient(u, x)

        u_xx = outer_tape.gradient(u_x, x)
        u_model = u_xx + 100.0 * u
        u_true = tf.zeros_like(u_model)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def boundary_loss_1(
        self, model, tape: tf.GradientTape, x, block, prev_model, prev_block, **kwargs
    ):
        """y(0) = 0"""
        ic_x = tf.constant([[0.0]])
        tape.watch(ic_x)

        x_norm = block.normalization(ic_x)
        predicted = model(x_norm)
        predicted_unnorm: tf.Tensor = block.unnormalization(predicted)
        windowed = block.window_function(ic_x)
        u = windowed * predicted_unnorm

        u_true = tf.zeros_like(u)
        phys_loss = tf.reduce_mean(tf.square(u_true - u))

        return phys_loss

    @tf.function
    def boundary_loss_2(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x,
        block,
        prev_model,
        prev_block,
        **kwargs
    ):
        """y'(0) = 10"""
        x = tf.constant([[0.0]])
        tape.watch(x)

        with tf.GradientTape() as inner_tape:
            inner_tape.watch(x)
            x_norm = block.normalization(x)
            predicted = model(x_norm)
            predicted_unnorm: tf.Tensor = block.unnormalization(predicted)
            windowed = block.window_function(x)
            u = windowed * predicted_unnorm
            u_x = inner_tape.gradient(u, x)

        u_true = tf.constant([[10.0]])
        phys_loss = tf.reduce_mean(tf.square(u_true - u_x))

        return phys_loss

    @tf.function
    def solution(self, x):
        return tf.sin(10 * x)
