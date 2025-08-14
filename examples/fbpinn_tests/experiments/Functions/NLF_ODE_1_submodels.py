import tensorflow as tf
from examples.fbpinn_tests.phys_losses import PhysLoss


class NLF_ODE_1(PhysLoss):
    def __init__(self, description: str = "", **kwargs):
        description = "y' + (y - 2x) / (x + 0.1) = 0, y(0) = 2"
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
        """y' + (y - 2x) / (x + 0.1) = 0"""
        tape.watch(x)

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

            u = windowed * predicted_unnorm + windowed_left * predicted_unnorm_left
        else:
            u = windowed * predicted_unnorm

        u_x = tape.gradient(u, x)
        u_model = u_x + (u - 2 * x) / (x + 0.1)
        u_true = tf.zeros_like(u_model)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def boundary_loss_1(
        self,
        model,
        tape: tf.GradientTape,
        x,
        block,
        prev_model,
        prev_block,
    ):
        """y(0) = 2"""
        ic_x = tf.constant([[0.0]])
        tape.watch(ic_x)

        x_norm = block.normalization(ic_x)
        predicted = model(x_norm)
        predicted_unnorm: tf.Tensor = block.unnormalization(predicted)
        windowed = block.window_function(ic_x)
        u = windowed * predicted_unnorm

        u_true = tf.constant(2, dtype=tf.float32, shape=u.shape)
        phys_loss = tf.reduce_mean(tf.square(u_true - u))

        return phys_loss

    @tf.function
    def solution(self, x):
        return (x * x + 0.2) / (x + 0.1)
