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
    def phys_loss(self, model: tf.keras.Model, tape: tf.GradientTape, x, **kwargs):
        """y' + (y - 2x) / (x + 0.1) = 0"""
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(x)

            u = model(x)
            u_x = inner_tape.gradient(u, x)
            u_model = u_x + (u - 2 * x) / (x + 0.1)
            u_true = tf.zeros_like(u_model)
            phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def boundary_loss_1(
        self, model: tf.keras.Model, tape: tf.GradientTape, x, **kwargs
    ):
        """y(0) = 2"""
        x = tf.constant([[0.0]])
        tape.watch(x)

        u_model = model(x)

        u_true = tf.constant(2, dtype=tf.float32, shape=u_model.shape)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def solution(self, x):
        return (x * x + 0.2) / (x + 0.1)
