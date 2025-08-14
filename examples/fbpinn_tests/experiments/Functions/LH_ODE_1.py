import tensorflow as tf
from examples.fbpinn_tests.phys_losses import PhysLoss


class LH_ODE_1(PhysLoss):
    def __init__(self, description: str = "", **kwargs):
        description = f"y'' + 100 y = 0, y(0) = 0"
        super().__init__(description)

        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (self.boundary_loss_1, [(0.0, 0.0)]),
            (self.boundary_loss_2, [(0.0, 0.0)]),
        ]

    @tf.function
    def phys_loss(
        self, model: tf.keras.Model, tape: tf.GradientTape, x, active_models, **kwargs
    ):
        """y'' + 100 y = 0"""
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(x)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(x)
                u = model(x, active_models=active_models)
                u = tf.squeeze(u, axis=-1)
                u_x = inner_tape.gradient(u, x)

        u_xx = outer_tape.gradient(u_x, x)
        u_model = u_xx + 100.0 * u
        u_true = tf.zeros_like(u_model)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def boundary_loss_1(
        self, model: tf.keras.Model, tape: tf.GradientTape, x, active_models, **kwargs
    ):
        """y(0) = 0"""
        x = tf.constant([[0.0]])
        tape.watch(x)

        u_model = model(x, active_models=active_models)

        u_true = tf.zeros_like(u_model)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def boundary_loss_2(
        self, model: tf.keras.Model, tape: tf.GradientTape, x, active_models, **kwargs
    ):
        """y'(0) = 10"""
        x = tf.constant([[0.0]])
        tape.watch(x)

        with tf.GradientTape() as inner_tape:
            inner_tape.watch(x)
            u = model(x, active_models=active_models)
            u = tf.squeeze(u, axis=-1)
            u_x = inner_tape.gradient(u, x)

        u_true = tf.constant([[10.0]])
        phys_loss = tf.reduce_mean(tf.square(u_true - u_x))

        return phys_loss

    @tf.function
    def solution(self, x):
        return tf.sin(10 * x)
