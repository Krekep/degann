import tensorflow as tf
from examples.fbpinn_tests.phys_losses import PhysLoss


class LF_ODE_1(PhysLoss):
    def __init__(self, description: str = "", omega: float = 10, **kwargs):
        description = f"du/dx = {omega} * cos({omega}*x), y(0) = 0"
        self.omega = omega
        super().__init__(description)

        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (self.boundary_loss_1, [(0.0, 0.0)]),
        ]

    @tf.function
    def phys_loss(
        self, model: tf.keras.Model, tape: tf.GradientTape, x, active_models, **kwargs
    ):
        """du/dx = omega * cos(omega*x)"""
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(x)

            u = model(x, active_models=active_models)
            u_x = inner_tape.gradient(u, x)
            u_model = u_x - self.omega * tf.cos(self.omega * x)
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
    def solution(self, x):
        return tf.sin(self.omega * x)
