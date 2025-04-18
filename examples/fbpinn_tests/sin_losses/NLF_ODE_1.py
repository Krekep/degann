import tensorflow as tf
from examples.fbpinn_tests.phys_losses import PhysLoss


class NLF_ODE_1(PhysLoss):
    def  __init__(self, description: str = "", omega: float = 10, **kwargs):
        description = "du/dx = omega * cos(omega*x), y(0) = 0"
        self.omega = omega
        super().__init__(description)
        
        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (
                self.boundary_loss_1, 
                {
                    "x": (0.0, 0.0)
                }
            ),
            (
                self.boundary_loss_2, 
                {
                    "x": (0.0, 0.0)
                }
            ),
        ]

    @tf.function
    def phys_loss(self, model: tf.keras.Model, tape: tf.GradientTape, x, **kwargs):
        """du/dx = omega * cos(omega*x)"""
        tape.watch(x)

        u = model(x)
        u_x = tape.gradient(u, x)
        u_model = u_x - self.omega * tf.cos(self.omega * x)
        u_true = tf.zeros_like(u_model)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def boundary_loss_1(self, model: tf.keras.Model, tape: tf.GradientTape, x, **kwargs):
        """y(0) = 0"""
        x = tf.constant([[0.0]])
        tape.watch(x)

        u_model = model(x)
        
        u_true = tf.zeros_like(u_model)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def boundary_loss_2(self, model: tf.keras.Model, tape: tf.GradientTape, x, **kwargs):
        """y'(0) = 1"""
        x = tf.constant([[0.0]])
        tape.watch(x)

        u_model = model(x)
        
        u_true = tf.ones_like(u_model)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    def solution(self, x):
        return tf.sin(self.omega * x)
