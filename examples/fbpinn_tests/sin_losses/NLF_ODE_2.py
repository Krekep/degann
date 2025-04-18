import tensorflow as tf
from examples.fbpinn_tests.phys_losses import PhysLoss


class NLF_ODE_2(PhysLoss):
    def  __init__(self, description: str = "", omega1: float = 10, omega2: float = 1, **kwargs):
        description = "du/dx = omega1 * cos(omega1*x) - omega2 * sin(omega2 x), y(0) = 0"
        self.omega1 = omega1
        self.omega2 = omega2
        super().__init__(description)
        
        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (
                self.boundary_loss_1, 
                [(0.0, 0.0)]
            ),
        ]
  
    @tf.function
    def phys_loss(self, model: tf.keras.Model, tape: tf.GradientTape, x, **kwargs):
        """du/dx = omega * cos(omega*x)"""
        tape.watch(x)

        u = model(x)
        u_x = tape.gradient(u, x)
        u_model = u_x - self.omega1 * tf.cos(self.omega1 * x) + self.omega2 * tf.sin(self.omega2 * x)
        u_true = tf.zeros_like(u_model)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def boundary_loss_1(self, model: tf.keras.Model, tape: tf.GradientTape, x, **kwargs):
        """y(0) = 1"""
        x = tf.constant([[0.0]])
        tape.watch(x)

        u_model = model(x)
        
        u_true = tf.ones_like(u_model)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def solution(self, x):
        return tf.sin(self.omega1 * x) + tf.cos(self.omega2 * x)

    @tf.function
    def first_der(self, x):
        return self.omega1 * tf.cos(self.omega1 * x) - self.omega2 * tf.sin(self.omega2 * x)
