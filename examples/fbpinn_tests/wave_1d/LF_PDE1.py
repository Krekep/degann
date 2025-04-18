import tensorflow as tf
from examples.fbpinn_tests.phys_losses import PhysLoss


class LF_PDE1(PhysLoss):
    def __init__(self, description: str = "", a: float = 1, X: float = 1.0, **kwargs):
        description = "d^2u/dt^2 = a^2 * d^2u/dx^2, u(0, x) = x^2, u_t(0, x) = 0"
        self.equation_class = "Hyperbolic PDE"
        self.a = a
        self.time_input = True
        super().__init__(description)

        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (self.boundary_loss_1, [(0.0, 0.0), None]),
            (self.boundary_loss_2, [(0.0, 0.0), None]),
        ]

    @tf.function
    def phys_loss(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        noise,
        active_models,
        **kwargs
    ):
        """d^2u/dt^2 = a^2 * d^2u/dx^2"""
        t, x = x_in[:, 0], x_in[:, 1]
        tape.watch(t)
        tape.watch(x)
        tape.watch(x_in)

        u = model(x_in, active_models)
        grads = tape.gradient(u, x_in)
        u_xx = tape.gradient(grads, x_in)[:, 1]
        u_tt = tape.gradient(grads, x_in)[:, 0]

        u_model = u_tt - self.a * self.a * u_xx
        u_true = tf.zeros_like(u_model)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def boundary_loss_1(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        noise,
        active_models,
        **kwargs
    ):
        """u(0, x) = x^2"""
        x_in = x_in[:, 1]
        t = tf.zeros_like(x_in)
        x = tf.stack([t, x_in], axis=1)
        tape.watch(x)

        u_model = model(x, active_models)

        u_true = tf.math.multiply(x_in, x_in)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

        return phys_loss

    @tf.function
    def boundary_loss_2(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        noise,
        active_models,
        **kwargs
    ):
        """u_t(0, x) = 0"""
        x_in = x_in[:, 1]
        t = tf.zeros_like(x_in)
        x = tf.stack([t, x_in], axis=1)
        tape.watch(x)
        tape.watch(t)

        u_model = model(x, active_models)
        grads = tape.gradient(u_model, x)
        u_t = grads[:, 0]

        u_true = tf.constant(0, shape=u_t.shape, dtype=tf.float32)
        phys_loss = tf.reduce_mean(tf.square(u_true - u_t))

        return phys_loss

    @tf.function
    def solution(self, x):
        t = x[:, 0]
        x = x[:, 1]
        return t * t + x * x

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
