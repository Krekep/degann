import math
import tensorflow as tf

from tensorflow.debugging import Assert, assert_equal
from examples.fbpinn_tests.phys_losses import PhysLoss


class LH_PDE1(PhysLoss):
    def __init__(self, description: str = "", a: float = 1, **kwargs):
        description = "d^2u/dt^2 - 4 * d^2u/dx^2 = 0, u(0, x) = sin(pi * x) + 1/2 * sin(4 * pi * x), u_t(0, x) = 0, u(t, 0) = 0, u(t, 1) = 0"
        self.equation_class = "Hyperbolic PDE"
        self.a = tf.constant(a, dtype=tf.float32)
        self.time_input = True
        super().__init__(description)

        self.full_losses = [self.phys_loss]
        self.sub_losses = [
            (self.boundary_loss_1, [(0.0, 0.0), None]),
            (self.boundary_loss_2, [(0.0, 0.0), None]),
            (self.boundary_loss_3, [None, (0.0, 0.0)]),
            (self.boundary_loss_4, [None, (1.0, 1.0)]),
        ]

    @tf.function
    def phys_loss(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        active_models,
        **kwargs
    ):
        """d^2u/dt^2 - 4 * d^2u/dx^2 = 0"""
        t, x = x_in[:, 0:1], x_in[:, 1:2]
        # tape.watch(t)
        # tape.watch(x)        
        
        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch(t)
            outer_tape.watch(x)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(t)
                inner_tape.watch(x)
                u = model(tf.concat([t, x], axis=1), active_models=active_models)
                # assert_equal(u.shape.rank, 1, "Model() have len(shape) != 1")
                # if u.shape.rank > 1 and u.shape[-1] == 1:
                u = tf.squeeze(u, axis=-1)
                du_dt, du_dx = inner_tape.gradient(u, [t, x])  # Форма (n,)
            # du_dx = inner_tape.gradient(u, x)  # Форма (n,)
        
        u_tt = outer_tape.gradient(du_dt, t)  # Форма (n,)
        u_xx = outer_tape.gradient(du_dx, x)  # Форма (n,)
        
        # Добавьте в конец функции перед return:
        # tf.debugging.check_numerics(u_tt, "Invalid u_tt")
        # tf.debugging.check_numerics(u_xx, "Invalid u_xx")
        
        del inner_tape, outer_tape
        
        u_model = u_tt - 4.0 * u_xx
        
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
        active_models,
        **kwargs
    ):
        """u(0, x) = sin(pi * x) + 1/2 * sin(4 * pi * x)"""
        x_wout_t = x_in[:, 1]
        t = tf.zeros_like(x_wout_t)
        x = tf.stack([t, x_wout_t], axis=1)
        
        u_model = model(x, active_models=active_models)
        u_model = tf.squeeze(u_model, axis=-1)

        u_true = tf.sin(math.pi * x_wout_t) + 1/2 * tf.sin(4 * math.pi * x_wout_t)
        print("Boundary loss 1", u_true.shape, u_model.shape)
        diff = u_true - u_model
        phys_loss = tf.reduce_mean(tf.square(diff))

        return phys_loss

    @tf.function
    def boundary_loss_2(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        active_models,
        **kwargs
    ):
        """u_t(0, x) = 0"""
        # tape.watch(x_in)
        x_wout_t = x_in[:, 1]
        t = tf.zeros_like(x_wout_t)
        x = tf.stack([t, x_wout_t], axis=1)
        # tape.watch(x)
        # tape.watch(t)

        # u_model = model(x, active_models=active_models)
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(x_wout_t)
            inner_tape.watch(t)
            u_model = model(tf.stack([t, x_wout_t], axis=1), active_models=active_models)
            # assert_equal(u_model.shape.rank, 1, "Model() have len(shape) != 1")
            # if u.shape.rank > 1 and u.shape[-1] == 1:
            u_model = tf.squeeze(u_model, axis=-1)
        grads = inner_tape.gradient(u_model, t)
        u_t = grads
        # u_t = tf.squeeze(u_t, axis=-1)

        u_true = tf.constant(0, shape=u_t.shape, dtype=tf.float32)
        # assert_equal(u_true.shape, u_t.shape, "Discrepancy between the shapes of predicted and true data")
        diff = u_true - u_t
        # print("Boundary loss 2", diff.shape, u_t.shape, u_true.shape)
        # phys_loss = tf.reduce_mean(tf.abs(diff))
        phys_loss = tf.reduce_mean(tf.square(diff))

        return phys_loss
        # return tf.constant(0, dtype=tf.float32)

    @tf.function
    def boundary_loss_3(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        active_models,
        **kwargs
    ):
        """u(t, 0) = 0"""
        t_wout_x = x_in[:, 0]
        x_zeros = tf.zeros_like(t_wout_x)
        x = tf.stack([t_wout_x, x_zeros], axis=1)
        
        u_model = model(x, active_models=active_models)
        u_model = tf.squeeze(u_model, axis=-1)

        u_true = tf.constant([0.0], shape=u_model.shape, dtype=tf.float32)
        diff = u_true - u_model
        phys_loss = tf.reduce_mean(tf.square(diff))

        return phys_loss
    
    @tf.function
    def boundary_loss_4(
        self,
        model: tf.keras.Model,
        tape: tf.GradientTape,
        x_in,
        active_models,
        **kwargs
    ):
        """, u(t, 1) = 0"""
        t_wout_x = x_in[:, 0]
        x_zeros = tf.zeros_like(t_wout_x)
        x = tf.stack([t_wout_x, x_zeros], axis=1)
        
        u_model = model(x, active_models=active_models)
        u_model = tf.squeeze(u_model, axis=-1)

        u_true = tf.constant([0.0], shape=u_model.shape, dtype=tf.float32)
        diff = u_true - u_model
        phys_loss = tf.reduce_mean(tf.square(diff))

        return phys_loss
    
    @tf.function
    def solution(self, x_in):
        t = x_in[:, 0]
        x = x_in[:, 1]
        res = tf.sin(math.pi * x) * tf.cos(2 * math.pi * t) + 1/2 * tf.sin(4 * math.pi * x) * tf.cos(8 * math.pi * t)
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
    
