from abc import ABC
from typing import Callable

import tensorflow as tf
from tensorflow import keras


def sign(x):
    return -1 if x < 0 else 1


class RelativeError(tf.keras.losses.Loss, ABC):
    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.NONE,
        name="relative",
        eps=1e-6,
        **kwargs
    ):
        super(RelativeError, self).__init__(reduction=reduction, name=name, **kwargs)
        self.eps = eps

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_upd = tf.where(abs(y_true) <= self.eps, 1.0 * sign(y_true), y_true)
        y = tf.math.divide(y_pred, y_upd)
        loss = tf.math.reduce_mean(tf.abs(y - 1))
        return loss


class RelativeAbsoluteError(tf.keras.losses.Loss, ABC):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="rae", **kwargs):
        super(RelativeAbsoluteError, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        true_mean = tf.reduce_mean(y_true)
        squared_error_num = tf.reduce_sum(tf.abs(y_true - y_pred))
        squared_error_den = tf.reduce_sum(tf.abs(y_true - true_mean))

        squared_error_den = tf.cond(
            pred=tf.equal(squared_error_den, tf.constant(0.0)),
            true_fn=lambda: tf.constant(1.0),
            false_fn=lambda: squared_error_den,
        )

        loss = squared_error_num / squared_error_den
        return loss


class MaxAbsoluteDeviation(tf.keras.losses.Loss, ABC):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.NONE, name="my_mae", **kwargs
    ):
        super(MaxAbsoluteDeviation, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.reduce_max(tf.math.abs(y_true - y_pred))
        return loss


class MaxAbsolutePercentageError(tf.keras.losses.Loss, ABC):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.NONE, name="maxAPE", **kwargs
    ):
        super(MaxAbsolutePercentageError, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.reduce_max(tf.math.abs((y_true - y_pred) / y_true))
        return loss


class RMSE(tf.keras.losses.Loss, ABC):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.NONE, name="maxAPE", **kwargs
    ):
        super(RMSE, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.sqrt(tf.math.reduce_mean((y_pred - y_true) ** 2))
        return loss


# Reduction should be set to None?
_losses: dict = {
    "Huber": keras.losses.Huber(),
    "LogCosh": keras.losses.LogCosh(),
    "MeanAbsoluteError": keras.losses.MeanAbsoluteError(),
    "MeanAbsolutePercentageError": keras.losses.MeanAbsolutePercentageError(),
    "MaxAbsolutePercentageError": MaxAbsolutePercentageError(),
    "MeanSquaredError": keras.losses.MeanSquaredError(),
    "RootMeanSquaredError": RMSE(),
    "MeanSquaredLogarithmicError": keras.losses.MeanSquaredLogarithmicError(),
    "RelativeError": RelativeError(),
    "RelativeAbsoluteError": RelativeAbsoluteError(),
    "MaxAbsoluteDeviation": MaxAbsoluteDeviation(),
}


def get_loss(name: str):
    return _losses.get(name)


def get_all_loss_functions() -> dict[str, Callable]:
    return _losses
