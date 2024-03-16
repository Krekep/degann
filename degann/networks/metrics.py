from abc import ABC
from typing import Callable

import tensorflow as tf
from tensorflow import keras

from degann.networks.losses import get_all_loss_functions


# Non-differentiable
class InlierRatio(tf.keras.losses.Loss, ABC):
    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.NONE,
        name="inlier_ratio",
        threshold=0.05,
        **kwargs
    ):
        super(InlierRatio, self).__init__(reduction=reduction, name=name, **kwargs)
        self.threshold = threshold

    def __call__(self, y_true, y_pred, sample_weight=None):
        y = tf.math.divide((y_true - y_pred), tf.where(y_true == 0.0, 1.0, y_true))
        loss = tf.math.reduce_mean(tf.where(tf.abs(y) <= self.threshold, 0.0, 1.0))
        return loss


class MaxDeviation(tf.keras.losses.Loss, ABC):
    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.NONE,
        name="max_deviation",
        treeshold=0.05,
        **kwargs
    ):
        super(MaxDeviation, self).__init__(reduction=reduction, name=name, **kwargs)
        self.treeshold = treeshold

    def __call__(self, y_true, y_pred, sample_weight=None):
        y = tf.math.divide((y_true - y_pred), tf.where(y_true == 0.0, 1.0, y_true))
        loss = tf.math.reduce_max(tf.where(tf.abs(y) <= self.treeshold, 0.0, abs(y)))
        return loss


class MeanDeviation(tf.keras.losses.Loss, ABC):
    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.NONE,
        name="mean_deviation",
        treeshold=0.05,
        **kwargs
    ):
        super(MeanDeviation, self).__init__(reduction=reduction, name=name, **kwargs)
        self.treeshold = treeshold

    def __call__(self, y_true, y_pred, sample_weight=None):
        y = tf.math.divide((y_true - y_pred), tf.where(y_true == 0.0, 1.0, y_true))
        loss = tf.math.reduce_mean(
            tf.where(tf.abs(y) <= self.treeshold, self.treeshold, abs(y))
        )
        return loss


_metrics: dict = {
    "RootMeanSquaredError": keras.metrics.RootMeanSquaredError(),
    "InlierRatio": InlierRatio(),
    "MaxDeviation": MaxDeviation(),
    "MeanDeviation": MeanDeviation(),
}

_metrics = dict(get_all_loss_functions(), **_metrics)


def get_metric(name: str):
    return _metrics.get(name)


def get_all_metric_functions() -> dict[str, Callable]:
    return _metrics
