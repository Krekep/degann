from abc import ABC
from typing import Callable

import tensorflow as tf
from tensorflow import keras

from src.networks.losses import get_all_loss_functions


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
    # "AUC": keras.metrics.AUC(),                                                         # Not usable in my task
    # "Accuracy": keras.metrics.Accuracy(),                                               # Not usable in my task
    # "BinaryAccuracy": keras.metrics.BinaryAccuracy(),                                   # Not usable in my task
    # "BinaryIoU": keras.metrics.BinaryIoU(),                                             # Not usable in my task
    # "FalseNegatives": keras.metrics.FalseNegatives(),                                   # Not usable in my task
    # "FalsePositives": keras.metrics.FalsePositives(),                                   # Not usable in my task
    # "IoU": keras.metrics.IoU(),                                                         # Not usable in my task
    # "MeanIoU": keras.metrics.MeanIoU(),                                                 # Not usable in my task
    # "OneHotIoU": keras.metrics.OneHotIoU(),                                             # Not usable in my task
    # "OneHotMeanIoU": keras.metrics.OneHotMeanIoU(),                                     # Not usable in my task
    # "Precision": keras.metrics.Precision(),                                             # Not usable in my task
    # "PrecisionAtRecall": keras.metrics.PrecisionAtRecall(),                             # Not usable in my task
    # "Recall": keras.metrics.Recall(),                                                   # Not usable in my task
    # "RecallAtPrecision": keras.metrics.RecallAtPrecision(),                             # Not usable in my task
    # "SensitivityAtSpecificity": keras.metrics.SensitivityAtSpecificity(),               # Not usable in my task
    # "SparseCategoricalAccuracy": keras.metrics.SparseCategoricalAccuracy(),             # Not usable in my task
    # "SparseTopKCategoricalAccuracy": keras.metrics.SparseTopKCategoricalAccuracy(),     # Not usable in my task
    # "SpecificityAtSensitivity": keras.metrics.SpecificityAtSensitivity(),               # Not usable in my task
    # "Sum": keras.metrics.Sum(),                                                         # Not usable in my task
    # "TopKCategoricalAccuracy": keras.metrics.TopKCategoricalAccuracy(),                 # Not usable in my task
    # "TrueNegatives": keras.metrics.TrueNegatives(),                                     # Not usable in my task
    # "TruePositives": keras.metrics.TruePositives(),                                     # Not usable in my task
    #
    # "Mean": keras.metrics.Mean(),  # Not usable in my task? Take only one collection of values (not y_true and y_pred)
    # "MeanRelativeError": keras.metrics.MeanRelativeError( * need normalize |y_true - y_pred| / normalizer * ),                             # Not usable in my task?
    # "MeanTensor": keras.metrics.MeanTensor(),  # Not usable in my task? Take only one collection of values (not y_true and y_pred)
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
