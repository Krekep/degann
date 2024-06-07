import gc
import time

import keras.backend as k
from keras.callbacks import Callback
from keras.callbacks import History


class MemoryCleaner(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


class MeasureTrainTime(Callback):
    """
    Callback for measuring time.
    Supports measuring training time,
    measuring the time of each epoch during training,
    and measuring the running time of the predict method
    """

    def __init__(self):
        super(MeasureTrainTime, self).__init__()
        self.start_train_time = 0
        self.end_train_time = 0

        self.start_evaluate_time = 0
        self.end_evaluate_time = 0

        self.start_predict_time = 0
        self.end_predict_time = 0

        self.start_epoch_time = 0
        self.end_epoch_time = 0

    def on_test_begin(self, logs=None):
        self.model.trained_time["predict_time"] = 0
        self.start_evaluate_time = time.perf_counter()

    def on_test_end(self, logs=None):
        self.end_evaluate_time = time.perf_counter()
        self.model.trained_time["predict_time"] = (
            self.end_evaluate_time - self.start_evaluate_time
        )

    def on_predict_begin(self, logs=None):
        self.model.trained_time["predict_time"] = 0
        self.start_predict_time = time.perf_counter()

    def on_predict_end(self, logs=None):
        self.end_predict_time = time.perf_counter()
        self.model.trained_time["predict_time"] = (
            self.end_predict_time - self.start_predict_time
        )

    def on_train_begin(self, logs=None):
        self.model.trained_time["train_time"] = 0.0
        self.model.trained_time["epoch_time"] = []
        self.start_train_time = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self.start_epoch_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.end_epoch_time = time.perf_counter()
        self.model.trained_time["epoch_time"].append(
            self.end_epoch_time - self.start_epoch_time
        )

    def on_train_end(self, logs=None):
        self.end_train_time = time.perf_counter()
        self.model.trained_time["train_time"] = (
            self.end_train_time - self.start_train_time
        )


class LightHistory(History):
    """
    Class based on Keras.History,
    but which only stores information about the last training epoch,
    not the entire process
    """

    def __init__(self):
        super(History, self).__init__()
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch = epoch
        for k, v in logs.items():
            self.history[k] = v

        # Set the history attribute on the model after the epoch ends. This will
        # make sure that the state which is set is the latest one.
        self.model.history = self
