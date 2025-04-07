from datetime import datetime
import random
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.optimizer.set_jit(True)
from keras import mixed_precision

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

from matplotlib import pyplot as plt

import mlflow
from degann.geometry import RectangleDomain
from degann.geometry.decomposition import Block
from degann.networks.topology.tffbpinn3 import TensorflowFBPINN
from tensorflow.keras.callbacks import EarlyStopping
from examples.fbpinn_tests.sin_losses import fbpinn_orig_sin_cos, boundary_loss_sin_cos
from examples.fbpinn_tests.plot_functions import plot_each_submodel, plot_model

original_function = lambda x: tf.sin(15.0 * x) + tf.cos(x)


# Создаем расписание, зависящее от эпох
class EpochBasedScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_rate, last_rate, steps, steps_per_epoch):
        self.initial_rate = initial_rate
        self.last_rate = last_rate
        self.steps = steps
        self.steps_per_epoch = steps_per_epoch

    def __call__(self, step):
        epoch = tf.cast(step // self.steps_per_epoch, tf.float32)  # Текущая эпоха
        lr = (
            self.initial_rate
            - (self.initial_rate - self.last_rate) / self.steps * epoch
        )
        return tf.maximum(lr, self.last_rate)  # Не опускаемся ниже last_rate


phys_loss = fbpinn_orig_sin_cos
which = phys_loss.__doc__

run_id = random.randint(1, 10000)
run_name = f"FBPINN_{run_id}"
mlflow.start_run(run_name=run_name)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("mlflow.runName", f"model{run_id}")
# Уникальная директория для каждого эксперимента
log_dir = f"logs/fit/model{run_id}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
summary_writer = tf.summary.create_file_writer(log_dir)


initial_rate = 2e-2
last_rate = 1e-4
steps = 20000
# lr = lambda step: max(last_rate, initial_rate - (initial_rate - last_rate) / steps * step)
lr = EpochBasedScheduler(initial_rate, last_rate, steps, 1)
# lr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=initial_rate, first_decay_steps=steps, t_mul=1.0, m_mul=0.7, alpha=last_rate)

callbacks = None
callbacks = [EarlyStopping(monitor="loss", patience=400)]
lc = 0
rc = np.pi * 4
mlflow.log_param("left_bound", lc)
mlflow.log_param("right_bound", rc)

model_config = {
    "input_size": 1,
    "output_size": 1,
    "activation_func": ["tanh", "tanh", "linear"],
    "block_size": 0.72,
    "models_size": [32, 32],
    "overlap": 0.3,
    "offset": True,
    "points_per_block": 100,
}
mlflow.log_params(model_config)

nn = TensorflowFBPINN(
    **model_config,
    # physic_loss=physic_loss,
    # boundary_loss=[(boundary_loss_1, [0.0]), (boundary_loss_2, [0.0])],
    physic_loss=phys_loss,
    boundary_loss=[(boundary_loss_sin_cos, [0.0])],
    domain=RectangleDomain([lc], [rc]),
    summary_writer=summary_writer,
)

fb = nn.blocks[0][1]
fb.data = np.linspace(lc, fb.right_up_corner, 100)
print("Number of submodels", len(nn.blocks))

nn.custom_compile(optimizer="AdamW", rate=lr, loss_func="MSE", run_eagerly=False)

x = np.linspace(lc, rc, num=5000, dtype=np.float32).reshape((-1, 1))
# y = tf.cast(original_function(x), dtype=tf.float32)
y = original_function(x)

tf.summary.trace_on(graph=True, profiler=False)
y_pred_before_train = nn.predict(x)
with summary_writer.as_default():
    tf.summary.trace_export(name="model_graph", step=0, profiler_outdir=log_dir)

y_pred_before_train = nn.predict(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

train_config = {
    "epochs": 20_000,
    "patience": 2500,
    "eval_interval": 1,
    "batch_size": 50,
    "log_interval": 1000,
    "mode": "sequence",
}
mlflow.log_params(train_config)

nn.train(
    **train_config,
    verbose=0,
    callbacks=callbacks,
    val_function=original_function,
)
loss_after_train = nn.evaluate(x, y, verbose=0)
print("Before", loss_before_train)
print("After", loss_after_train)
mlflow.log_metric("Loss before training", loss_before_train)
mlflow.log_metric("Loss after training", loss_after_train)

fig, axes = plt.subplots(nrows=2, ncols=1)
# plot_each_submodel(x, y, nn, axes[0])
plot_model(x, y, nn, axes[1])
