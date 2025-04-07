from datetime import datetime
import random
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    print("We got a GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("Sorry, no GPU for you...")
from matplotlib import pyplot as plt

import mlflow
from degann.geometry import RectangleDomain
from degann.geometry.decomposition import Block
from degann.networks.topology.tffbpinn import TensorflowFBPINN
from tensorflow.keras.callbacks import EarlyStopping
from examples.fbpinn_tests.sin_losses import (
    fbpinn_orig_sin_full,
    boundary_loss_full,
    boundary_loss_full2,
)
from examples.fbpinn_tests.plot_functions import plot_each_submodel, plot_model

original_function = lambda x: tf.sin(1 * x)


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


phys_loss = fbpinn_orig_sin_full
which = phys_loss.__doc__

run_id = random.randint(1, 10000)
run_name = f"FBPINN_{run_id}"
mlflow.start_run(run_name=run_name)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("mlflow.runName", f"model{run_id}")
# Уникальная директория для каждого эксперимента
log_dir = f"logs/fit/model{run_id}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
summary_writer = tf.summary.create_file_writer(log_dir)


initial_rate = 1e-4
last_rate = 1e-6
steps = 100000
mlflow.log_param("initial learning rate", initial_rate)
mlflow.log_param("last learning rate", last_rate)
mlflow.log_param("scheduler steps", steps)
lr = EpochBasedScheduler(initial_rate, last_rate, steps, 1)
# lr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=initial_rate, first_decay_steps=steps, t_mul=1.0, m_mul=0.7, alpha=last_rate)

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
    "offset": False,
    "points_per_block": 100,
}
mlflow.log_params(model_config)

nn = TensorflowFBPINN(
    **model_config,
    physic_loss=phys_loss,
    boundary_loss=[(boundary_loss_full, [0.0]), (boundary_loss_full2, [0.0])],
    domain=RectangleDomain([lc], [rc]),
    summary_writer=summary_writer,
)

fb = nn.blocks[0][1]
fb.data = tf.linspace(
    tf.constant(lc, dtype=tf.float32),
    tf.constant(fb.right_up_corner, dtype=tf.float32),
    100,
)
print("Number of submodels", len(nn.blocks))

nn.custom_compile(optimizer="AdamW", rate=lr, loss_func="MSE", run_eagerly=False)

x = tf.reshape(
    tf.linspace(
        tf.constant(lc, dtype=tf.float32), tf.constant(rc, dtype=tf.float32), num=5000
    ),
    (-1, 1),
)
y = original_function(x)

tf.summary.trace_on(graph=True, profiler=False)
y_pred_before_train = nn.predict(x)
with summary_writer.as_default():
    tf.summary.trace_export(name="model_graph", step=0, profiler_outdir=log_dir)

y_pred_before_train = nn.predict(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

train_config = {
    "epochs": 30_000,
    "patience": 3000,
    "eval_interval": 1,
    "batch_size": 5000,
    "log_interval": 1000,
    "mode": "full",
}
mlflow.log_params(train_config)

# nn.full_train(
nn.train(
    **train_config,
    verbose=0,
    val_function=original_function,
    val_input=x,
    png_salt=str(run_id),
)
loss_after_train = nn.evaluate(x, y, verbose=0)
print("Before", loss_before_train)
print("After", loss_after_train)
mlflow.log_metric("Loss before training", loss_before_train)
mlflow.log_metric("Loss after training", loss_after_train)
mlflow.end_run()

fig, axes = plt.subplots(nrows=2, ncols=1)
plot_each_submodel(x, y, nn, axes[0])
plot_model(x, y, nn, axes[1])
plt.show()
