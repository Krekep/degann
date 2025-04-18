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
from examples.fbpinn_tests.plot_functions import plot_each_submodel, plot_model
from wave_1d.LF_PDE1 import LF_PDE1


# Создаем расписание, зависящее от эпох
class EpochBasedScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_start, warmup_len, initial_rate, last_rate, steps, steps_per_epoch):
        self.initial_rate = initial_rate
        self.last_rate = last_rate
        self.steps = steps
        self.steps_per_epoch = steps_per_epoch
        self.warmup_start = warmup_start
        self.warmup_len = warmup_len
        self.current_step = 0

    def __call__(self, step):
        self.current_step += 1
        if self.current_step <= self.warmup_len:
            lr = self.warmup_start + (self.initial_rate - self.warmup_start) / self.warmup_len * self.current_step
        else:
            lr = (
                self.initial_rate
                - (self.initial_rate - self.last_rate) / (self.steps - self.warmup_len) * self.current_step
            )
            lr = tf.maximum(lr, self.last_rate)
        return lr

lc = [0]
rc = [1.0]
pde = LF_PDE1()
phys_loss = pde.phys_loss
which = pde.description

mlflow.set_experiment("FBPINN PDE Wave 1D")
run_id = random.randint(1, 10000)
run_name = f"FBPINN_{run_id}"
mlflow.start_run(run_name=run_name)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("mlflow.runName", f"model{run_id}")
mlflow.set_tag("Class", pde.equation_class)
log_dir = f"logs/fit/model{run_id}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
summary_writer = tf.summary.create_file_writer(log_dir)


warmup_start = 0
warmup_len = 2000
initial_rate = 1e-3
last_rate = 1e-6
steps = 50000
mlflow.log_param("warmup learning rate start", warmup_start)
mlflow.log_param("warmup steps", warmup_len)
mlflow.log_param("initial learning rate", initial_rate)
mlflow.log_param("initial learning rate", initial_rate)
mlflow.log_param("last learning rate", last_rate)
mlflow.log_param("scheduler steps", steps)
lr = EpochBasedScheduler(warmup_start, warmup_len, initial_rate, last_rate, steps, 1)

mlflow.log_param("left_bound", lc)
mlflow.log_param("right_bound", rc)

model_config = {
    "input_size": 2,
    "output_size": 1,
    "activation_func": ["tanh", "tanh", "linear"],
    "block_size": 0.4,
    "models_size": [32, 32],
    "overlap": 0.2,
    "offset": False,
    "points_per_block": 200,
    "time_input": True,
    "end_time": 2.0,
    "time_step": 0.2
}
mlflow.log_params(model_config)

nn = TensorflowFBPINN(
    **model_config,
    physic_loss=phys_loss,
    boundary_loss=pde.sub_losses,
    domain=RectangleDomain(lc, rc),
    summary_writer=summary_writer,
)

fb = nn.blocks[0][1]
fb.data = tf.linspace(
    tf.constant(lc, dtype=tf.float32),
    tf.constant(fb.right_up_corner, dtype=tf.float32),
    model_config["points_per_block"] // 2,
)
print("Number of submodels", len(nn.blocks))

nn.custom_compile(optimizer="AdamW", rate=lr, loss_func="MSE", run_eagerly=False)

x_value = tf.linspace(
        tf.constant(lc, dtype=tf.float32), tf.constant(rc, dtype=tf.float32), num=5000
    )
t = tf.constant(1.0, shape=x_value.shape)
x = tf.concat([t, x_value], axis=1)
y = pde.solution(x)

tf.summary.trace_on(graph=True, profiler=False)
y_pred_before_train = nn.predict(x)
with summary_writer.as_default():
    tf.summary.trace_export(name="model_graph", step=0, profiler_outdir=log_dir)

y_pred_before_train = nn.predict(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

train_config = {
    "epochs": 8_000,
    "patience": 10_000,
    "eval_interval": 1,
    "batch_size": 10_000,
    "log_interval": 200,
    "mode": "full",
}
mlflow.log_params(train_config)

# nn.full_train(
nn.train(
    **train_config,
    callbacks=None,
    verbose=0,
    ode=pde,
    val_input=x_value,
    png_salt=str(run_id),
)
loss_after_train = nn.evaluate(x, y, verbose=0)
print("Before", loss_before_train)
print("After", loss_after_train)
mlflow.log_metric("Loss before training", loss_before_train)
mlflow.log_metric("Loss after training", loss_after_train)
mlflow.end_run()


for t_py in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    fig, axes = plt.subplots(nrows=2, ncols=1)
    t = tf.constant(t_py, shape=x_value.shape)
    x = tf.concat([t, x_value], axis=1)
    plot_each_submodel(x, x_value, y, nn, axes[0])
    plot_model(x, x_value, y, nn, axes[1])
    plt.savefig(
        f"FBPINN_{str(run_id)}_t{str(t_py)}.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
# plot_each_submodel(x, y, nn, axes[0])
# plot_model(x, y, nn, axes[1])
# plt.show()
