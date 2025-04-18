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
from sin_losses.NLF_ODE_2 import NLF_ODE_2


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
                - (self.initial_rate - self.last_rate) / self.steps * self.current_step
            )
            lr = tf.maximum(lr, self.last_rate)
        return lr

ode = NLF_ODE_2()
phys_loss = ode.phys_loss
which = ode.description

mlflow.set_experiment("FBPINN Sin ODE")
run_id = random.randint(1, 10000)
run_name = f"FBPINN_{run_id}"
mlflow.start_run(run_name=run_name)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("mlflow.runName", f"model{run_id}")
log_dir = f"logs/fit/model{run_id}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
summary_writer = tf.summary.create_file_writer(log_dir)


warmup_start = 0
warmup_len = 1000
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
    "points_per_block": 1000,
}
mlflow.log_params(model_config)

nn = TensorflowFBPINN(
    **model_config,
    physic_loss=phys_loss,
    boundary_loss=ode.sub_losses,
    domain=RectangleDomain([lc], [rc]),
    summary_writer=summary_writer,
)

fb = nn.blocks[0][1]
fb.data = tf.linspace(
    tf.constant(lc, dtype=tf.float32),
    tf.constant(fb.right_up_corner, dtype=tf.float32),
    model_config["points_per_block"],
)
print("Number of submodels", len(nn.blocks))

nn.custom_compile(optimizer="AdamW", rate=lr, loss_func="MSE", run_eagerly=False)

x = tf.reshape(
    tf.linspace(
        tf.constant(lc, dtype=tf.float32), tf.constant(rc, dtype=tf.float32), num=5000
    ),
    (-1, 1),
)
y = ode.solution(x)

# tf.summary.trace_on(graph=True, profiler=False)
y_pred_before_train = nn.predict(x)
# with summary_writer.as_default():
#     tf.summary.trace_export(name="model_graph", step=0, profiler_outdir=log_dir)

y_pred_before_train = nn.predict(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

train_config = {
    "epochs": 100_000,
    "patience": 3000,
    "eval_interval": 1,
    "batch_size": 10_000_000,
    "log_interval": 10000,
    "mode": "full",
}
mlflow.log_params(train_config)

# noise = tf.constant(0, shape=nn.data.shape, dtype=tf.float32)
noise_py = [0.0] * len(nn.data)
indices = [200, 15000, 29800]
for idx in indices:
    noise_py[idx] = ode.solution(nn.data[idx]).cpu().numpy().item() * 0.2
with tf.device('/GPU:0'):
    noise = tf.constant(noise_py, shape=nn.data.shape, dtype=tf.float32)
nn.train(
    **train_config,
    callbacks=None,
    verbose=0,
    ode=ode,
    val_input=x,
    png_salt=str(run_id),
    noise=noise
)
loss_after_train = nn.evaluate(x, y, verbose=0)
print("Before", loss_before_train)
print("After", loss_after_train)
mlflow.log_metric("Loss before training", loss_before_train)
mlflow.log_metric("Loss after training", loss_after_train)
mlflow.end_run()

fig, axes = plt.subplots(nrows=2, ncols=1)
# plot_each_submodel(x, x, y, nn, axes[0])
plot_model(x, x, y, nn, axes[0])
x_model = tf.sort(nn.data, axis=0)
y_model_true = ode.solution(x_model)
y_noise = y_model_true + noise
axes[1].plot(x_model, y_noise, label="Real data + noise", color="red")
y_pred = nn(x_model)
axes[1].plot(x_model, y_pred, label="Predicted", color="green")
axes[1].scatter(tf.gather(x_model, indices=indices), tf.gather(y_noise, indices=indices), label="Noise offset", color="blue")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].legend()
axes[1].grid()
plt.savefig(
    f"FBPINN_{str(run_id)}.png", dpi=300, bbox_inches="tight"
)
plt.show()
