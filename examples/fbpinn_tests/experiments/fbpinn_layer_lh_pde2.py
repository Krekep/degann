from datetime import datetime
import random
import numpy as np
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

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
from degann.networks.topology.tffbpinn import LayerScheduler, LossScheduler, TensorflowFBPINN
from examples.fbpinn_tests.plot_functions import plot_each_submodel, plot_model
from examples.fbpinn_tests.PDEs.losses.LH_PDE2 import LH_PDE2


# Создаем расписание, зависящее от эпох
class EpochBasedScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, warmup_start, warmup_len, initial_rate, last_rate, steps, steps_per_epoch
    ):
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
            lr = (
                self.warmup_start
                + (self.initial_rate - self.warmup_start)
                / self.warmup_len
                * self.current_step
            )
        else:
            lr = (
                self.initial_rate
                - (self.initial_rate - self.last_rate)
                / (self.steps - self.warmup_len)
                * self.current_step
            )
            lr = tf.maximum(lr, self.last_rate)
        return lr


x_lim = 1.0
t_lim = 1.0
lc = [0, 0]
rc = [t_lim, x_lim]
pde = LH_PDE2()
phys_loss = pde.phys_loss
which = pde.description

mlflow.set_experiment("FBPINN PDE Heat 1D")
run_id = random.randint(1, 10000)
run_name = f"FBPINN_{run_id}"
mlflow.start_run(run_name=run_name)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("mlflow.runName", f"model{run_id}")
mlflow.set_tag("Class", pde.equation_class)


warmup_start = 0
warmup_len = 100
initial_rate = 1e-3
last_rate = 1e-5
steps = 50000
mlflow.log_param("warmup learning rate start", warmup_start)
mlflow.log_param("warmup steps", warmup_len)
mlflow.log_param("initial learning rate", initial_rate)
mlflow.log_param("initial learning rate", initial_rate)
mlflow.log_param("last learning rate", last_rate)
mlflow.log_param("scheduler steps", steps)
lr = EpochBasedScheduler(warmup_start, warmup_len, initial_rate, last_rate, steps, 1)
# lr = tf.keras.optimizers.schedules.ExponentialDecay(
#     1e-3, 5000, 0.9, staircase=True
# )
# lr = 1e-3

mlflow.log_param("left_bound", lc)
mlflow.log_param("right_bound", rc)

model_config = {
    "input_size": 2,
    "output_size": 1,
    "activation_func": ["tanh", "tanh", "linear"],
    "block_size": [0.6, 0.9],
    "models_size": [32, 32],
    "overlap": [0.15, 0.25],
    "offset": True,
    "points_per_block": 2000,
    "losses_weight": [10, 1, 1, 1]
}
mlflow.log_params(model_config)

nn = TensorflowFBPINN(
    **model_config,
    physic_loss=phys_loss,
    boundary_loss=pde.sub_losses,
    domain=RectangleDomain(lc.copy(), rc.copy()),
)

fb = nn.blocks[0][1]
fb.data = tf.linspace(
    tf.constant(lc, dtype=tf.float32),
    tf.constant(fb.right_up_corner, dtype=tf.float32),
    model_config["points_per_block"],
)
print("Number of submodels", len(nn.blocks))
print("Blocks per axis", nn.decomposition.blocks_per_axis)
print("Blocks per layer", sum(nn.decomposition.blocks_per_axis[1:]))

nn.custom_compile(optimizer="AdamW", rate=lr, loss_func="RelativeL1Loss", run_eagerly=False)

x = tf.linspace(
    tf.constant([t_lim / 2, 0], dtype=tf.float32),
    tf.constant([t_lim / 2, x_lim], dtype=tf.float32),
    num=10000,
)
y = pde.solution(x)

y_pred_before_train = nn.predict(x)

y_pred_before_train = nn.predict(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

layer_scheduler = LayerScheduler(
    n=len(nn.blocks),
    left_bound_step=sum(nn.decomposition.blocks_per_axis[1:]),
    right_bound_step=sum(nn.decomposition.blocks_per_axis[1:]),
    left_bound_schedule=25000,
    right_bound_schedule=10000,
    start_left_bound=0,
    start_right_bound=sum(nn.decomposition.blocks_per_axis[1:])
)
loss_scheduler = LossScheduler(
    k=len(pde.sub_losses) + 1,
    boundary_indices=list(range(len(pde.sub_losses)))
)
train_config = {
    "epochs": 20,
    "patience": 500_000,
    "eval_interval": 400,
    "batch_size": 10_000,
    "log_interval": 10000,
    "mode": "layer",
    "layer_scheduler": layer_scheduler,
    "loss_scheduler": loss_scheduler,
}
mlflow.log_params(train_config)

nn.train(
    **train_config,
    callbacks=None,
    verbose=0,
    ode=pde,
    val_input=x,
    png_salt=str(run_id),
)
loss_after_train = nn.evaluate(x, y, verbose=0)
print("Before", loss_before_train)
print("After", loss_after_train)
mlflow.log_metric("Loss before training", loss_before_train)
mlflow.log_metric("Loss after training", loss_after_train)
mlflow.end_run()
nn.save_weights(f"fbpinn{run_id}.weights.h5")

x_plot = tf.linspace(
    tf.constant(lc[1], dtype=tf.float32),
    tf.constant(rc[1], dtype=tf.float32),
    num=10000,
)
for t_py in [i / 10 for i in range(0, int(10 * t_lim + 1))]:
    fig, axes = plt.subplots(nrows=2, ncols=1)
    t = tf.constant(t_py, shape=x_plot.shape)
    x = tf.stack([t, x_plot], axis=1)
    y = pde.solution(x)
    plot_each_submodel(x, x_plot, y, nn, axes[0])
    plot_model(x, x_plot, y, nn, axes[1])
    plt.savefig(f"FBPINN_{str(run_id)}_t{str(t_py)}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
