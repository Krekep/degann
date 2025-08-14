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
from degann.networks.topology.layer_scheduler import SequenceLayerScheduler
from tensorflow.keras.callbacks import EarlyStopping
from examples.fbpinn_tests.plot_functions import plot_each_submodel, plot_model
from Functions.LH_PDE1_submodels import LH_PDE1


pde = LH_PDE1()
phys_loss = pde.phys_loss
which = pde.description

mlflow.set_experiment("FBPINN PDE Wave 1D")
run_id = random.randint(1, 10000)
run_name = f"FBPINN_All_{run_id}"
mlflow.start_run(run_name=run_name)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("mlflow.runName", run_name)
log_dir = f"logs/fit/model{run_id}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


lr = 1e-3
mlflow.log_param("Learning rate", lr)

x_lim = 1.0
t_lim = 1.0
lc = [0, 0]
rc = [t_lim, x_lim]
mlflow.log_param("left_bound", lc)
mlflow.log_param("right_bound", rc)

model_config = {
    "input_size": 2,
    "output_size": 1,
    "activation_func": ["tanh", "tanh", "tanh", "linear"],
    "block_size": [0.3, 0.6],
    "models_size": [32, 32, 32],
    "overlap": [0.1, 0.2],
    "offset": True,
    "points_per_block": 1000,
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

nn.custom_compile(optimizer="AdamW", rate=lr, loss_func="MSE", run_eagerly=False)

x = tf.linspace(
    tf.constant([t_lim / 2, 0], dtype=tf.float32),
    tf.constant([t_lim / 2, x_lim], dtype=tf.float32),
    num=10000,
)
y = pde.solution(x)

y_pred_before_train = nn.predict(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

train_config = {
    "epochs": 100000,
    "patience": 2000_000,
    "eval_interval": 100,
    "batch_size": 10_000,
    "log_interval": 10000,
    "mode": "all",
}
mlflow.log_params(train_config)

# nn.full_train(
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
nn.save_weights(f"lh_pde_1_all_{run_id}.weights.h5")

x_plot = tf.linspace(
    tf.constant(lc[1], dtype=tf.float32),
    tf.constant(rc[1], dtype=tf.float32),
    num=10000,
)
for t_py in [i / 20 for i in range(0, int(20 * t_lim + 1))]:
    fig, axes = plt.subplots(nrows=2, ncols=1)
    t = tf.constant(t_py, shape=x_plot.shape)
    x = tf.stack([t, x_plot], axis=1)
    y = pde.solution(x)
    plot_each_submodel(x, x_plot, y, nn, axes[0])
    plot_model(x, x_plot, y, nn, axes[1])
    plt.savefig(f"FBPINN_{str(run_id)}_t{str(t_py)}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
