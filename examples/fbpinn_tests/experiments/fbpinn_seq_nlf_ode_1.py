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
from degann.networks.topology.tffbpinn import LayerScheduler, TensorflowFBPINN
from tensorflow.keras.callbacks import EarlyStopping
from examples.fbpinn_tests.plot_functions import plot_each_submodel, plot_model
from Functions.NLF_ODE_1_submodels import NLF_ODE_1


ode = NLF_ODE_1()
phys_loss = ode.phys_loss
which = ode.description

mlflow.set_experiment("FBPINN NLF_ODE_1")
run_id = random.randint(1, 10000)
run_name = f"FBPINN_Seq_{run_id}"
mlflow.start_run(run_name=run_name)
mlflow.set_tag("Training Info", f"FBPINN model for {which}")
mlflow.set_tag("mlflow.runName", run_name)
log_dir = f"logs/fit/model{run_id}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

lr = 1e-3
mlflow.log_param("Learning rate", lr)

lc = 0
rc = 1
mlflow.log_param("left_bound", lc)
mlflow.log_param("right_bound", rc)

model_config = {
    "input_size": 1,
    "output_size": 1,
    "activation_func": ["tanh", "tanh", "linear"],
    "block_size": [0.72],
    "models_size": [16, 16],
    "overlap": [0.3],
    "offset": False,
    "points_per_block": 1500,
}
mlflow.log_params(model_config)

nn = TensorflowFBPINN(
    **model_config,
    physic_loss=phys_loss,
    boundary_loss=ode.sub_losses,
    domain=RectangleDomain([lc], [rc]),
)

fb = nn.blocks[0][1]
fb.data = tf.linspace(
    tf.constant(lc, dtype=tf.float32),
    tf.constant(fb.right_up_corner, dtype=tf.float32),
    model_config["points_per_block"],
)
print("Number of submodels", len(nn.blocks))
mlflow.log_param("Number of submodels", len(nn.blocks))

nn.custom_compile(optimizer="AdamW", rate=lr, loss_func="MSE", run_eagerly=False)

x = tf.reshape(
    tf.linspace(
        tf.constant(lc, dtype=tf.float32), tf.constant(rc, dtype=tf.float32), num=20000
    ),
    (-1, 1),
)
y = ode.solution(x)

y_pred_before_train = nn.predict(x)
loss_before_train = nn.evaluate(x, y, verbose=0)

train_config = {
    "epochs": 100_000,
    "patience": 200_000,
    "eval_interval": 10,
    "batch_size": 10_000,
    "log_interval": 10000,
    "mode": "sequence",
}
mlflow.log_params(train_config)

# nn.full_train(
nn.train(
    **train_config,
    callbacks=None,
    verbose=0,
    ode=ode,
    val_input=x,
    png_salt=str(run_id),
)
loss_after_train = nn.evaluate(x, y, verbose=0)
print("Before", loss_before_train)
print("After", loss_after_train)
mlflow.log_metric("Loss before training", loss_before_train)
mlflow.log_metric("Loss after training", loss_after_train)
mlflow.end_run()
nn.save_weights(f"nlf_ode_1_seq_{run_id}.weights.h5")

fig, axes = plt.subplots(nrows=2, ncols=1)
plot_each_submodel(x, x, y, nn, axes[0])
plot_model(x, x, y, nn, axes[1])
plt.show()
