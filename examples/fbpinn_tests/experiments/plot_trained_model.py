import mlflow
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


from examples.fbpinn_tests.plot_functions import plot_each_submodel, plot_model
from degann.networks.layers.tf_dense import TensorflowDense
from degann.networks.topology.pinn import PhysicsInformedNet
from degann.networks.topology.tf_densenet import TensorflowDenseNet
from degann.geometry import RectangleDomain
from degann.networks.topology.tffbpinn import (
    LayerScheduler,
    LossScheduler,
    TensorflowFBPINN,
)
from examples.fbpinn_tests.experiments.Functions.LH_PDE2 import LH_PDE2


x_lim = 1.0
t_lim = 1.0
lc = [0, 0]
rc = [t_lim, x_lim]
pde = LH_PDE2()
phys_loss = pde.phys_loss
which = pde.description
lr = 1e-3
eq = "lh_pde_2"
run_id = "dep_full_1087"
run_id_np = "dep_full_5810"

model_config = {
    "input_size": 2,
    "output_size": 1,
    "activation_func": ["tanh", "tanh", "linear"],
    "block_size": [0.6, 0.9],
    "models_size": [32, 32],
    "overlap": [0.15, 0.25],
    "offset": True,
    "points_per_block": 2000,
    "losses_weight": [10, 1, 1, 1],
}
mlflow.log_params(model_config)


loaded_model = TensorflowFBPINN(
    **model_config,
    physic_loss=phys_loss,
    boundary_loss=pde.sub_losses,
    domain=RectangleDomain(lc.copy(), rc.copy()),
)
loaded_model.custom_compile(
    optimizer="AdamW", rate=lr, loss_func="MSE", run_eagerly=False
)
loaded_model.build((None, 2))
loaded_model.load_weights(f"{eq}_{run_id}.weights.h5")

loaded_model_nonpretrained = TensorflowFBPINN(
    **model_config,
    physic_loss=phys_loss,
    boundary_loss=pde.sub_losses,
    domain=RectangleDomain(lc.copy(), rc.copy()),
)
loaded_model_nonpretrained.custom_compile(
    optimizer="AdamW", rate=lr, loss_func="MSE", run_eagerly=False
)
loaded_model_nonpretrained.build((None, 2))
loaded_model_nonpretrained.load_weights(f"{eq}_{run_id_np}_non_pretrained.weights.h5")

x = tf.linspace(
    tf.constant([t_lim / 2, 0], dtype=tf.float32),
    tf.constant([t_lim / 2, x_lim], dtype=tf.float32),
    num=10000,
)
y = pde.solution(x)

y_pred_before_train = loaded_model.predict(x)
loss_after_train = loaded_model.evaluate(x, y, verbose=0)

x_plot = tf.linspace(
    tf.constant(lc[1], dtype=tf.float32),
    tf.constant(rc[1], dtype=tf.float32),
    num=10000,
)
fig, axes = plt.subplots(nrows=2, ncols=2)
for i, t_py in enumerate([0.0, 0.1, 0.2, 0.3]):
    t = tf.constant(t_py, shape=x_plot.shape)
    x = tf.stack([t, x_plot], axis=1)
    y = pde.solution(x)
    y_model = loaded_model(x)
    y_model_np = loaded_model_nonpretrained(x)

    r = i // 2
    c = i % 2
    axes[r, c].plot(x_plot, y, label="Truth", color="red")
    axes[r, c].plot(x_plot, y_model_np, label="Non pretrained", color="blue")
    axes[r, c].plot(x_plot, y_model, label="Pretrained", color="green")
    axes[r, c].grid(visible=True, which="major")
    axes[r, c].grid(visible=True, which="minor")
    axes[r, c].set_title(f"Time {t_py}")
    axes[r, c].legend()
plt.show()
