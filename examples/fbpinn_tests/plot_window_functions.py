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

from degann.geometry import RectangleDomain
from degann.geometry.decomposition import Decomposition

original_function = lambda x: tf.sin(1 * x)

lc = 0
rc = np.pi * 4

domain = RectangleDomain([lc], [rc])
block_size = 0.74
overlap = 0.3
offset = True
points_per_block = 100
decomposition = Decomposition(
    domain=domain,
    block_size=block_size,
    overlap=overlap,
    offset=offset,
    points_per_block=points_per_block,
)

temp = []
for block in decomposition.blocks:
    block.data = tf.reshape(
        tf.convert_to_tensor(block.get_data(), dtype=tf.float32), shape=(-1, 1)
    )
    temp.append(block.data)
x = tf.concat(temp, axis=0)
x = tf.sort(x, axis=0)
y = 0

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
for i, block in enumerate(decomposition.blocks):
    print(f"Block {i}")
    w = block.window_function(x)
    y += w
    axes[0].plot(x, w)
print("End loop")
axes[1].plot(x, y)
print("Plot shit")
axes[0].grid()
axes[1].grid()
print("Made grid")
plt.show()
print("End of program")
