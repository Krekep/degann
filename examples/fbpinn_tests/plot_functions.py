from matplotlib import pyplot as plt
import tensorflow as tf


def plot_each_submodel(model_input, x, y, nn, axes):
    axes.plot(x, y, label="Real", color="orange")
    x_plot = tf.convert_to_tensor(model_input, dtype=tf.float32)
    for i, (submodel, block) in enumerate(nn.blocks):
        x_norm = block.normalization(x_plot)
        predicted = submodel(x_norm)
        predicted_unnorm: tf.Tensor = block.unnormalization(predicted)
        windowed = block.window_function(x_plot)
        result = windowed * predicted_unnorm
        y_submodel = result
        axes.plot(x, y_submodel)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.legend()
    axes.grid()
    axes.set_title("Actual vs Predicted Data")
    # plt.show()


def plot_model(model_input, x, y, nn, axes):
    y_pred = nn(model_input)
    axes.plot(x, y, label="Real", color="orange")
    axes.plot(x, y_pred, label="Predicted", color="green")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.legend()
    axes.grid()
