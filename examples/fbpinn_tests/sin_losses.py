import tensorflow as tf
from degann.geometry.decomposition import Block
from degann.networks.topology.tffbpinn3 import TensorflowFBPINN


def physic_loss(
    model: TensorflowFBPINN,
    tape: tf.GradientTape,
    x,
    block: Block,
    prev_model: TensorflowFBPINN,
    prev_block: Block,
    make_shit: bool = True,
    **kwargs
):
    """solution function for y'' + 100y = 0, y(0) = 0, y'(0) = 10"""
    if make_shit:
        x_norm = block.normalization(x)
        predicted = model(x_norm)
        predicted_unnorm: tf.Tensor = block.unnormalization(predicted)

        windowed = block.window_function(x)
        if prev_model is not None:
            x_left_norm = prev_block.normalization(x)
            predicted_left = prev_model(x_left_norm)
            predicted_unnorm_left: tf.Tensor = block.unnormalization(predicted_left)
            windowed_left = prev_block.window_function(x)

            u = windowed * predicted_unnorm + windowed_left * predicted_unnorm_left
        else:
            u = windowed * predicted_unnorm
    else:
        u = model(x)

    u_x = tape.gradient(u, x)
    u_xx = tape.gradient(u_x, x)
    u_model = u_xx + 100 * u
    u_true = tf.zeros_like(u_model)
    phys_loss = tf.reduce_mean(tf.square(u_true - u_model))

    return phys_loss


def boundary_loss_1(
    model: TensorflowFBPINN,
    tape: tf.GradientTape,
    x,
    block: Block,
    prev_model: TensorflowFBPINN,
    prev_block: Block,
    make_shit: bool = True,
    **kwargs
):
    """solution function for y'' + 100y = 0, y(0) = 0, y'(0) = 10"""
    ic_x = tf.constant([[0.0]])
    tape.watch(ic_x)

    if make_shit:
        x_norm = block.normalization(ic_x)
        predicted = model(x_norm)
        predicted_unnorm: tf.Tensor = block.unnormalization(predicted)
        windowed = block.window_function(ic_x)
        u = windowed * predicted_unnorm
    else:
        u = model(ic_x)

    u_ic_true = tf.zeros_like(ic_x)
    ic_loss = tf.reduce_mean(tf.square(u_ic_true - u))

    return ic_loss


def boundary_loss_2(
    model: TensorflowFBPINN,
    tape: tf.GradientTape,
    x,
    block,
    prev_model: TensorflowFBPINN,
    prev_block: Block,
    make_shit: bool = True,
    **kwargs
):
    """second initial condition for y'' + 100y = 0, y(0) = 0, y'(0) = 10"""
    ic_x = tf.constant([[0.0]])
    tape.watch(ic_x)

    if make_shit:
        x_norm = block.normalization(ic_x)
        predicted = model(x_norm)
        predicted_unnorm: tf.Tensor = block.unnormalization(predicted)
        windowed = block.window_function(ic_x)
        u = windowed * predicted_unnorm
    else:
        u = model(ic_x)

    u_x = tape.gradient(u, ic_x)
    u_true = tf.constant([[10.0]])
    ic_loss = tf.reduce_mean(tf.square(u_true - u_x))

    return ic_loss
