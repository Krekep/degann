import random
from typing import Union, Any, Tuple, List

import numpy as np
import tensorflow as tf

from degann.networks.callbacks import MemoryCleaner
from degann.networks import get_all_optimizers, get_all_metric_functions
from degann.networks import activations, imodel, losses

_default_shapes = [
    [10, 10, 10, 10, 10, 10],
    [80, 80, 80],
    [32, 16, 8, 4],
    [4, 8, 16, 32],
    [10, 10],
    [30],
]


def _create_random_network(
    min_layers=1,
    max_layers=5,
    min_neurons=3,
    max_neurons=60,
) -> list[Union[list[int], list, list[Union[str, Any]], list[None]]]:
    """
    Create random neural network from the passed parameters.

    Parameters
    ----------
    min_layers: int
        Minimal count of layers in neural net
    max_layers: int
        Maximal count of layers in neural net
    min_neurons: int
        Minimal count of neurons per layer
    max_neurons: int
        Maximal count of neurons per layer
    Returns
    -------
    net: network.INetwork
        Random neural network
    """

    layers = random.randint(min_layers, max_layers)
    shape = [random.randint(min_neurons, max_neurons) for _ in range(layers)]
    act = []
    decorator_param = []
    all_act_names = list(activations.get_all_activations().keys())
    for _ in shape:
        act.append(random.choice(all_act_names))
        # TODO: activation func can take additional arguments
        # but at this moment I dont create random arguments (instead of *None* in decorator_params)
        decorator_param.append(None)
    act.append("linear")
    decorator_param.append(None)

    nets_param = [shape, act, decorator_param]
    return nets_param


def _normalize_array(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Scale array from [a, b] to [0, 1]

    Parameters
    ----------
    x: np.ndarray
        Array for scaling

    Returns
    -------
    x: Tuple[np.ndarray, float]
        Scaled array and scale coefficient
    """
    m = abs(np.amax(x))
    if m != 0:
        x = x / m
    return x, m


def train(
    x_data: np.ndarray,
    y_data: np.ndarray,
    val_data: tuple[np.ndarray, np.ndarray] = None,
    **kwargs,
) -> tuple[imodel.IModel, dict[Union[str, Any], Any]]:
    """
    Choose and return neural network which present the minimal average absolute deviation.
    x_data and y_data is numpy 2d arrays (in case we don't have multiple-layer input/output).

    Parameters
    ----------
    x_data: np.ndarray
        Array of inputs --- [input1, input2, ...]
    y_data: np.ndarray
        List of outputs --- [output1, output2, ...]
    val_data: np.ndarray
        Validation dataset

    Returns
    -------
    net: imodel.IModel
        Best neural network for this dataset
    history: Dict[str, list]
        History of training for this network
    """
    # default config
    args = {
        "debug": False,
        "eps": 1e-2,
        "epochs": 100,
        "normalize": False,
        "name_salt": "",
        "loss_function": "MeanSquaredError",
        "optimizer": "SGD",
        "metrics": [
            "MeanSquaredError",
            "MeanAbsoluteError",
            "CosineSimilarity",
        ],
        "validation_metrics": [
            "MeanSquaredError",
            "MeanAbsoluteError",
            "CosineSimilarity",
        ],
        "use_rand_net": True,
        "net_type": "DenseNet",
        "nets_param": [
            [
                shape,  # shape
                ["sigmoid"] * len(shape) + ["linear"],  # activation functions
                [None] * (len(shape) + 1),  # decorator parameters for activation
            ]
            for shape in _default_shapes
        ],
    }
    for kw in kwargs:
        args[kw] = kwargs[kw]

    if args["debug"]:
        print("Start train func")

    # determining the number of inputs and outputs of the neural network
    if type(x_data[0]) is np.ndarray:
        input_len = len(x_data[0])
    else:
        input_len = 1

    if type(y_data[0]) is np.ndarray:
        output_len = len(y_data[0])
    else:
        output_len = 1

    # prepare data (normalize)
    norm_coff = 1
    if args["normalize"]:
        x_data, norm_coff = _normalize_array(x_data)
        # y_data, norm_coff = _normalize_array(y_data)
        if args["debug"]:
            print(f"Normalization coefficient is {norm_coff}")

    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)

    # prepare neural networks
    if args["debug"]:
        print("Prepare neural networks and data")
    nets = []
    for parameters in args["nets_param"]:
        shape: list[int] = parameters[0]
        act = parameters[1]
        decorator_param = parameters[2]
        str_shape = "_".join(map(str, shape))
        curr_net = imodel.IModel(
            input_size=input_len,
            block_size=shape,
            output_size=output_len,
            activation_func=act,
            decorator_params=decorator_param,
            net_type=args["net_type"],
            name=f"net{args['name_salt']}_{str_shape}",
            is_debug=args["debug"],
        )
        nets.append(curr_net)
    if args["use_rand_net"]:
        rand_net_params = _create_random_network(input_len, output_len)
        str_shape = "_".join(map(str, rand_net_params[0]))
        rand_net = imodel.IModel(
            input_size=input_len,
            block_size=rand_net_params[0],
            output_size=output_len,
            activation_func=rand_net_params[1],
            decorator_params=rand_net_params[2],
            net_type=args["net_type"],
            name=f"net{args['name_salt']}_{str_shape}",
        )
        nets.append(rand_net)

    # compile
    for nn in nets:
        nn.compile(
            rate=args["eps"],
            optimizer=args["optimizer"],
            loss_func=args["loss_function"],
            metrics=args["metrics"],
            # run_eagerly=True,
        )

    if args["debug"]:
        print("Success prepared")

    # train
    history = []
    for i, nn in enumerate(nets):
        verb = 0
        if args["debug"]:
            print(nn)
            verb = 1
        temp_his = nn.train(
            x_data,
            y_data,
            epochs=args["epochs"],
            validation_data=val_data,
            callbacks=[MemoryCleaner()],
            verbose=verb,
        )
        temp_last_res = dict()
        for key in temp_his.history:
            temp_last_res[key] = temp_his.history[key].copy()

        history.append(temp_last_res)
    result_net = nets[0]
    result_history = history[0]
    min_err = history[0]["loss"]
    for i in range(1, len(nets)):
        if history[i]["loss"] < min_err:
            min_err = history[i]["loss"]
            result_net = nets[i]
            result_history = history[i]
    if args["debug"]:
        print(f"Minimal loss error is {min_err} {args['name_salt']}")
    return result_net, result_history


def pattern_search(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_val: np.ndarray = None,
    y_val: np.ndarray = None,
    **kwargs,
) -> list[list[dict, float, float, imodel.IModel]]:
    """
    Choose and return neural network which present the minimal average absolute deviation.
    x_data and y_data is numpy 2d arrays (in case we don't have multiple-layer input/output).

    Parameters
    ----------
    x_data: np.ndarray
        Array of inputs --- [input1, input2, ...]
    y_data: np.ndarray
        List of outputs --- [output1, output2, ...]
    x_val: np.ndarray
        Array of inputs for validate train
    y_val: np.ndarray
        Array of outputs for validate train
    Returns
    -------
    net: network.INetwork
        Best neural network for this dataset
    """

    # default config
    args = {
        "loss_functions": [key for key in losses.get_all_loss_functions()],
        "optimizers": [key for key in get_all_optimizers()],
        "metrics": [key for key in get_all_metric_functions()],
        "validation_metrics": [key for key in get_all_metric_functions()],
        "net_shapes": _default_shapes,
        "activations": [key for key in activations.get_all_activations()],
        "rates": [1e-2, 5e-3, 1e-3],
        "epochs": [50, 200],
        "normalize": [False],
    }

    for arg in args:
        if kwargs.get(arg) is not None:
            args[arg] = kwargs[arg]
            kwargs.pop(arg)

    val_data = None
    if x_val is not None and y_val is not None:
        val_data = (x_val, y_val)

    # Networks parameters --- shape and activation functions
    nets_param = []
    for shape in args["net_shapes"]:
        if len(shape) != 0:
            for activation in args["activations"]:
                nets_param.append(
                    [
                        shape,
                        [activation] * len(shape) + ["linear"],
                        [None] * (len(shape) + 1),
                    ]
                )
        else:
            nets_param.append(
                [
                    shape,
                    ["linear"],
                    [None],
                ]
            )

    metaparams = []
    for loss_func in args["loss_functions"]:
        for normalize in args["normalize"]:
            for optimizer in args["optimizers"]:
                for epochs in args["epochs"]:
                    for rate in args["rates"]:
                        metaparams.append(dict())
                        metaparams[-1]["loss_function"] = loss_func
                        metaparams[-1]["normalize"] = normalize
                        metaparams[-1]["optimizer"] = optimizer
                        metaparams[-1]["epochs"] = epochs
                        metaparams[-1]["eps"] = rate
                        metaparams[-1]["metrics"] = args["metrics"]
                        metaparams[-1]["validation_metrics"] = args[
                            "validation_metrics"
                        ]
                        metaparams[-1]["nets_param"] = nets_param
                        metaparams[-1].update(kwargs)

    best_nets: List[List[dict, float, float, imodel.IModel]] = []
    if kwargs.get("debug"):
        print(f"Grid search size {len(metaparams)}")
        print("Amount of networks for each set of parameters", len(nets_param))
    for i, params in enumerate(metaparams):
        if kwargs.get("debug"):
            print(f"Number of set {i}")
        trained, history = train(x_data, y_data, val_data=val_data, **params)
        loss = history["loss"][-1]
        if val_data is not None:
            val_loss = history["val_loss"][-1]
        else:
            val_loss = 10**9
        best_nets.append([params, loss, val_loss, trained])

    best_nets.sort(key=lambda x: [x[1], x[2]])
    return best_nets
