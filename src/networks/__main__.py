from src.networks import activations
from src.networks.imodel import IModel
from src.networks.topology.densenet import DenseNet
from tests.utils import init_params

act_func, weight_initializer1, bias_initializer = init_params(
    act_name="linear", weight_name="ones", bias_name="zeros"
)
weight_initializer2 = init_params(weight_name="zeros")[0]
prec_act = activations.perceptron_threshold

# layer = ILayer(inp_size=1, shape=1, activation=act_func, weight=weight_initializer1, bias=bias_initializer)
# layer1 = MyDense(input_dim=3, units=1, activation_funcs=act_func, weight_initializer=weight_initializer1, bias_initializer=bias_initializer)
#
# print(layer.get_config())
# print(layer.get_weights())

nn = IModel.create_neuron(
    3,
    2,
    [2, 2],
    activation=prec_act(1),
    weight=weight_initializer1,
    biases=bias_initializer,
    activation_names=["perceptron_threshold"] * 3,
    decorator_params=[1] * 3,
)

nn1 = IModel.create_neuron(
    1,
    1,
    [1],
    activation=prec_act(2),
    weight=weight_initializer2,
    biases=bias_initializer,
    activation_names=["perceptron_threshold"],
    decorator_params=[2],
)

densenet = DenseNet(input_size=1, block_size=[1], output_size=1)

# nn.export_to_cpp("nn")
# print(densenet.get_config())

nn.export_to_file("nn.txt")
nn1.export_to_file("nn1_before.txt")

nn1.from_file("nn.txt")

nn1.export_to_file("nn1_after.txt")

# print("**** KERAS OBJECTS ****")
#
# kl = keras.layers.Dense(4)
# print(kl.get_config())
# print(kl.get_weights())
#
# k = keras.Sequential()
# k.add(keras.layers.Dense(3, input_shape=(4,)))
# # Afterwards, we do automatic shape inference:
# k.add(layer1)
# print(k.get_config())
# print(k.get_weights())
#
# # k.save_weights("test.h5")
#
# k2 = keras.Sequential()
# k2.add(keras.layers.Dense(3, input_shape=(4,)))
# # Afterwards, we do automatic shape inference:
# k2.add(keras.layers.Dense(1, input_shape=(3,)))
# print(k2.get_config())
# print(k2.get_weights())
# k2.load_weights("test.h5")
# print(k2.get_weights())
