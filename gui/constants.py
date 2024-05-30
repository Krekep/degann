from PyQt6 import QtWidgets, QtGui

minimum_police_size = QtWidgets.QSizePolicy(
    QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
)
minimum_police_size.setHorizontalStretch(0)
minimum_police_size.setVerticalStretch(0)
minimum_police_size.setHeightForWidth(True)

expand_minimum_police_size = QtWidgets.QSizePolicy(
    QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
)
expand_minimum_police_size.setHorizontalStretch(0)
expand_minimum_police_size.setVerticalStretch(0)
expand_minimum_police_size.setHeightForWidth(True)

expand_fixed_police_size = QtWidgets.QSizePolicy(
    QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
)
expand_minimum_police_size.setHorizontalStretch(0)
expand_minimum_police_size.setVerticalStretch(0)
expand_minimum_police_size.setHeightForWidth(True)

fixed_police_size = QtWidgets.QSizePolicy(
    QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
)
fixed_police_size.setHorizontalStretch(0)
fixed_police_size.setVerticalStretch(0)
fixed_police_size.setHeightForWidth(True)

font_12pt = QtGui.QFont()
font_12pt.setPointSize(12)

font_6pt = QtGui.QFont()
font_6pt.setPointSize(6)

param_desc_phrases = [
    # "Number of random search runs",
    # "Number of runs of simulated annealing method",
    # "Maximum neural network length",
    # "Minimal neural network length",
    # "Block size in a neural network",
    # "Offset in neural network blocks",
    # "Alphabet of a neural network",
    # "Minimum number of training epochs",
    # "Maximum number of training epochs",
    # "Number of iterations in search algorithms",
    "Loss function: ",
    "Loss function threshold: ",
    "Optimizer: ",
]
ann_param_desc_phrases = [
    "Simulated Annealing Method Parameters",
    "Method for determining the distance to a neighbor",
    "Temperature decrease method",
]
parameter_value_gui_names = [
    "launch_random_count_text",
    "launch_ann_count_text",
    "max_len_text",
    "min_len_text",
    "block_size_text",
    "offset_text",
    "alphabet_text",
    "min_epoch_text",
    "max_epoch_text",
    "iteration_count_text",
    # "loss_func_combobox",
    "loss_threshold_text",
    # "optimizer_combobox",
]
parameter_value_code_names = [
    "launch_count_random_search",
    "launch_count_simulated_annealing",
    "nn_max_length",
    "nn_min_length",
    "nn_alphabet_block_size",
    "nn_alphabet_offset",
    "nn_alphabet",
    "min_train_epoch",
    "max_train_epoch",
    "iteration_count",
    # "loss_function",
    "loss_threshold",
    # "optimizer",
]
ann_param_gui_names = [
    # "ann_dist_comboBox",
    # "ann_temp_comboBox",
    "dist_offset",
    "dist_scale",
    "temp_speed",
]
ann_param_code_names = [
    # "distance_to_neighbor",
    "dist_offset",
    "dist_scale",
    # "temperature_reduction_method",
    "temperature_speed",
]
