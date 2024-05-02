import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtWidgets import QMainWindow

import degann.networks
from degann import IModel
from degann.expert import selector

_minimum_police_size = QtWidgets.QSizePolicy(
    QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
)
_minimum_police_size.setHorizontalStretch(0)
_minimum_police_size.setVerticalStretch(0)
_minimum_police_size.setHeightForWidth(True)

_expand_minimum_police_size = QtWidgets.QSizePolicy(
    QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
)
_expand_minimum_police_size.setHorizontalStretch(0)
_expand_minimum_police_size.setVerticalStretch(0)
_expand_minimum_police_size.setHeightForWidth(True)

_fixed_police_size = QtWidgets.QSizePolicy(
    QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
)
_fixed_police_size.setHorizontalStretch(0)
_fixed_police_size.setVerticalStretch(0)
_fixed_police_size.setHeightForWidth(True)

_font_12pt = QtGui.QFont()
_font_12pt.setPointSize(12)

_font_6pt = QtGui.QFont()
_font_6pt.setPointSize(6)

_param_desc_phrases = [
    "Number of random search runs",
    "Number of runs of simulated annealing method",
    "Maximum neural network length",
    "Minimal neural network length",
    "Block size in a neural network",
    "Offset in neural network blocks",
    "Alphabet of a neural network",
    "Minimum number of training epochs",
    "Maximum number of training epochs",
    "Number of iterations in search algorithms",
    "Loss function",
    "Loss function threshold",
    "Optimizer",
]
_ann_param_desc_phrases = [
    "Simulated Annealing Method Parameters",
    "Method for determining the distance to a neighbor",
    "Temperature decrease method",
]
_parameter_value_gui_names = [
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
_parameter_value_code_names = [
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
_ann_param_gui_names = [
    # "ann_dist_comboBox",
    # "ann_temp_comboBox",
    "dist_offset",
    "dist_scale",
    "temp_speed",
]
_ann_param_code_names = [
    # "distance_to_neighbor",
    "dist_offset",
    "dist_scale",
    # "temperature_reduction_method",
    "temperature_speed",
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setObjectName("MainWindow")
        self.resize(1000, 600)
        self.centralwidget = QtWidgets.QWidget(parent=self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.main_area = QtWidgets.QVBoxLayout()
        self.main_area.setObjectName("main_area")

        dataset_area = self.setup_dataset_area()
        self.main_area.addLayout(dataset_area)

        self.metaparameters_area = QtWidgets.QHBoxLayout()
        self.metaparameters_area.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetMinimumSize
        )
        self.metaparameters_area.setObjectName("metaparameters_area")

        self.train_area = QtWidgets.QVBoxLayout()
        self.train_area.setObjectName("train_area")

        self.parameters_area = QtWidgets.QHBoxLayout()
        self.parameters_area.setObjectName("parameters_area")

        param_desc_area = self.setup_parameters_desc_area()
        self.parameters_area.addLayout(param_desc_area)

        param_value_area = self.setup_parameters_value_area()
        self.parameters_area.addLayout(param_value_area)
        self.train_area.addLayout(self.parameters_area)

        ann_parameters_area = self.setup_ann_parameters_area()
        self.train_area.addLayout(ann_parameters_area)

        self.train_button = QtWidgets.QPushButton(parent=self.centralwidget)
        self.train_button.setText("Train")
        self.train_button.setObjectName("train_button")
        self.train_button.clicked.connect(self.start_train)
        self.train_area.addWidget(self.train_button)

        self.metaparameters_area.addLayout(self.train_area)

        tags_area = self.setup_tags_area()

        self.metaparameters_area.addLayout(tags_area)
        self.main_area.addLayout(self.metaparameters_area)

        export_area = self.setup_export_area()
        self.main_area.addLayout(export_area)

        self.status_text = QtWidgets.QTextBrowser(parent=self.centralwidget)
        self.status_text.setSizePolicy(_minimum_police_size)
        self.status_text.setMinimumSize(QtCore.QSize(0, 35))
        self.status_text.setFont(_font_12pt)
        self.status_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_text.setObjectName("status_text")
        self.main_area.addWidget(self.status_text)

        self.gridLayout_2.addLayout(self.main_area, 3, 0, 1, 1)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1037, 26))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        QtCore.QMetaObject.connectSlotsByName(self)

    def setup_dataset_area(self):
        dataset_area = QtWidgets.QVBoxLayout()
        dataset_area.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        dataset_area.setObjectName("dataset_area")

        load_dataset_text = QtWidgets.QTextBrowser(parent=self.centralwidget)
        load_dataset_text.setSizePolicy(_minimum_police_size)
        load_dataset_text.setMinimumSize(QtCore.QSize(0, 35))
        load_dataset_text.setObjectName("load_dataset_text")
        load_dataset_text.setFontPointSize(12)
        load_dataset_text.setPlainText("Load dataset")
        load_dataset_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dataset_area.addWidget(load_dataset_text)

        load_dataset_area = QtWidgets.QHBoxLayout()
        load_dataset_area.setObjectName("load_dataset_area")
        load_dataset_plaintext = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        load_dataset_plaintext.setSizePolicy(_expand_minimum_police_size)
        load_dataset_plaintext.setFont(_font_12pt)
        load_dataset_plaintext.setMinimumSize(QtCore.QSize(800, 35))
        load_dataset_plaintext.setObjectName("load_dataset_plaintext")
        load_dataset_area.addWidget(load_dataset_plaintext)

        dataset_size_area = QtWidgets.QVBoxLayout()
        x_dataset_size_area = QtWidgets.QHBoxLayout()
        x_dataset_size_text = QtWidgets.QTextBrowser(parent=self.centralwidget)
        x_dataset_size_text.setSizePolicy(_minimum_police_size)
        x_dataset_size_text.setMinimumSize(QtCore.QSize(20, 25))
        x_dataset_size_text.setObjectName("x_dataset_size_text")
        x_dataset_size_text.setFontPointSize(6)
        x_dataset_size_text.setPlainText("X size")
        x_dataset_size_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        x_dataset_size_area.addWidget(x_dataset_size_text)

        x_dataset_size_plaintext = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        x_dataset_size_plaintext.setSizePolicy(_minimum_police_size)
        x_dataset_size_plaintext.setMinimumSize(QtCore.QSize(20, 25))
        x_dataset_size_plaintext.setObjectName("x_dataset_size_plaintext")
        x_dataset_size_plaintext.setFont(_font_6pt)
        x_dataset_size_area.addWidget(x_dataset_size_plaintext)
        dataset_size_area.addLayout(x_dataset_size_area)

        y_dataset_size_area = QtWidgets.QHBoxLayout()
        y_dataset_size_text = QtWidgets.QTextBrowser(parent=self.centralwidget)
        y_dataset_size_text.setSizePolicy(_minimum_police_size)
        y_dataset_size_text.setMinimumSize(QtCore.QSize(20, 25))
        y_dataset_size_text.setObjectName("y_dataset_size_text")
        y_dataset_size_text.setFontPointSize(6)
        y_dataset_size_text.setPlainText("Y size")
        y_dataset_size_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        y_dataset_size_area.addWidget(y_dataset_size_text)

        y_dataset_size_plaintext = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        y_dataset_size_plaintext.setSizePolicy(_minimum_police_size)
        y_dataset_size_plaintext.setMinimumSize(QtCore.QSize(20, 25))
        y_dataset_size_plaintext.setObjectName("y_dataset_size_plaintext")
        y_dataset_size_plaintext.setFont(_font_6pt)
        y_dataset_size_area.addWidget(y_dataset_size_plaintext)
        dataset_size_area.addLayout(y_dataset_size_area)
        load_dataset_area.addLayout(dataset_size_area)

        load_dataset_button = QtWidgets.QPushButton(parent=self.centralwidget)
        load_dataset_button.setSizePolicy(_fixed_police_size)
        load_dataset_button.setMinimumSize(QtCore.QSize(100, 40))
        load_dataset_button.setObjectName("load_dataset_button")
        load_dataset_button.setText("Load")
        load_dataset_button.clicked.connect(self.load_dataset)
        load_dataset_area.addWidget(load_dataset_button)
        dataset_area.addLayout(load_dataset_area)

        return dataset_area

    def setup_parameters_desc_area(self):
        param_desc_area = QtWidgets.QVBoxLayout()
        param_desc_area.setObjectName("param_desc_area")

        text_browser_count = len(_param_desc_phrases)
        for i in range(text_browser_count):
            textBrowser = QtWidgets.QTextBrowser(parent=self.centralwidget)
            textBrowser.setSizePolicy(_minimum_police_size)
            textBrowser.setMinimumSize(QtCore.QSize(0, 26))
            textBrowser.setObjectName(f"textBrowser_{i + 1}")
            textBrowser.setPlainText(_param_desc_phrases[i])
            param_desc_area.addWidget(textBrowser)
        return param_desc_area

    def setup_parameters_value_area(self):
        param_value_area = QtWidgets.QVBoxLayout()
        param_value_area.setObjectName("param_value_area")

        launch_random_count_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        launch_random_count_text.setSizePolicy(_minimum_police_size)
        launch_random_count_text.setMinimumSize(QtCore.QSize(0, 26))
        launch_random_count_text.setObjectName("launch_random_count_text")
        param_value_area.addWidget(launch_random_count_text)

        launch_ann_count_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        launch_ann_count_text.setSizePolicy(_minimum_police_size)
        launch_ann_count_text.setMinimumSize(QtCore.QSize(0, 26))
        launch_ann_count_text.setObjectName("launch_ann_count_text")
        param_value_area.addWidget(launch_ann_count_text)

        max_len_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        max_len_text.setSizePolicy(_minimum_police_size)
        max_len_text.setMinimumSize(QtCore.QSize(0, 26))
        max_len_text.setObjectName("max_len_text")
        param_value_area.addWidget(max_len_text)

        min_len_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        min_len_text.setSizePolicy(_minimum_police_size)
        min_len_text.setMinimumSize(QtCore.QSize(0, 26))
        min_len_text.setObjectName("min_len_text")
        param_value_area.addWidget(min_len_text)

        block_size_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        block_size_text.setSizePolicy(_minimum_police_size)
        block_size_text.setMinimumSize(QtCore.QSize(0, 26))
        block_size_text.setObjectName("block_size_text")
        param_value_area.addWidget(block_size_text)

        offset_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        offset_text.setSizePolicy(_minimum_police_size)
        offset_text.setMinimumSize(QtCore.QSize(0, 26))
        offset_text.setObjectName("offset_text")
        param_value_area.addWidget(offset_text)

        alphabet_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        alphabet_text.setSizePolicy(_minimum_police_size)
        alphabet_text.setMinimumSize(QtCore.QSize(0, 26))
        alphabet_text.setObjectName("alphabet_text")
        param_value_area.addWidget(alphabet_text)

        min_epoch_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        min_epoch_text.setSizePolicy(_minimum_police_size)
        min_epoch_text.setMinimumSize(QtCore.QSize(0, 26))
        min_epoch_text.setObjectName("min_epoch_text")
        param_value_area.addWidget(min_epoch_text)

        max_epoch_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        max_epoch_text.setSizePolicy(_minimum_police_size)
        max_epoch_text.setMinimumSize(QtCore.QSize(0, 26))
        max_epoch_text.setObjectName("max_epoch_text")
        param_value_area.addWidget(max_epoch_text)

        iteration_count_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        iteration_count_text.setSizePolicy(_minimum_police_size)
        iteration_count_text.setMinimumSize(QtCore.QSize(0, 26))
        iteration_count_text.setObjectName("iteration_count_text")
        param_value_area.addWidget(iteration_count_text)

        loss_func_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        loss_func_combobox.setSizePolicy(_minimum_police_size)
        loss_func_combobox.setMinimumSize(QtCore.QSize(0, 26))
        loss_func_combobox.setObjectName("loss_func_combobox")
        for loss_func in degann.networks.get_all_loss_functions().keys():
            loss_func_combobox.addItem(loss_func)
        param_value_area.addWidget(loss_func_combobox)

        loss_threshold_text = QtWidgets.QLineEdit(parent=self.centralwidget)
        loss_threshold_text.setSizePolicy(_minimum_police_size)
        loss_threshold_text.setMinimumSize(QtCore.QSize(0, 26))
        loss_threshold_text.setObjectName("loss_threshold_text")
        param_value_area.addWidget(loss_threshold_text)

        optimizer_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        optimizer_combobox.setSizePolicy(_minimum_police_size)
        optimizer_combobox.setMinimumSize(QtCore.QSize(0, 26))
        optimizer_combobox.setObjectName("optimizer_combobox")
        for optimizer in degann.networks.get_all_optimizers().keys():
            optimizer_combobox.addItem(optimizer)
        param_value_area.addWidget(optimizer_combobox)

        return param_value_area

    def setup_ann_parameters_area(self):
        ann_parameters_area = QtWidgets.QHBoxLayout()
        ann_parameters_area.setObjectName("ann_parameters_area")

        textBrowser_13 = QtWidgets.QTextBrowser(parent=self.centralwidget)
        textBrowser_13.setSizePolicy(_minimum_police_size)
        textBrowser_13.setMinimumSize(QtCore.QSize(0, 26))
        textBrowser_13.setObjectName("textBrowser_13")
        textBrowser_13.setPlainText(_ann_param_desc_phrases[0])
        ann_parameters_area.addWidget(textBrowser_13)

        ann_param_desc_area = QtWidgets.QVBoxLayout()
        ann_param_desc_area.setObjectName("ann_param_desc_area")
        textBrowser_14 = QtWidgets.QTextBrowser(parent=self.centralwidget)
        textBrowser_14.setSizePolicy(_minimum_police_size)
        textBrowser_14.setMinimumSize(QtCore.QSize(0, 26))
        textBrowser_14.setObjectName("textBrowser_14")
        textBrowser_14.setPlainText(_ann_param_desc_phrases[1])
        ann_param_desc_area.addWidget(textBrowser_14)

        textBrowser_15 = QtWidgets.QTextBrowser(parent=self.centralwidget)
        textBrowser_15.setSizePolicy(_minimum_police_size)
        textBrowser_15.setMinimumSize(QtCore.QSize(0, 26))
        textBrowser_15.setObjectName("textBrowser_15")
        textBrowser_15.setPlainText(_ann_param_desc_phrases[2])
        ann_param_desc_area.addWidget(textBrowser_15)

        ann_parameters_area.addLayout(ann_param_desc_area)
        ann_param_value_area = QtWidgets.QVBoxLayout()
        ann_param_value_area.setObjectName("ann_param_value_area")

        ann_dist_comboBox = QtWidgets.QComboBox(parent=self.centralwidget)
        ann_dist_comboBox.setMinimumSize(QtCore.QSize(0, 26))
        ann_dist_comboBox.setObjectName("ann_dist_comboBox")
        ann_dist_comboBox.addItem("Linear")
        ann_dist_comboBox.addItem("Const")
        ann_param_value_area.addWidget(ann_dist_comboBox)
        ann_temp_comboBox = QtWidgets.QComboBox(parent=self.centralwidget)

        ann_temp_comboBox.setMinimumSize(QtCore.QSize(0, 26))
        ann_temp_comboBox.setObjectName("ann_temp_comboBox")
        ann_temp_comboBox.addItem("Linear")
        ann_temp_comboBox.addItem("Exponent")
        ann_param_value_area.addWidget(ann_temp_comboBox)
        ann_parameters_area.addLayout(ann_param_value_area)

        ann_method_consts_area = QtWidgets.QVBoxLayout()
        ann_method_consts_area.setObjectName("ann_method_consts_area")

        ann_dist_consts_area = QtWidgets.QHBoxLayout()
        ann_dist_consts_area.setObjectName("ann_dist_consts_area")

        dist_offset = QtWidgets.QLineEdit(parent=self.centralwidget)
        dist_offset.setMinimumSize(QtCore.QSize(0, 26))
        dist_offset.setObjectName("dist_offset")
        ann_dist_consts_area.addWidget(dist_offset)

        dist_scale = QtWidgets.QLineEdit(parent=self.centralwidget)
        dist_scale.setMinimumSize(QtCore.QSize(0, 26))
        dist_scale.setObjectName("dist_scale")
        ann_dist_consts_area.addWidget(dist_scale)
        ann_method_consts_area.addLayout(ann_dist_consts_area)

        ann_temp_consts_area = QtWidgets.QHBoxLayout()
        ann_temp_consts_area.setObjectName("ann_temp_consts_area")

        temp_speed = QtWidgets.QLineEdit(parent=self.centralwidget)
        temp_speed.setMinimumSize(QtCore.QSize(0, 26))
        temp_speed.setObjectName("temp_speed")
        ann_temp_consts_area.addWidget(temp_speed)
        ann_method_consts_area.addLayout(ann_temp_consts_area)

        ann_parameters_area.addLayout(ann_param_value_area)
        ann_parameters_area.addLayout(ann_method_consts_area)

        return ann_parameters_area

    def setup_tags_area(self):
        tags_area = QtWidgets.QVBoxLayout()
        tags_area.setObjectName("tags_area")
        tags_value_area = QtWidgets.QFormLayout()
        tags_value_area.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetMinimumSize
        )
        tags_value_area.setObjectName("tags_value_area")

        eq_type_text = QtWidgets.QTextEdit(parent=self.centralwidget)
        eq_type_text.setSizePolicy(_fixed_police_size)
        eq_type_text.setMinimumSize(QtCore.QSize(10, 35))
        eq_type_text.setFontPointSize(12)
        eq_type_text.setPlainText("Type of equation")
        eq_type_text.setObjectName("eq_type_text")
        tags_value_area.setWidget(
            0, QtWidgets.QFormLayout.ItemRole.LabelRole, eq_type_text
        )

        eq_type_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        eq_type_combobox.setMinimumSize(QtCore.QSize(0, 40))
        eq_type_combobox.setFont(_font_12pt)
        eq_type_combobox.setObjectName("eq_type_combobox")
        for tag in selector.tags["type"]:
            eq_type_combobox.addItem(tag)
        eq_type_combobox.setCurrentText("unknown")
        tags_value_area.setWidget(
            0, QtWidgets.QFormLayout.ItemRole.FieldRole, eq_type_combobox
        )

        precision_text = QtWidgets.QTextEdit(parent=self.centralwidget)
        precision_text.setEnabled(True)
        precision_text.setSizePolicy(_minimum_police_size)
        precision_text.setMinimumSize(QtCore.QSize(0, 35))
        precision_text.setFont(_font_12pt)
        precision_text.setPlainText("Required level of accuracy")
        precision_text.setObjectName("precision_text")
        tags_value_area.setWidget(
            1, QtWidgets.QFormLayout.ItemRole.LabelRole, precision_text
        )

        precision_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        precision_combobox.setSizePolicy(_minimum_police_size)
        precision_combobox.setMinimumSize(QtCore.QSize(0, 40))
        precision_combobox.setFont(_font_12pt)
        precision_combobox.setObjectName("precision_combobox")
        for precision in selector.tags["precision"]:
            precision_combobox.addItem(precision)
        precision_combobox.setCurrentText("median")
        tags_value_area.setWidget(
            1, QtWidgets.QFormLayout.ItemRole.FieldRole, precision_combobox
        )

        work_time_text = QtWidgets.QTextEdit(parent=self.centralwidget)
        work_time_text.setSizePolicy(_fixed_police_size)
        work_time_text.setMinimumSize(QtCore.QSize(0, 35))
        work_time_text.setFont(_font_12pt)
        work_time_text.setPlainText("Running time of a trained neural network")
        work_time_text.setObjectName("work_time_text")
        tags_value_area.setWidget(
            2, QtWidgets.QFormLayout.ItemRole.LabelRole, work_time_text
        )

        work_time_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        work_time_combobox.setMinimumSize(QtCore.QSize(0, 40))
        work_time_combobox.setFont(_font_12pt)
        work_time_combobox.setObjectName("work_time_combobox")
        for work_time in selector.tags["work time"]:
            work_time_combobox.addItem(work_time)
        work_time_combobox.setCurrentText("medium")
        tags_value_area.setWidget(
            2, QtWidgets.QFormLayout.ItemRole.FieldRole, work_time_combobox
        )

        data_size_text = QtWidgets.QTextEdit(parent=self.centralwidget)
        data_size_text.setSizePolicy(_fixed_police_size)
        data_size_text.setMinimumSize(QtCore.QSize(10, 35))
        data_size_text.setFont(_font_12pt)
        data_size_text.setObjectName("data_size_text")
        data_size_text.setPlainText("Number of samples in passed data")
        tags_value_area.setWidget(
            3, QtWidgets.QFormLayout.ItemRole.LabelRole, data_size_text
        )

        data_size_combobox = QtWidgets.QComboBox(parent=self.centralwidget)
        data_size_combobox.setMinimumSize(QtCore.QSize(0, 40))
        data_size_combobox.setFont(_font_12pt)
        data_size_combobox.setObjectName("data_size_combobox")
        for data_size in selector.tags["data size"]:
            data_size_combobox.addItem(data_size)
        data_size_combobox.setCurrentText("auto")
        tags_value_area.setWidget(
            3, QtWidgets.QFormLayout.ItemRole.FieldRole, data_size_combobox
        )
        tags_area.addLayout(tags_value_area)

        selection_button = QtWidgets.QPushButton(parent=self.centralwidget)
        selection_button.setObjectName("selection_button")
        selection_button.setText("select parameters")
        selection_button.clicked.connect(self.select_parameters)
        tags_area.addWidget(selection_button)

        return tags_area

    def setup_export_area(self):
        export_area = QtWidgets.QVBoxLayout()
        export_area.setObjectName("export_area")

        export_nn_text = QtWidgets.QTextBrowser(parent=self.centralwidget)
        export_nn_text.setSizePolicy(_minimum_police_size)
        export_nn_text.setFont(_font_12pt)
        export_nn_text.setPlainText("Export neural network")
        export_nn_text.setMinimumSize(QtCore.QSize(0, 35))
        export_nn_text.setObjectName("export_nn_text")
        export_area.addWidget(export_nn_text)

        export_buttons_area = QtWidgets.QHBoxLayout()
        export_buttons_area.setObjectName("export_buttons_area")

        export_nn_plaintext = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        export_nn_plaintext.setSizePolicy(_minimum_police_size)
        export_nn_plaintext.setMinimumSize(QtCore.QSize(200, 35))
        export_nn_plaintext.setObjectName("export_nn_plaintext")
        export_buttons_area.addWidget(export_nn_plaintext)

        export_button = QtWidgets.QPushButton(parent=self.centralwidget)
        export_button.setSizePolicy(_fixed_police_size)
        export_button.setMinimumSize(QtCore.QSize(100, 40))
        export_button.setObjectName("export_button")
        export_button.clicked.connect(self.export_to_file)
        export_button.setText("Export")
        export_buttons_area.addWidget(export_button)

        export_cpp_button = QtWidgets.QPushButton(parent=self.centralwidget)
        export_cpp_button.setSizePolicy(_fixed_police_size)
        export_cpp_button.setMinimumSize(QtCore.QSize(100, 40))
        export_cpp_button.setObjectName("export_cpp_button")
        export_cpp_button.clicked.connect(self.export_to_cpp)
        export_cpp_button.setText("Export to cpp")
        export_buttons_area.addWidget(export_cpp_button)

        export_area.addLayout(export_buttons_area)

        return export_area

    def load_dataset(self):
        path_to_data = self.centralwidget.findChild(
            QtWidgets.QPlainTextEdit, "load_dataset_plaintext"
        ).toPlainText()
        self.x_dataset_size = int(
            self.centralwidget.findChild(
                QtWidgets.QPlainTextEdit, "x_dataset_size_plaintext"
            ).toPlainText()
        )
        self.y_dataset_size = int(
            self.centralwidget.findChild(
                QtWidgets.QPlainTextEdit, "y_dataset_size_plaintext"
            ).toPlainText()
        )
        self.train_data_x, self.train_data_y = np.genfromtxt(
            path_to_data,
            delimiter=",",
            unpack=True,
            usecols=list(range(self.x_dataset_size)),
        ), np.genfromtxt(
            path_to_data,
            delimiter=",",
            usecols=list(
                range(self.x_dataset_size, self.x_dataset_size + self.y_dataset_size)
            ),
        )
        self.train_data_x = self.train_data_x.reshape((self.x_dataset_size, -1)).T
        self.train_data_y = self.train_data_y.reshape((self.y_dataset_size, -1)).T

        self.centralwidget.findChild(
            QtWidgets.QTextBrowser, "status_text"
        ).setPlainText("Success load")

    def select_parameters(self):
        tags = {
            "type": self.centralwidget.findChild(
                QtWidgets.QComboBox, "eq_type_combobox"
            ).currentText(),
            "precision": self.centralwidget.findChild(
                QtWidgets.QComboBox, "precision_combobox"
            ).currentText(),
            "work time": self.centralwidget.findChild(
                QtWidgets.QComboBox, "work_time_combobox"
            ).currentText(),
            "data size": self.centralwidget.findChild(
                QtWidgets.QComboBox, "data_size_combobox"
            ).currentText(),
        }
        self.parameters = selector.suggest_parameters(
            (self.train_data_x, self.train_data_y), tags
        )

        for gui_name, param_name in zip(
            _parameter_value_gui_names, _parameter_value_code_names
        ):
            self.centralwidget.findChild(QtWidgets.QLineEdit, gui_name).setText(
                str(self.parameters[param_name])
            )
        self.centralwidget.findChild(
            QtWidgets.QComboBox, "loss_func_combobox"
        ).setCurrentText(str(self.parameters["loss_function"]))
        self.centralwidget.findChild(
            QtWidgets.QComboBox, "optimizer_combobox"
        ).setCurrentText(str(self.parameters["optimizer"]))

        for gui_name, ann_param_name in zip(
            _ann_param_gui_names, _ann_param_code_names
        ):
            self.centralwidget.findChild(QtWidgets.QLineEdit, gui_name).setText(
                str(self.parameters["simulated_annealing_params"][ann_param_name])
            )
        self.centralwidget.findChild(
            QtWidgets.QComboBox, "ann_dist_comboBox"
        ).setCurrentText(
            str(self.parameters["simulated_annealing_params"]["distance_to_neighbor"])
        )
        self.centralwidget.findChild(
            QtWidgets.QComboBox, "ann_temp_comboBox"
        ).setCurrentText(
            str(
                self.parameters["simulated_annealing_params"][
                    "temperature_reduction_method"
                ]
            )
        )

        self.centralwidget.findChild(
            QtWidgets.QTextBrowser, "status_text"
        ).setPlainText("Success select parameters")

    def start_train(self):
        parameters = {
            "launch_count_random_search": "",
            "launch_count_simulated_annealing": "",
            "nn_max_length": "",
            "nn_min_length": "",
            "nn_alphabet_block_size": "",
            "nn_alphabet_offset": "",
            "nn_alphabet": "",
            "min_train_epoch": "",
            "max_train_epoch": "",
            "iteration_count": "",
            "loss_function": "",
            "loss_threshold": "",
            "optimizer": "",
            "simulated_annealing_params": dict(),
        }
        for code_name, gui_name in zip(
            _parameter_value_code_names, _parameter_value_gui_names
        ):
            temp = self.centralwidget.findChild(QtWidgets.QLineEdit, gui_name).text()
            if code_name != "nn_alphabet":
                temp = float(temp)
                if temp - int(temp) < 1e-6:
                    temp = int(temp)
            elif code_name == "nn_alphabet":
                alphabet_str = temp.replace("'", "")
                alphabet = alphabet_str[1:-1].split(", ")
                temp = alphabet
            parameters[code_name] = temp
        parameters["loss_function"] = self.centralwidget.findChild(
            QtWidgets.QComboBox, "loss_func_combobox"
        ).currentText()
        parameters["optimizer"] = self.centralwidget.findChild(
            QtWidgets.QComboBox, "optimizer_combobox"
        ).currentText()

        for code_name, gui_name in zip(_ann_param_code_names, _ann_param_gui_names):
            temp = self.centralwidget.findChild(QtWidgets.QLineEdit, gui_name).text()
            parameters["simulated_annealing_params"][code_name] = temp
        parameters["simulated_annealing_params"][
            "distance_to_neighbor"
        ] = self.centralwidget.findChild(
            QtWidgets.QComboBox, "ann_dist_comboBox"
        ).currentText()
        parameters["simulated_annealing_params"][
            "temperature_reduction_method"
        ] = self.centralwidget.findChild(
            QtWidgets.QComboBox, "ann_temp_comboBox"
        ).currentText()

        self.train_loss, self.trained_nn = selector.execute_pipeline(
            input_size=self.x_dataset_size,
            output_size=self.y_dataset_size,
            data=(self.train_data_x, self.train_data_y),
            parameters=parameters,
        )

        self.centralwidget.findChild(
            QtWidgets.QTextBrowser, "status_text"
        ).setPlainText(
            f"Neural network has been successfully trained. Final value of loss function = {self.train_loss}"
        )

    def export_to_file(self):
        export_path = self.centralwidget.findChild(
            QtWidgets.QPlainTextEdit, "export_nn_plaintext"
        ).toPlainText()
        nn = IModel(
            input_size=self.x_dataset_size,
            block_size=[],
            output_size=self.y_dataset_size,
        )
        nn.from_dict(self.trained_nn)
        nn.export_to_file(export_path)
        self.centralwidget.findChild(
            QtWidgets.QTextBrowser, "status_text"
        ).setPlainText(f"Success export to {export_path + '.apg'}")

    def export_to_cpp(self):
        export_path = self.centralwidget.findChild(
            QtWidgets.QPlainTextEdit, "export_nn_plaintext"
        ).toPlainText()
        nn = IModel(
            input_size=self.x_dataset_size,
            block_size=[],
            output_size=self.y_dataset_size,
        )
        nn.from_dict(self.trained_nn)
        nn.export_to_cpp(export_path)
        self.centralwidget.findChild(
            QtWidgets.QTextBrowser, "status_text"
        ).setPlainText(f"Success export to {export_path + '.cpp'}")
