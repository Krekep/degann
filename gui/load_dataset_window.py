import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtWidgets import QMainWindow, QLayout

from gui.constants import minimum_police_size, expand_minimum_police_size, font_12pt, font_6pt, fixed_police_size, \
    expand_fixed_police_size

_load_dataset_text = ("Loading the dataset.\n" +
                      "The dataset should be a .csv file of `x size`+`y size` columns, " +
                      "where the first `x size` columns are feature vectors (written in a line), " +
                      "and the last `y size` columns are their corresponding value vectors.")


class LoadDatasetLayout():
    def __init__(self, centralwidget):
        self.centralwidget = centralwidget
        self.setup_ui()

    def setup_ui(self):
        self.dataset_area = self.setup_dataset_area()

    def setup_dataset_area(self):
        dataset_area = QtWidgets.QVBoxLayout(self.centralwidget)
        dataset_area.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        dataset_area.setObjectName("dataset_area")

        load_dataset_text = QtWidgets.QTextBrowser(parent=self.centralwidget)
        load_dataset_text.setSizePolicy(minimum_police_size)
        load_dataset_text.setMinimumSize(QtCore.QSize(0, 35))
        load_dataset_text.setObjectName("load_dataset_text")
        load_dataset_text.setFontPointSize(12)
        lines = _load_dataset_text.strip().split("\n")
        for line in lines:
            load_dataset_text.append(line)
            load_dataset_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dataset_area.addWidget(load_dataset_text)

        # load_dataset_plaintext = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        # load_dataset_plaintext.setSizePolicy(expand_minimum_police_size)
        # load_dataset_plaintext.setFont(font_12pt)
        # load_dataset_plaintext.setMinimumSize(QtCore.QSize(400, 40))
        # load_dataset_plaintext.setObjectName("load_dataset_plaintext")
        # dataset_area.addWidget(load_dataset_plaintext)

        dataset_size_area = QtWidgets.QVBoxLayout()
        x_dataset_size_area = QtWidgets.QHBoxLayout()
        x_dataset_size_text = QtWidgets.QTextBrowser(parent=self.centralwidget)
        x_dataset_size_text.setSizePolicy(minimum_police_size)
        x_dataset_size_text.setMinimumSize(QtCore.QSize(20, 25))
        x_dataset_size_text.setObjectName("x_dataset_size_text")
        x_dataset_size_text.setFontPointSize(12)
        x_dataset_size_text.setPlainText("X size")
        x_dataset_size_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        x_dataset_size_area.addWidget(x_dataset_size_text)

        x_dataset_size_plaintext = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        x_dataset_size_plaintext.setSizePolicy(minimum_police_size)
        x_dataset_size_plaintext.setMinimumSize(QtCore.QSize(20, 25))
        x_dataset_size_plaintext.setObjectName("x_dataset_size_plaintext")
        x_dataset_size_plaintext.setFont(font_12pt)
        x_dataset_size_area.addWidget(x_dataset_size_plaintext)
        dataset_size_area.addLayout(x_dataset_size_area)

        y_dataset_size_area = QtWidgets.QHBoxLayout()
        y_dataset_size_text = QtWidgets.QTextBrowser(parent=self.centralwidget)
        y_dataset_size_text.setSizePolicy(minimum_police_size)
        y_dataset_size_text.setMinimumSize(QtCore.QSize(20, 25))
        y_dataset_size_text.setObjectName("y_dataset_size_text")
        y_dataset_size_text.setFontPointSize(12)
        y_dataset_size_text.setPlainText("Y size")
        y_dataset_size_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        y_dataset_size_area.addWidget(y_dataset_size_text)

        y_dataset_size_plaintext = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        y_dataset_size_plaintext.setSizePolicy(minimum_police_size)
        y_dataset_size_plaintext.setMinimumSize(QtCore.QSize(20, 25))
        y_dataset_size_plaintext.setObjectName("y_dataset_size_plaintext")
        y_dataset_size_plaintext.setFont(font_12pt)
        y_dataset_size_area.addWidget(y_dataset_size_plaintext)
        dataset_size_area.addLayout(y_dataset_size_area)
        # load_dataset_area.addLayout(dataset_size_area)
        dataset_area.addLayout(dataset_size_area)

        load_dataset_button = QtWidgets.QPushButton(parent=self.centralwidget)
        load_dataset_button.setSizePolicy(expand_fixed_police_size)
        load_dataset_button.setMinimumSize(QtCore.QSize(100, 40))
        load_dataset_button.setObjectName("load_dataset_button")
        load_dataset_button.setText("Load")
        # load_dataset_area.addWidget(load_dataset_button)
        dataset_area.addWidget(load_dataset_button)
        # dataset_area.addLayout(load_dataset_area)

        return dataset_area
