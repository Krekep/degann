import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QMainWindow, QLayout

from gui.constants import minimum_police_size, expand_minimum_police_size, font_12pt, font_6pt, fixed_police_size, \
    expand_fixed_police_size

_export_nn_text = ("Export trained neural network.\n" +
                   "You must specify a name for the output file in the text field")


class ExportNNLayout():
    def __init__(self, centralwidget):
        self.centralwidget = centralwidget
        self.setup_ui()

    def setup_ui(self):
        self.main_area = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_area.setObjectName("ExportNNLayout")

        back_button = QtWidgets.QPushButton(parent=self.centralwidget)
        back_button.setSizePolicy(expand_fixed_police_size)
        back_button.setMinimumSize(QtCore.QSize(50, 40))
        back_button.setObjectName("back_button")
        back_button.setText("Back")
        self.main_area.addWidget(back_button)

        self.main_area.addLayout(self.setup_export_area())

    def setup_export_area(self):
        export_area = QtWidgets.QVBoxLayout()
        export_area.setObjectName("export_area")

        export_nn_text = QtWidgets.QTextBrowser(parent=self.centralwidget)
        export_nn_text.setSizePolicy(minimum_police_size)
        export_nn_text.setFont(font_12pt)
        lines = _export_nn_text.strip().split("\n")
        for line in lines:
            export_nn_text.append(line)
            export_nn_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        export_nn_text.setMinimumSize(QtCore.QSize(0, 75))
        export_nn_text.setObjectName("export_nn_text")
        export_area.addWidget(export_nn_text)

        export_nn_plaintext = QtWidgets.QPlainTextEdit("Replace this text with file name", parent=self.centralwidget)
        font = QFont()
        font.setItalic(True)
        font.setPointSize(12)
        export_nn_plaintext.setFont(font)
        export_nn_plaintext.setSizePolicy(minimum_police_size)
        export_nn_plaintext.setMinimumSize(QtCore.QSize(200, 35))
        export_nn_plaintext.setObjectName("export_nn_plaintext")
        export_area.addWidget(export_nn_plaintext)

        export_buttons_area = QtWidgets.QHBoxLayout()
        export_buttons_area.setObjectName("export_buttons_area")

        export_button = QtWidgets.QPushButton(parent=self.centralwidget)
        export_button.setSizePolicy(fixed_police_size)
        export_button.setMinimumSize(QtCore.QSize(100, 40))
        export_button.setObjectName("export_button")
        export_button.setText("Export")
        export_buttons_area.addWidget(export_button)

        export_cpp_button = QtWidgets.QPushButton(parent=self.centralwidget)
        export_cpp_button.setSizePolicy(fixed_police_size)
        export_cpp_button.setMinimumSize(QtCore.QSize(100, 40))
        export_cpp_button.setObjectName("export_cpp_button")
        export_cpp_button.setText("Export to cpp")
        export_buttons_area.addWidget(export_cpp_button)

        export_area.addLayout(export_buttons_area)

        return export_area
