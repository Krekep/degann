"""
Provide some helpful functions
"""

import pickle
from typing import Tuple

import numpy as np

from degann.networks import imodel


def import_csv_table(path: str) -> np.ndarray:
    """
    Import csv table as numpy array.

    Parameters
    ----------
    path: str
        Path to csv table
    Returns
    -------
    table: np.ndarray
        Parsing result
    """
    table = np.genfromtxt(path, delimiter=",")
    return table


def export_csv_table(table: np.ndarray, path: str) -> None:
    """
    Export numpy array to csv table.

    Parameters
    ----------
    table: np.ndarray
        Table to export
    path: str
        Path to csv table
    Returns
    -------
    None
    """
    np.savetxt(path, table, delimiter=",")


def split_table_by_ans(
    table: np.ndarray, len_answers=1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splitting the original table into tables of variables and answers by length of answer.

    Parameters
    ----------
    table: np.ndarray
        Input table
    len_answers: int
        Amount of answer variables (in columns)
    Returns
    -------
    tables: Tuple[np.ndarray, np.ndarray]
        Pair of inputs and results tables
    """
    x = table[:, :-len_answers]
    y = table[:, -len_answers:]
    return x, y


def split_table_by_inp(table: np.ndarray, len_input=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splitting the original table into tables of variables and answers by length of input.

    Parameters
    ----------
    table: np.ndarray
        Input table
    len_input: int
        Amount of input variables (in columns)
    Returns
    -------
    tables: Tuple[np.ndarray, np.ndarray]
        Pair of inputs and results tables
    """
    x = table[:, :len_input]
    y = table[:, len_input:]
    return x, y


def shuffle_table(table: np.ndarray) -> np.ndarray:
    """
    For shuffling table.

    Parameters
    ----------
    table: np.ndarray
        Table to be shuffled
    Returns
    -------
    table: np.ndarray
        Result of shuffle
    """
    np.random.shuffle(table)
    return table
