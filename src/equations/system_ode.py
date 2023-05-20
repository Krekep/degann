"""
Provide class for solve system of ODE
"""

from types import FunctionType
from typing import Union, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt

from src.equations import utils
from src.networks.utils import export_csv_table

__all__ = ["SystemODE"]


class SystemODE(object):
    def __init__(self, debug=False):
        self._sol = None
        self._func = []
        self._func_arguments = ""
        self._initial_values = []
        self._size = 0
        self._debug = debug

    def _f(self, t, y):
        """
        Additional function for scipy.solve_ivp

        Parameters
        ----------
        t
        y: list

        Returns
        -------
        res: list
        """
        res = []
        for i, y_i in enumerate(y):
            res.append(self._func[i](*y))
        return res

    def prepare_equations(self, n: int, equations: List[List[str]]) -> None:
        """
        Parse passed equations and build functions for them

        Parameters
        ----------
        n: int
            System size
        equations: list[list[str]]
            System of equations. Equation yi' = f(x, yj...) and Cauchy initial value yi(x0) = yi0
        """
        self._size = n
        self._func_arguments = ""
        for i in range(n):
            self._func_arguments += f"y{i}, "
        self._func_arguments = self._func_arguments[:-2]

        for i, cond in enumerate(equations):
            string_func = f"def dy{i}_dt({self._func_arguments}): return {cond[0]}"
            if self._debug:
                print("Builded function:", string_func)
            code = compile(string_func, "<string>", "exec")

            self._func.append(FunctionType(code.co_consts[0], globals(), "temp"))
            self._initial_values.append(utils.extract_iv(cond[1])[1])

    def solve(
        self, interval: Tuple[float, float], points: Union[int, list] = None
    ) -> None:
        """
        Solve given equations

        Parameters
        ----------
        interval: tuple[int, int]
            Decision interval
        points: int | list
            Amount of points per interval (or list of points)
        """
        t_span = np.array([interval[0], interval[1]])
        if points is not None:
            if isinstance(points, list):
                times = points
            else:
                times = np.linspace(t_span[0], t_span[1], points)
            self._sol = solve_ivp(
                self._f, t_span, self._initial_values, t_eval=times, method="LSODA"
            )
        else:
            self._sol = solve_ivp(self._f, t_span, self._initial_values, method="LSODA")
        if self._debug:
            print("Success solve")

            plt.rc("font", size=24)
            plt.figure()
            t = self._sol.t
            y = self._sol.y

            result_path = f"solveTable.csv"
            t_for_export = t[:, np.newaxis]
            export_csv_table(np.concatenate((t_for_export, y.T), axis=1), result_path)

            for i, y_i in enumerate(y):
                plt.plot(t, y_i, "-", label=f"{i}")
            plt.legend()
            plt.show()

    def build_table(self, eq_num: list = None) -> np.ndarray:
        """
        Builds a table for solution

        Parameters
        ----------
        eq_num : list
            For which equation in system build table

        Returns
        -------
        table: list
            Table with 1+k column, where 1 is the number of variables (ODE have 1 variable),
            and the last k columns contain the solution of ODE system
        """
        if self._sol is None:
            return np.array([])

        if eq_num is None:
            t = self._sol.t
            y = self._sol.y
            res = np.vstack([t, *y])
            res = res.T
        else:
            t = self._sol.t
            y = self._sol.y[eq_num, :]
            res = np.vstack([t, *y])
            res = res.T
        return res
