import math
from typing import Tuple

import numpy as np

__all__ = [
    "exp",
    "lin",
    "log",
    "sin",
    "gauss",
    "hyperbol",
    "const",
    "sig",
]

from scipy.integrate import solve_ivp

from degann import SystemODE


def LF_ODE_1_solution(x):
    """solution function for y' + 3y = 0, y(0) = 1"""
    return np.power(np.e, 3 * x)


def LF_ODE_2_solution(x):
    """solution function for y'= 0, y(0) = 1"""
    return 1


def LF_ODE_3_solution(x):
    """solution function for y' - y = 0, y(0) = 1"""
    return np.power(np.e, x)


def LH_ODE_1_solution(x):
    """solution function for y'' + 100y = 0, y(0) = 0, y'(0) = 10"""
    return np.sin(10 * x)


def LH_ODE_2_solution(x):
    """solution function for y'' + 1/5y' + y + 1/5*e^(-1/5x)*cos(x) = 0, y(0) = 0, y(1) = sin(1) / e^0.2"""
    return np.power(np.e, -1 / 5 * x) * np.sin(x)


def NLF_ODE_1_solution(x):
    """solution function for y' + (y - 2x) / x = 0, y(0.1) = 20.1"""
    return x + 2 / x


def NLF_ODE_2_solution(x):
    """solution function for sin(x)*y' + y*cos(x) - 1 = 0, y(0.1) = 2.1 / sin(0.1)"""
    return (x + 2) / np.sin(x)


def S_ODE_1_solution(x, size=2):
    """
    y0' = -100 * y1 y0(0)=10
    y1' = y0 y0(0)=0
    """
    return 10 * np.cos(10 * x), np.sin(10 * x)


def S_ODE_2_table(points_array: list, interval: Tuple[float, float] = (0, np.pi)):
    """
    y0' = y1 * y2 y0(0)=0
    y1' = -y0 * y2 y1(0)=0
    y2' = -0.5 * y0 * y1 y2(0) = 0
    """
    size = 3
    sode = "y1*y2 y0(0)=0\n" "-y0*y2 y1(0)=1\n" "-0.5*y0*y1 y2(0)=1"
    temp = sode.split("\n")
    prepared_sode = []
    for eq in temp:
        prepared_sode.append(eq.split(" "))

    solver = SystemODE(debug=True)
    solver.prepare_equations(size, prepared_sode)
    solver.solve(interval, points_array)
    res_table = solver.build_table()
    return res_table


def ST_LF_ODE_1_solution(x):
    """solution function for stiff y' + 15y = 0, y(0) = 1"""
    return np.power(np.e, -15 * x)


# Python cant calculate this
# def ST_LH_ODE_2_solution(x):
#     """ solution function for stiff y'' + 1001y' + 1000y = 0, y(0) = 1, y'(0) = 0 """
#     return 1/999 * np.power(np.e, -1000 * x) * (1000 * np.power(np.e, 999 * x) - 1)


def ST_LH_ODE_2_table(points_array: list, interval: Tuple[float, float] = (0, 1)):
    """solution function for stiff y'' + 1001y' + 1000y = 0, y(0) = 1, y'(0) = 0"""
    size = 2
    sode = "y1 y0(0)=1\n" "-1001*y1-1000*y0 y1(0)=0"
    temp = sode.split("\n")
    prepared_sode = []
    for eq in temp:
        prepared_sode.append(eq.split(" "))

    solver = SystemODE()
    solver.prepare_equations(size, prepared_sode)
    solver.solve(interval, points_array)
    res_table = solver.build_table([0])
    return res_table


def ST_S_ODE_3_table(points_array: list, interval: Tuple[float, float] = (0, 40)):
    def roberts_deriv(t, y):
        """ODEs for Robertson's chemical reaction system."""
        x, y, z = y
        xdot = -0.04 * x + 1.0e4 * y * z
        ydot = 0.04 * x - 1.0e4 * y * z - 3.0e7 * y**2
        zdot = 3.0e7 * y**2
        return xdot, ydot, zdot

    # Initial and final times.
    t0, tf = interval[0], interval[1]
    times = points_array
    # Initial conditions: [X] = 1; [Y] = [Z] = 0.
    y0 = 1, 0, 0
    # Solve, using a method resilient to stiff ODEs.
    soln = solve_ivp(roberts_deriv, (t0, tf), y0, t_eval=times, method="Radau")
    # print(soln.nfev, "evaluations required.")

    # Plot the concentrations as a function of time. Scale [Y] by 10**YFAC
    # so its variation is visible on the same axis used for [X] and [Z].
    # YFAC = 4
    # plt.plot(soln.t, soln.y[0], label='[X]')
    # plt.plot(soln.t, 10 ** YFAC * soln.y[1], label=r'$10^{}\times$[Y]'.format(YFAC))
    # plt.plot(soln.t, soln.y[2], label='[Z]')
    # plt.xlabel('time /s')
    # plt.ylabel('concentration /arb. units')
    # plt.legend()
    # plt.show()

    t = soln.t
    y = soln.y
    res = np.vstack([t, *y])
    res = res.T
    return res


exp = LF_ODE_1_solution
lin = lambda x: x * 2
log = lambda x: np.log(x + 1)
sin = LH_ODE_1_solution
gauss = lambda x: np.float_power(np.e, -((x - 0.5) ** 2) / 0.08)
hyperbol = lambda x: (np.float_power(x, 2) + 0.5) / (x + 0.1)
const = LF_ODE_2_solution
sig = lambda x: 1 / (1 + np.float_power(np.e, -x))
multidim = lambda x, y, z: np.sin(5 * x) * np.log2(1 + y) / np.sqrt(1 + z)
