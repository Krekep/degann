import numpy as np
from typing import Tuple, Callable

__all__ = ["func_to_tables"]

rng = np.random.default_rng()


# def prepare_interval(interval: Tuple[float, float], step: float, distr="uniform"):
#     a = interval[0]
#     b = interval[1]
#
#     points_count = int((b - a) / step)
#     scale = b - a
#     if distr == "uniform":
#         d = distributions[distr](loc=a, scale=b - a, size=points_count)
#     elif distr == "norm":
#         d = distributions[distr](
#             loc=a + (b - a) / 2, scale=(b - a) / 2, size=2 * points_count
#         )
#     elif distr == "binom" or distr == "nbinom":
#         d = (
#             distributions[distr](
#                 n=points_count * 30, p=0.5, loc=0, size=points_count * 2
#             )
#             / 30
#             / points_count
#             * scale
#             + a
#         )
#
#     temp = np.unique(d[(a <= d) & (d <= b)]).tolist()
#     res = sorted(temp)
#
#     return res


def prepare_uniform_interval(interval: Tuple[float, float], step: float):
    a = interval[0]
    b = interval[1]
    res = []
    while a <= b:
        res.append(a)
        a += step

    return res


list_sol_functions = [
    (LF_ODE_1_solution, (0, 1)),
    (LF_ODE_2_solution, (0, 1)),
    (LF_ODE_3_solution, (0, 1)),
    (NLF_ODE_1_solution, (0.1, 1)),
    (NLF_ODE_2_solution, (0.1, 1)),
    (NLF_ODE_3_solution, (0.1, 1)),
    (NLF_ODE_4_solution, (0.1, 1)),
    (ST_LF_ODE_1_solution, (0, 1)),
    (LH_ODE_1_solution, (0, 1)),
    (LH_ODE_2_solution, (0, 1)),
    (S_ODE_1_solution, (0, 1)),
]

list_table_functions = [
    (S_ODE_2_table, (0, np.pi)),
    (ST_LH_ODE_2_table, (0, 1)),
    (ST_S_ODE_3_table, (0, 40)),
]


def func_to_tables(
    funcs: dict[str, tuple[Callable, tuple[float, float]]], step=0.05, val_step=0.001
):
    for name, v in funcs:
        func = v[0]
        interval = v[1]
        for folder, s in zip(["data", "validation_data"], [step, val_step]):
            x = prepare_uniform_interval(interval, s)
            table = func(x)
            np.savetxt(f"./solution_tables/{folder}/{name}.csv", table, delimiter=",")


if __name__ == "__main__":
    step = 0.05
    for i in range(3):
        func = list_table_functions[i][0]
        interval = list_table_functions[i][1]
        x = prepare_uniform_interval(interval, step)
        table = func(x)
        np.savetxt(f"./solution_tables/data/{func.__name__}.csv", table, delimiter=",")

    step = 0.05
    for i in range(0, 11):
        func = list_sol_functions[i][0]
        interval = list_sol_functions[i][1]
        x = prepare_uniform_interval(interval, step)
        y = [func(i) for i in x]
        z = []
        for j in range(len(x)):
            if isinstance(y[j], tuple) or isinstance(y[j], list):
                z.append([x[j]] + list(y[j]))
            else:
                z.append([x[j], y[j]])
        table = np.array(z)
        # table = table.transpose()
        np.savetxt(f"./solution_tables/data/{func.__name__}.csv", table, delimiter=",")

    # build validation data
    step = 0.001
    for i in range(3):
        func = list_table_functions[i][0]
        interval = list_table_functions[i][1]
        x = prepare_uniform_interval(interval, step)
        table = func(x)
        np.savetxt(
            f"./solution_tables/validation_data/{func.__name__}.csv",
            table,
            delimiter=",",
        )

    step = 0.001
    for i in range(0, 11):
        func = list_sol_functions[i][0]
        interval = list_sol_functions[i][1]
        x = prepare_uniform_interval(interval, step)
        y = [func(i) for i in x]
        z = []
        for j in range(len(x)):
            if isinstance(y[j], tuple) or isinstance(y[j], list):
                z.append([x[j]] + list(y[j]))
            else:
                z.append([x[j], y[j]])
        table = np.array(z)
        # table = table.transpose()
        np.savetxt(
            f"./solution_tables/validation_data/{func.__name__}.csv",
            table,
            delimiter=",",
        )
