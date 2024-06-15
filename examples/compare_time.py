import os
import sys

sys.path.append("../")
from degann.networks.imodel import IModel
from degann.networks.cpp_utils import get_available_vectorized_levels


def compile(path_to_compiler: str, flags: str, test: int):
    os.system(f"g++ -fno-tree-vectorize test_{test}.cpp -o test_{test}.exe")


def run_file(path: str):
    os.system(path)


def remove_files(path):
    os.remove(f"{path}.cpp")
    os.remove(f"{path}.hpp")
    os.remove(f"{path}.exe")


def time_test(input_size, shapes, act_funcs, output_size, test):
    shapes_size, vec_levels_size = len(shapes), 0
    vectorized_levels = {0: "none"}  # the newer instruction, the higher number
    available_vectorized_levels = get_available_vectorized_levels()
    res = []
    if "sse" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "sse"
    if "avx" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "avx"
    if "avx512f" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "avx512f"

    nn = IModel(input_size, shapes, output_size, act_funcs)
    vec_levels_size += 1
    if test == 1:
        results = open("./time_results.txt", "w")
    results = open("./time_results.txt", "a")
    results.write(f"test #{test}:\n")
    for i in range(vec_levels_size):
        results.write(f"vectorized level: {vectorized_levels[i]}, time:")
        # generate code
        nn.export_to_cpp(
            f"test_{i}", vectorized_level=vectorized_levels[i], main="time_test"
        )
        # compile code
        if i == 0:
            compile("g++", "-fno-tree-vectorize", i)
        else:
            compile("g++", f"-m{vectorized_levels[i]}", i)
        # run compiled code and record runtime
        run_file(f"./test_{i}.exe")
        result = open("result.txt", "r")
        res.append(result.read())
        results.write(res[-1])
        results.write("\n")
        result.close()

    for i in range(vec_levels_size):  # remove files
        remove_files(f"./test_{i}")
    os.remove(f"result.txt")
    results.close()


time_test(
    1,
    [150, 150, 150, 150, 150],
    ["linear", "softsign", "hard_sigmoid", "softsign", "relu"],
    1,
    1,
)
# time_test(1, [200, 200, 200, 200, 200, 200], ["hard_sigmoid", "relu", "softsign", "linear", "softsign", "softsign"], 1, 2)
# time_test(1, [400, 400], ["linear", "hard_sigmoid"], 1, 3)
# time_test(1, [500, 500, 500, 500], ["relu", "linear", "hard_sigmoid", "relu"], 1, 4)
