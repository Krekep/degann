import time
import os
import sys
sys.path.append("../")
from degann.networks.imodel import IModel
from degann.networks.cpp_utils import get_available_vectorized_levels

def val_test(input_size, shapes, act_funcs, output_size, test):
    shapes_size, vec_levels_size = len(shapes), 0
    vectorized_levels = {0: "none"}
    available_vectorized_levels = get_available_vectorized_levels()

    if "sse" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "sse"
    if "avx" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "avx"
    if "avx512f" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "avx512f"

    d = {"layers": "", "weights": "", "biases": ""}
    nn = IModel(input_size, shapes, output_size, act_funcs)
    vec_levels_size += 1
    for i in range(vec_levels_size):
        nn.export_to_cpp(f"test_{i}",
                         vectorized_level=vectorized_levels[i],
                         layers=d["layers"],
                         weights=d["weights"],
                         biases=d["biases"],
                         main="val_test")
        with open(f"test_{i}.cpp", "r") as f:
            addto = ""
            for line in f:
                if "float layer_0" in line:
                    addto = "layers"
                elif "float weight_0_1" in line:
                    addto = "weights"
                elif "float bias_1" in line:
                    addto = "biases"
                elif "for" in line or "vectorized" in line:
                    break
                if addto:
                    d[addto] += line

    if test == 1:
        results = open("./data/value_results.txt", 'w')
    results = open("./data/value_results.txt", 'a')

    for i in range(vec_levels_size):
        if i == 0:
            os.system(f"g++ -fno-tree-vectorize test_{i}.cpp -o test_{i}.exe")
            continue
        os.system(f"g++ -m{vectorized_levels[i]} test_{i}.cpp -o test_{i}.exe")

    results.write(f"test #{test}:\n")
    for i in range(vec_levels_size):
        results.write(f"vectorized level: {vectorized_levels[i]}, result: ")
        start_time = time.time()
        os.system(f"./test_{i}.exe")
        end_time = time.time()
        result = open("result.txt", 'r')
        results.write(result.read())
        result.close()
        results.write(f", time:{end_time - start_time}\n")

    for i in range(vec_levels_size):
        os.remove(f"./test_{i}.cpp")
        os.remove(f"./test_{i}.hpp")
        os.remove(f"./test_{i}.exe")
    os.remove("result.txt")
    results.close()

def time_test(input_size, shapes, act_funcs, output_size, test):
    shapes_size, vec_levels_size = len(shapes), 0
    vectorized_levels = {0: "none"}
    available_vectorized_levels = get_available_vectorized_levels()

    if "sse" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "sse"
    if "avx" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "avx"
    if "avx512f" in available_vectorized_levels:
        vec_levels_size += 1
        vectorized_levels[vec_levels_size] = "avx512f"

    d = {"layers": "", "weights": "", "biases": ""}
    nn = IModel(input_size, shapes, output_size, act_funcs)
    vec_levels_size += 1
    if test == 1:
        results = open("./data/time_results.txt", 'w')
    results = open("./data/time_results.txt", 'a')
    results.write(f"test #{test}:\n")
    for i in range(vec_levels_size):
        results.write(f"vectorized level: {vectorized_levels[i]}, time:")
        nn.export_to_cpp(f"test_{i}",
                         vectorized_level=vectorized_levels[i],
                         main="time_test")
        if i == 0:
            os.system(f"g++ -fno-tree-vectorize test_{i}.cpp -o test_{i}.exe")
        else:
            os.system(f"g++ -m{vectorized_levels[i]} test_{i}.cpp -o test_{i}.exe")
        os.system(f"./test_{i}.exe")
        result = open("result.txt", 'r')
        results.write(result.read())
        results.write("\n")
        result.close()

    for i in range(vec_levels_size):
        os.remove(f"./test_{i}.cpp")
        os.remove(f"./test_{i}.hpp")
        os.remove(f"./test_{i}.exe")
    os.remove(f"result.txt")
    results.close()

val_test(1, [150, 150, 150, 150, 150], ['linear', 'softsign', 'hard_sigmoid', 'softsign', 'relu'], 1, 1)
val_test(1, [200, 200, 200, 200, 200, 200], ['hard_sigmoid', 'relu', 'softsign', 'linear', 'softsign', 'softsign'], 1, 2)
val_test(1, [400, 400], ['linear', 'hard_sigmoid'], 1, 3)
val_test(1, [500, 500, 500, 500], ['relu', 'linear', 'hard_sigmoid', 'relu'], 1, 4)
time_test(1, [150, 150, 150, 150, 150], ['linear', 'softsign', 'hard_sigmoid', 'softsign', 'relu'], 1, 1)
time_test(1, [150, 150, 150, 150, 150], ['linear', 'softsign', 'hard_sigmoid', 'softsign', 'relu'], 1, 2)
time_test(1, [150, 150, 150, 150, 150], ['linear', 'softsign', 'hard_sigmoid', 'softsign', 'relu'], 1, 3)
time_test(1, [150, 150, 150, 150, 150], ['linear', 'softsign', 'hard_sigmoid', 'softsign', 'relu'], 1, 4)