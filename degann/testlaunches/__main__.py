from degann.testlaunches import experiments, build_tables

if __name__ == "__main__":
    names = []
    for func, _ in build_tables.list_sol_functions:
        names.append(func.__name__)

    for func, _ in build_tables.list_table_functions:
        names.append(func.__name__)

    data = experiments.prepare_tables(names, "data")
    val_data = experiments.prepare_tables(names, "validation_data")

    config = {
        "epochs": [50, 100, 200],
        "optimizers": ["Adam", "SGD", "RMSprop"],
        "loss_functions": [
            "Huber",
            "MeanSquaredError",
            "MeanAbsolutePercentageError",
            "MaxAbsoluteDeviation",
        ],
        "rates": [1e-3],
        "net_shapes": [
            [32, 16, 8, 4],
            [4, 8, 16, 32],
            [10, 10, 10, 10, 10, 10],
            [80, 80, 80],
        ],
        # "activations": ["relu"],
        "use_rand_net": False,
    }
    experiments.do_experiments(names, data, val_data, **config)
