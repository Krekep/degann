from src.testlaunches import experiments, build_tables

if __name__ == "__main__":
    names = []
    for func, _ in build_tables.list_sol_functions:
        names.append(func.__name__)

    for func, _ in build_tables.list_table_functions:
        names.append(func.__name__)

    data = experiments.prepare_tables(names, "data")
    val_data = experiments.prepare_tables(names, "validation_data")

    config = {
        "epochs": [3],
        "optimizers": ["Adam"],
        "rates": [1e-3],
        "net_shapes": [[5, 5], [10]],
        "activations": ["relu"],
        "use_rand_net": False,
    }
    experiments.do_experiments(names, data, val_data, **config)
