from testlaunches import build_tables
from testlaunches import experiments

if __name__ == "__main__":
    names = []
    for func, _ in build_tables.list_sol_functions:
        names.append(func.__name__)

    for func, _ in build_tables.list_table_functions:
        names.append(func.__name__)

    data = experiments.prepare_tables(names, "data")
    val_data = experiments.prepare_tables(names, "validation_data")

    config = {}
    experiments.do_experiments(names, data, val_data, **config)
