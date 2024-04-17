from collections import defaultdict
import pandas as pd
import os
from collections import namedtuple


def import_results(
    folder_path: str = "results",
) -> defaultdict[namedtuple, list[pd.DataFrame]]:
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)

    AggregatedFileName = namedtuple(
        "AggregatedFileName",
        [
            "alg_name",
            "threshold",
            "dist",
            "temperature",
            "func_name",
            "data_size",
            "loss",
            "opt",
        ],
    )
    data: defaultdict[namedtuple, list[pd.DataFrame]] = defaultdict(list)
    for file_name in file_names:
        metaparams = file_name.split("_")

        if metaparams[0].__contains__("Random"):
            alg_name, threshold = metaparams[0][:6], metaparams[0][6:-1]
            func_name = metaparams[1]
            iter = metaparams[2]
            data_size = metaparams[3]
            epoch = metaparams[4]  # always zero
            loss = metaparams[5]
            opt = metaparams[6]

            df = pd.read_csv(
                folder_path + "/" + file_name,
                names=[
                    "shape",
                    "activations",
                    "code",
                    "epoch",
                    "opt",
                    "loss_func",
                    "loss",
                    "val_loss",
                    "train_time",
                ],
            )

            afn = AggregatedFileName(
                alg_name,
                threshold,
                0,
                0,
                func_name,
                data_size,
                loss,
                opt,
            )
            data[afn].append(df)

        if metaparams[0].__contains__("Ann"):
            alg_name, threshold, dist, temperature = metaparams[0].split(",")
            threshold = threshold[5:]
            func_name = metaparams[1]
            iter = metaparams[2]
            data_size = metaparams[3]
            epoch = metaparams[4]  # always zero
            loss = metaparams[5]
            opt = metaparams[6]

            df = pd.read_csv(
                folder_path + "/" + file_name,
                names=[
                    "shape",
                    "activations",
                    "code",
                    "epoch",
                    "opt",
                    "loss_func",
                    "loss",
                    "val_loss",
                    "train_time",
                ],
            )

            afn = AggregatedFileName(
                alg_name,
                threshold,
                dist,
                temperature,
                func_name,
                data_size,
                loss,
                opt,
            )
            data[afn].append(df)

    return data
