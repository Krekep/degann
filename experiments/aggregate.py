from collections import defaultdict
from upload_to_google import work_sheets, table, sheets
import pandas as pd
import os


def _count_statistic(df_list, threshold):
    mean_nn = 0
    mean_time = 0
    count_success = 0
    for df in df_list:
        last_loss_value = df["loss"].iloc[-1]
        if last_loss_value <= threshold:
            count_success += 1
            mean_nn += len(df.index)
            mean_time += df["train_time"].sum()
    mean_nn = mean_nn / max(count_success, 1)
    mean_time = mean_time / max(count_success, 1)
    return mean_time, mean_nn, count_success, len(df_list)


if __name__ == "__main__":
    folder_path = "results"

    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)

    data = defaultdict(list)
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

            aggregated_file_name = (
                alg_name,
                threshold,
                func_name,
                data_size,
                loss,
                opt,
            )
            data[aggregated_file_name].append(df)

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

            aggregated_file_name = (
                alg_name,
                threshold,
                dist,
                temperature,
                func_name,
                data_size,
                loss,
                opt,
            )
            data[aggregated_file_name].append(df)

    for description, df_list in data.items():
        alg_name = description[0]
        if alg_name == "Random":
            threshold, func_name, data_size, loss, opt = description[1:]
            mean_time, mean_nn, count_success, all_launches = _count_statistic(
                df_list, float(threshold)
            )
            if all_launches < 20:
                mean_time = -1
                mean_nn = -1
                count_success = -all_launches / 100
            sheets[func_name][(alg_name)][loss][threshold][data_size][
                "Time(s)"
            ].value = round(mean_time, 2)
            sheets[func_name][(alg_name)][loss][threshold][data_size][
                "Count"
            ].value = round(mean_nn, 2)
            sheets[func_name][(alg_name)][loss][threshold][data_size][
                "Launch 20"
            ].value = int(100 * (count_success / all_launches))
        if alg_name == "Ann":
            (
                threshold,
                distance,
                temperature,
                func_name,
                data_size,
                loss,
                opt,
            ) = description[1:]
            mean_time, mean_nn, count_success, all_launches = _count_statistic(
                df_list, float(threshold)
            )
            if all_launches < 20:
                mean_time = -1
                mean_nn = -1
                count_success = -all_launches / 100
            sheets[func_name][(alg_name, temperature, distance)][loss][threshold][
                data_size
            ]["Time(s)"].value = round(mean_time, 2)
            sheets[func_name][(alg_name, temperature, distance)][loss][threshold][
                data_size
            ]["Count"].value = round(mean_nn, 2)
            sheets[func_name][(alg_name, temperature, distance)][loss][threshold][
                data_size
            ]["Launch 20"].value = int(100 * (count_success / all_launches))
    for key, sheet in sheets.items():
        cell_list = list()
        for _, sh in sheet.items():
            for _, lv in sh.items():
                for _, tv in lv.items():
                    for _, dsv in tv.items():
                        for _, cell in dsv.items():
                            cell_list.append(cell)
        table.worksheet(key.capitalize()).update_cells(cell_list)
