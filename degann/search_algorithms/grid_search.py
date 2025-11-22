from degann.networks.topology.parameter_space import ParameterSpace
from datetime import datetime
from typing import Tuple
from degann.networks.callbacks import MeasureTrainTime


def grid_search(
        input_size: int,
        output_size: int,
        data: tuple,
        params: ParameterSpace,
        val_data: tuple = None,
        logging=False,
        file_name: str = "",
        verbose=False,
) -> Tuple[float, int, str, str, dict]:
    best_net: dict = dict()
    best_loss: float = 1e6
    best_epoch: int = 0
    best_loss_func: str = ""
    best_opt: str = ""
    time_viewer = MeasureTrainTime()
    configs = params.create_parameter_space()
    for config in configs:
        if verbose:
            print(len(config["block_size"]), datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
        curr_loss, curr_val_loss, curr_nn = params.train(
            config=config,
            input_size=input_size,
            output_size=output_size,
            data=data,
            val_data=val_data,
            logging=logging,
            file_name=file_name,
            callbacks=[time_viewer],
        )

        if best_loss > curr_loss:
            best_net = curr_nn
            best_loss = curr_loss
            best_epoch = config["num_epoch"]
            best_loss_func = config["loss_func"]
            best_opt = config["optimizer"]
    return best_loss, best_epoch, best_loss_func, best_opt, best_net
