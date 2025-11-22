from typing import Tuple
from .utils import update_random_generator
from degann.networks.topology.parameter_space import ParameterSpace


def random_search(
        input_size: int,
        output_size: int,
        data: tuple,
        params: ParameterSpace,
        iterations: int,
        threshold: float,
        max_iter: int = 1,
        val_data: tuple = None,
        logging=False,
        file_name: str = "",
        callbacks: list = None,
        verbose: bool = False,
        update_gen_cycle: int = 0,
) -> Tuple[float, int, str, str, dict]:
    best_net = None
    best_loss = 1e6
    best_epoch = None
    loss = None
    opt = None

    if iterations == -1:
        start_config = params.get_random_config()
        curr_loss, curr_val_loss, curr_nn = params.train(
            config=start_config,
            input_size=input_size,
            output_size=output_size,
            data=data,
            val_data=val_data,
            logging=logging,
            file_name=file_name,
            callbacks=callbacks,
        )
        i = 1
        while curr_loss > threshold and i != max_iter:
            if verbose:
                print(
                    f"Random search until less than threshold. Last loss = {curr_loss}. Iterations = {i}"
                )
            update_random_generator(i, cycle_size=update_gen_cycle)
            config = params.get_random_config()

            curr_loss, curr_val_loss, curr_nn = params.train(
                config=config,
                input_size=input_size,
                output_size=output_size,
                data=data,
                val_data=val_data,
                logging=logging,
                file_name=file_name,
                callbacks=callbacks,
            )
            i += 1
            if curr_loss < best_loss:
                best_epoch = config["num_epoch"]
                best_net = curr_nn
                best_loss = curr_loss
                loss = config["loss_func"]
                opt = config["optimizer"]
    else:
        for i in range(iterations):
            update_random_generator(i, cycle_size=update_gen_cycle)
            config = params.get_random_config()

            curr_loss, curr_val_loss, curr_nn = params.train(
                config=config,
                input_size=input_size,
                output_size=output_size,
                data=data,
                val_data=val_data,
                logging=logging,
                file_name=file_name,
                callbacks=callbacks,
            )

            if curr_loss < best_loss:
                best_epoch = config["num_epoch"]
                best_net = curr_nn
                best_loss = curr_loss
                loss = config["loss_func"]
                opt = config["optimizer"]
    return best_loss, best_epoch, loss, opt, best_net
