from typing import Tuple, Optional, List
from degann.networks.imodel import IModel
from degann.networks.topology.abstracts import NetConfig
from degann.search_algorithms.utils import update_random_generator, log_to_file


def train(
        config: NetConfig,
        num_epochs: int,
        data: tuple,
        repeat: int = 1,
        update_gen_cycle: int = 0,
        val_data: Optional[tuple] = None,
        logging: bool = False,
        file_name: str = "",
        callbacks: Optional[list] = None,
) -> Tuple[float, float, dict]:
    """
    Train and evaluate model with given configuration.

    Parameters
    ----------
    config: NetConfig
        Configuration for training the model.
    num_epochs: int
        Number of training epochs.
    data: Tuple[Any, Any]
        Training data.
    repeat: int
        Number of training repetitions.
    update_gen_cycle: int
        Cycle size for random generator update.
    val_data: Tuple[Any, Any]
        Validation data.
    logging: bool
        Flag to enable logging.
    file_name: str
        Name for log files.
    callbacks: List[Any]
        List of training callbacks.

    Returns
    -------
    best_loss: float
        Best training loss achieved.
    best_val_loss: float
        Best validation loss achieved.
    best_net: dict
        Dictionary representation of the best network.
    """
    best_net = None
    best_loss = 1e6
    best_val_loss = 1e6

    for i in range(repeat):
        update_random_generator(i, cycle_size=update_gen_cycle)

        nn = IModel(config=config, net_type=config.net_type)
        nn.compile(**config.get_compile_kwargs)

        temp_his = nn.train(
            data[0],
            data[1],
            epochs=num_epochs,
            verbose=0,
            callbacks=callbacks,
            validation_data=val_data,
        )

        curr_loss = temp_his.history["loss"][-1]
        curr_val_loss = 1e6

        if val_data is not None:
            val_res = nn.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)
            curr_val_loss = val_res["loss"]

        if logging:
            history = config.to_dict()

            history["loss"] = [curr_loss]
            history["validation_loss"] = [curr_val_loss]
            history["epoch"] = [num_epochs]
            history["train_time"] = [nn.network.trained_time["train_time"]]

            fn = f"{file_name}_{config.net_type}_{len(data[0])}_{num_epochs}_{config.get_loss_func}_{config.get_optimizer}"
            log_to_file(history, fn)

        if curr_loss < best_loss:
            best_loss = curr_loss
            best_val_loss = curr_val_loss
            best_net = nn.to_dict()

    return best_loss, best_val_loss, best_net
