from itertools import product

from .nn_code import alph_n_full, alphabet_activations, decode
from degann.networks import imodel
from degann.search_algorithms.generate import random_generate
from .utils import update_random_generator, log_to_file


def random_search(
    in_size,
    out_size,
    data,
    opt,
    loss,
    iterations,
    min_epoch=100,
    max_epoch=700,
    val_data=None,
    callbacks=None,
    logging=False,
    nn_min_length: int = 1,
    nn_max_length: int = 6,
    nn_alphabet: list[str] = [
        "".join(elem) for elem in product(alph_n_full, alphabet_activations)
    ],
    alphabet_block_size: int = 1,
    alphabet_offset: int = 8,
    update_gen_cycle: int = 0,
    file_name: str = "",
):
    best_net = None
    best_loss = 1e6
    best_epoch = None
    for i in range(iterations):
        history = dict()
        update_random_generator(i, cycle_size=update_gen_cycle)
        gen = random_generate(
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            min_length=nn_min_length,
            max_length=nn_max_length,
            alphabet=nn_alphabet,
        )

        b, a = decode(
            gen[0].value(), block_size=alphabet_block_size, offset=alphabet_offset
        )
        curr_best = imodel.IModel(in_size, b, out_size, a + ["linear"])
        curr_best.compile(optimizer=opt, loss_func=loss)
        curr_epoch = gen[1].value()
        hist = curr_best.train(
            data[0], data[1], epochs=curr_epoch, verbose=0, callbacks=callbacks
        )
        curr_loss = hist.history["loss"][-1]
        curr_val_loss = (
            curr_best.evaluate(val_data[0], val_data[1], verbose=0, return_dict=True)[
                "loss"
            ]
            if val_data is not None
            else None
        )

        history["shapes"] = [curr_best.get_shape]
        history["activations"] = [a]
        history["code"] = [gen[0].value()]
        history["epoch"] = [gen[1].value()]
        history["optimizer"] = [opt]
        history["loss function"] = [loss]
        history["loss"] = [curr_loss]
        history["validation loss"] = [curr_val_loss]
        history["train_time"] = [curr_best.network.trained_time["train_time"]]
        if logging:
            fn = f"{file_name}_{len(data[0])}_0_{loss}_{opt}"
            log_to_file(history, fn)

        if curr_loss < best_loss:
            best_epoch = curr_epoch
            best_net = curr_best.to_dict()
            best_loss = curr_loss
    return best_loss, best_epoch, loss, opt, best_net


def random_search_endless(
    in_size,
    out_size,
    data,
    opt,
    loss,
    threshold,
    max_iter=-1,
    min_epoch=100,
    max_epoch=700,
    val_data=None,
    callbacks=None,
    nn_min_length: int = 1,
    nn_max_length: int = 6,
    nn_alphabet: list[str] = [
        "".join(elem) for elem in product(alph_n_full, alphabet_activations)
    ],
    alphabet_block_size: int = 1,
    alphabet_offset: int = 8,
    logging=False,
    file_name: str = "",
    verbose=False,
):
    nn_loss, nn_epoch, loss_f, opt_n, net = random_search(
        in_size,
        out_size,
        data,
        opt,
        loss,
        1,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
        val_data=val_data,
        nn_min_length=nn_min_length,
        nn_max_length=nn_max_length,
        nn_alphabet=nn_alphabet,
        alphabet_block_size=alphabet_block_size,
        alphabet_offset=alphabet_offset,
        callbacks=callbacks,
        logging=logging,
        file_name=file_name,
    )
    i = 1
    best_net = net
    best_loss = nn_loss
    best_epoch = nn_epoch
    while nn_loss > threshold and i != max_iter:
        if verbose:
            print(
                f"Random search until less than threshold. Last loss = {nn_loss}. Iterations = {i}"
            )
        nn_loss, nn_epoch, loss_f, opt_n, net = random_search(
            in_size,
            out_size,
            data,
            opt,
            loss,
            1,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            val_data=val_data,
            nn_min_length=nn_min_length,
            nn_max_length=nn_max_length,
            nn_alphabet=nn_alphabet,
            alphabet_block_size=alphabet_block_size,
            alphabet_offset=alphabet_offset,
            callbacks=callbacks,
            logging=logging,
            file_name=file_name,
        )
        i += 1
        if nn_loss < best_loss:
            best_net = net
            best_loss = nn_loss
            best_epoch = nn_epoch
    return best_loss, best_epoch, loss, opt, best_net, i
