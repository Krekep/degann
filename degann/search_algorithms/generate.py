import random
from typing import List, Tuple


def generate_neighbour(
    layer_sizes: List[int],
    activation_funcs: List[str],
    optimizer: str,
    num_epochs: int,
    all_layers: List[int],
    all_activations: List[str],
    all_optimizers: List[str],
    min_depth: int,
    max_depth: int,
    min_epoch: int,
    max_epoch: int,
    distance: float = 150.0,
) -> Tuple[List[int], List[str], str, int]:
    """
    Generator of a point in the neighbourhood of the current one in the parameter space.

    Parameters
    ----------
    layer_sizes : List[int]
        Current number of neurons in each hidden layer.
    activation_funcs : List[str]
        Current activation function for each hidden layer.
    optimizer : str
        Current optimizer name.
    num_epochs : int
        Current number of training epochs.
    all_layers : List[int]
        Pool of allowed layer sizes for mutation.
    all_activations : List[str]
        Pool of allowed activation functions for mutation.
    all_optimizers : List[str]
        Pool of allowed optimizers for mutation.
    min_depth : int
        Minimum allowed number of hidden layers.
    max_depth : int
        Maximum allowed number of hidden layers.
    min_epoch : int
        Minimum allowed number of training epochs.
    max_epoch : int
        Maximum allowed number of training epochs.
    distance : float, optional
        Proxy for mutation strength. Higher values permit larger jumps in the
        search space. Decreases as mutations are applied.

    Returns
    -------
    Tuple[List[int], List[str], str, int]
        Tuple containing:
        - new_layers: Modified hidden layer sizes.
        - new_activations: Modified activation functions for hidden layers.
        - new_optimizer: Possibly updated optimizer name.
        - new_epochs: Possibly updated number of training epochs.
    """
    new_layers = layer_sizes.copy()
    new_activations = activation_funcs.copy()
    new_optimizer = optimizer
    new_epochs = num_epochs
    log_value = 1.05
    is_stop = 0

    while distance > 0 and is_stop == 0:
        branch = random.random()

        if branch < 0.33:  # change epoch
            sign = random.random()
            if sign < 0.66:  # new epoch more than previous
                multiplier = log_value ** min(distance, 30)
                new_epochs = min(
                    random.randint(num_epochs, int(multiplier * num_epochs)),
                    max_epoch,
                )
            else:  # new epoch less than previous
                divisor = log_value ** min(distance, 30)
                new_epochs = max(
                    random.randint(int(num_epochs / divisor), num_epochs),
                    min_epoch,
                )
            distance -= abs(new_epochs - num_epochs) / 10

        elif branch < 1 and distance >= 1:  # change network topology
            chosen_layer = random.randint(0, len(new_layers) - 1)
            command = random.randint(1, 6)
            match command:
                case 1:  # add layer
                    if len(new_layers) < max_depth:
                        new_size = random.choice(all_layers)
                        new_act = random.choice(all_activations)
                        new_layers.append(new_size)
                        new_activations.insert(-1, new_act)
                        distance -= 10

                case 2:  # increase layer size
                    available = [s for s in all_layers if s > new_layers[chosen_layer]]
                    if available:
                        new_layers[chosen_layer] = random.choice(available)
                        distance -= 5

                case 3:  # decrease layer size
                    available = [s for s in all_layers if s < new_layers[chosen_layer]]
                    if available:
                        new_layers[chosen_layer] = random.choice(available)
                        distance -= 5

                case 4:  # remove last layer
                    if len(new_layers) > min_depth:
                        new_layers.pop()
                        new_activations.pop()
                        distance -= 10

                case 5:  # change activation for layer
                    if len(new_activations) > 1:
                        chosen_act = random.randint(0, len(new_activations) - 2)
                        new_activations[chosen_act] = random.choice(all_activations)
                        distance -= 5
                case 6:  # change optimizer
                    available = [o for o in all_optimizers if o != new_optimizer]
                    if available:
                        new_optimizer = random.choice(available)
                        distance -= 5

        is_stop = random.randint(0, 3)

    if (
        new_layers == layer_sizes
        and new_activations == activation_funcs
        and new_epochs == num_epochs
        and new_optimizer == optimizer
    ):
        idx = random.randint(0, len(new_layers) - 1)
        new_layers[idx] = random.choice(all_layers)

    return new_layers, new_activations, new_optimizer, new_epochs
