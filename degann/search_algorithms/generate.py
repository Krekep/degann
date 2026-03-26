import random
from typing import List, Tuple


def generate_neighbour(
    layer_sizes: List[int],
    activation_funcs: List[str],
    num_epochs: int,
    all_layers: List[int],
    all_activations: List[str],
    min_depth: int,
    max_depth: int,
    min_epoch: int,
    max_epoch: int,
    distance: float = 150.0,
) -> Tuple[List[int], List[str], int]:
    """
    Generator of a point in the neighbourhood of the current one in the parameter space.
    """
    new_layers = layer_sizes.copy()
    new_activations = activation_funcs.copy()
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

            command = random.randint(1, 5)
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

        is_stop = random.randint(0, 3)

    if (
        new_layers == layer_sizes
        and new_activations == activation_funcs
        and new_epochs == num_epochs
    ):
        idx = random.randint(0, len(new_layers) - 1)
        new_layers[idx] = random.choice(all_layers)

    return new_layers, new_activations, new_epochs


def mutate_block_sizes(
    block_sizes: List[int],
    layer_sizes: List[int],
    min_depth: int,
    max_depth: int,
    mutation_prob: float = 0.3,
) -> List[int]:
    new_blocks = block_sizes.copy()

    if random.random() < mutation_prob:
        if len(new_blocks) < max_depth and random.random() < 0.5:
            new_blocks.append(random.choice(layer_sizes))
        elif len(new_blocks) > min_depth:
            new_blocks.pop()

    for i in range(len(new_blocks)):
        if random.random() < mutation_prob:
            new_blocks[i] = random.choice(layer_sizes)
    return new_blocks


def mutate_activations(
    activations: List[str],
    all_activations: List[str],
    mutation_prob: float = 0.3,
) -> List[str]:
    new_activations = activations.copy()
    for i in range(len(new_activations) - 1):
        if random.random() < mutation_prob:
            new_activations[i] = random.choice(all_activations)
    return new_activations
