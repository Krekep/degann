import math
import random
from typing import Union, Callable

from degann.search_algorithms.nn_code import decode, alphabet_activations


class MetaParameter:
    """
    Abstract class for parameters in parameter space of neural networks
    """

    def distance(self, other):
        pass

    def value(self):
        pass


class CodeParameter(MetaParameter):
    """
    Topology parameter --- size of layers with activations
    """

    block_size = 1
    exp_size = 10

    def __init__(self, s: Union[str, "CodeParameter"]):
        if isinstance(s, CodeParameter):
            s = s.value()
        self.code = s
        self.blocks = [
            self.code[i : i + self.block_size + 1]
            for i in range(0, len(self.code), self.block_size + 1)
        ]

    def distance(self, other) -> float:
        """
        Distance between `self` topology and `other` topology

        Parameters
        ----------
        other: CodeParameter | str
            other neural network topology

        Returns
        -------
        distance: float
            distance between topologies
        """
        if isinstance(other, str):
            other = CodeParameter(other)
        val_a = 0
        act_a = []
        for block in self.blocks:
            dec = decode(block, block_size=self.block_size, offset=8)
            val_a += sum(dec[0])
            act_a.append(*dec[1])

        val_b = 0
        act_b = []
        for block in other.blocks:
            dec = decode(block, block_size=self.block_size, offset=8)
            val_b += sum(dec[0])
            act_b.append(*dec[1])

        diff = 0
        for i in range(min(len(act_a), len(act_b))):
            if act_a[i] != act_b[i]:
                diff += 1
        diff += CodeParameter.exp_size ** abs(len(act_a) - len(act_b))
        return abs(val_a - val_b) + diff

    def value(self) -> str:
        """
        Present CodeParameter as string code

        Returns
        -------
        str_topology: str
            topology as str
        """
        return "".join(self.blocks)


class EpochParameter(MetaParameter):
    """
    Epoch parameter --- count of epoch in training
    """

    log_value = 1.1
    pow_scale = 3

    def __init__(self, epoch: int):
        self.epoch = epoch

    def distance(self, other: Union[int, "EpochParameter"]) -> float:
        """
        Distance between `self` epoch and `other` epoch

        Parameters
        ----------
        other: EpochParameter | int
            other neural network topology

        Returns
        -------
        distance: float
            distance between epochs
        """
        if isinstance(other, int):
            return math.log(
                self.epoch**EpochParameter.pow_scale, EpochParameter.log_value
            ) - math.log(other**EpochParameter.pow_scale, EpochParameter.log_value)
        return math.log(
            self.epoch**EpochParameter.pow_scale, EpochParameter.log_value
        ) - math.log(other.epoch**EpochParameter.pow_scale, EpochParameter.log_value)

    def value(self) -> int:
        """
        Present EpochParameter as int value

        Returns
        -------
        str_topology: int
            epoch as int value
        """
        return self.epoch


def choose_neighbor(method: Callable, **kwargs):
    """
    Wrapper that returns a method with kwargs applied to it

    Parameters
    ----------
    method: Callable
       Method for determining the distance to another neural network
    kwargs

    Returns
    -------
    method: Callable
        Method with applied kwargs
    """
    return method(**kwargs)


def random_generate(
    alphabet: list[str],
    min_epoch: int = 100,
    max_epoch: int = 700,
    min_length: int = 1,
    max_length: int = 6,
) -> tuple[CodeParameter, EpochParameter]:
    """
    Random point generator in the parameter space of neural networks

    Parameters
    ----------
    alphabet: list[str]
        Alphabet defining possible layers of a neural network
    min_epoch: int
        Minimum number of training epochs
    max_epoch: int
        Maximum number of training epochs
    min_length: int
        Minimum count of hidden layers in neural network
    max_length: int
        Maximum count of hidden layers in neural network

    Returns
    -------
    point: tuple[CodeParameter, EpochParameter]
        random generated point
    """
    block = random.randint(min_length, max_length)
    code = ""

    for i in range(block):
        code += alphabet[random.randint(0, len(alphabet) - 1)]
    epoch = random.randint(min_epoch, max_epoch)
    return CodeParameter(code), EpochParameter(epoch)


def generate_neighbor(
    alphabet: list[str],
    parameters: tuple[str, int],
    distance: int = 150,
    min_epoch: int = 100,
    max_epoch: int = 700,
    min_length: int = 1,
    max_length: int = 6,
):
    """
    Generator of a point in the neighborhood of the current one in the parameter space of neural networks

    Parameters
    ----------
    alphabet: list[str]
        Alphabet defining possible layers of a neural network
    parameters: tuple[str, int]
        Start point
    distance: int
        Maximum distance from the starting point
    min_epoch: int
        Minimum number of training epochs
    max_epoch: int
        Maximum number of training epochs
    min_length: int
        Minimum count of hidden layers in neural network
    max_length: int
        Maximum count of hidden layers in neural network

    Returns
    -------
    point: tuple[CodeParameter, EpochParameter]
        neighbor of start point
    """
    code = parameters[0]
    epoch = parameters[1]
    is_stop = 0
    new_epoch = EpochParameter(epoch)
    new_code = CodeParameter(code)

    while distance > 0 and is_stop == 0:
        branch = random.random()
        curr_code = CodeParameter(new_code.value())
        if branch < 0.33:  # change epoch
            sign = random.random()
            if sign < 0.66:  # new epoch more than previous
                new_epoch = EpochParameter(
                    min(
                        random.randint(
                            epoch,
                            int(EpochParameter.log_value ** min(distance, 70) * epoch),
                        ),  # need to rewrite this formula
                        max_epoch,
                    )
                )
            else:  # new epoch less than previous
                new_epoch = EpochParameter(
                    max(
                        random.randint(
                            int(
                                epoch / (EpochParameter.log_value ** min(distance, 70))
                            ),
                            epoch,
                        ),  # need to rewrite this formula
                        min_epoch,
                    )
                )
            distance -= abs(new_epoch.distance(epoch))
        elif branch < 1 and distance >= 1:  # change code of neural network
            chosen_block = random.randint(0, len(curr_code.blocks) - 1)

            command = random.randint(1, 5)
            match command:
                case 1:  # add block
                    if len(curr_code.blocks) < max_length:
                        new_block = alphabet[random.randint(0, len(alphabet) - 1)]
                        curr_code = CodeParameter("".join(curr_code.blocks) + new_block)
                case 2:  # increase block size
                    current_block_size = int(curr_code.blocks[chosen_block][:-1], 16)
                    max_block_size = 15**CodeParameter.block_size
                    max_block_increase = min(max_block_size, int(distance))
                    new_block = (
                        hex(current_block_size + random.randint(0, max_block_increase))[
                            2:
                        ]
                        + curr_code.blocks[chosen_block][-1]
                    )
                    new_block = (
                        "0" * (CodeParameter.block_size - len(new_block) + 1)
                        + new_block
                    )
                    if new_block in alphabet:
                        curr_code.blocks[chosen_block] = new_block
                case 3:  # decrease size of block
                    current_block_size = int(curr_code.blocks[chosen_block][:-1], 16)
                    min_block_size = 0
                    max_block_decrease = min(current_block_size, int(distance))
                    new_block = (
                        hex(
                            max(
                                current_block_size
                                - random.randint(0, max_block_decrease),
                                min_block_size,
                            )
                        )[2:]
                        + curr_code.blocks[chosen_block][-1]
                    )
                    new_block = (
                        "0" * (CodeParameter.block_size - len(new_block) + 1)
                        + new_block
                    )
                    if new_block in alphabet:
                        curr_code.blocks[chosen_block] = new_block
                case 4:  # remove last block
                    if len(curr_code.blocks) > min_length:
                        temp = CodeParameter(curr_code.value())
                        temp.blocks.pop()
                        if abs(temp.distance(curr_code)) < distance:
                            curr_code.blocks.pop()
                case 5:  # change activation for block
                    new_block = (
                        curr_code.blocks[chosen_block][:-1]
                        + alphabet_activations[
                            random.randint(0, len(alphabet_activations) - 1)
                        ]
                    )
                    if new_block in alphabet:
                        curr_code.blocks[chosen_block] = new_block
                case _:
                    pass
            distance -= abs(new_code.distance(curr_code))
            if distance < 0:
                distance += abs(new_code.distance(curr_code))
                temp_dist = abs(new_code.distance(curr_code))
                # print(
                #     "DEBUG",
                #     new_code.blocks,
                #     curr_code.blocks,
                #     chosen_block,
                #     command,
                #     distance,
                #     temp_dist,
                # )
                distance -= temp_dist
            new_code = curr_code
        is_stop = random.randint(0, 3)
    return new_code, new_epoch
