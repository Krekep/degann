import math
import random

from degann.networks.nn_code import decode, alph_n_full, alph_a


def choose_neighbor(method, **kwargs):
    return method(**kwargs)


def random_generate(**kwargs):
    block = random.randint(1, 4)
    code = ""
    for i in range(block):
        code += (
            alph_n_full[random.randint(0, len(alph_n_full) - 1)]
            + alph_a[random.randint(0, len(alph_a) - 1)]
        )
    epoch = random.randint(EpochParameter.min_epoch, EpochParameter.max_epoch)
    return CodeParameter(code), EpochParameter(epoch)


def generate_neighbor(parameters, distance: int = 150):
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
            if sign < 0.5:  # new epoch more than previous
                new_epoch = EpochParameter(
                    min(
                        random.randint(
                            epoch, int(EpochParameter.log_value**distance * epoch)
                        ),
                        EpochParameter.max_epoch,
                    )
                )
            else:  # new epoch less than previous
                new_epoch = EpochParameter(
                    max(
                        random.randint(
                            int(epoch / (EpochParameter.log_value**distance)), epoch
                        ),
                        EpochParameter.min_epoch,
                    )
                )
            distance -= abs(new_epoch.distance(epoch))
        elif branch < 1 and distance >= 1:  # change code
            chosen_block = random.randint(0, len(curr_code.blocks) - 1)

            command = random.randint(1, 5)
            match command:
                case 1:  # add block
                    if distance >= 9:
                        new_block = (
                            alph_n_full[
                                random.randint(
                                    0, min(int(distance - 9), len(alph_n_full) - 1)
                                )
                            ]
                            + alph_a[random.randint(0, len(alph_a) - 1)]
                        )
                        curr_code = CodeParameter("".join(curr_code.blocks) + new_block)
                case 2:  # increase block size
                    new_block = (
                        hex(
                            min(
                                int(curr_code.blocks[chosen_block][:-1], 16)
                                + random.randint(1, max(1, int(distance))),
                                15**CodeParameter.block_size,
                            )
                        )[2:]
                        + curr_code.blocks[chosen_block][-1]
                    )
                    new_block = (
                        "0" * (CodeParameter.block_size - len(new_block) + 1)
                        + new_block
                    )
                    curr_code.blocks[chosen_block] = new_block
                case 3:  # decrease size of block
                    new_block = (
                        hex(
                            max(
                                int(curr_code.blocks[chosen_block][:-1], 16)
                                - random.randint(1, max(1, int(distance))),
                                0,
                            )
                        )[2:]
                        + curr_code.blocks[chosen_block][-1]
                    )
                    new_block = (
                        "0" * (CodeParameter.block_size - len(new_block) + 1)
                        + new_block
                    )
                    curr_code.blocks[chosen_block] = new_block
                case 4:  # remove last block
                    if len(curr_code.blocks) > 1:
                        temp = CodeParameter(curr_code.value())
                        temp.blocks.pop()
                        if abs(temp.distance(curr_code)) < distance:
                            curr_code.blocks.pop()
                case 5:  # change activation for block
                    new_block = (
                        curr_code.blocks[chosen_block][:-1]
                        + alph_a[random.randint(0, len(alph_a) - 1)]
                    )
                    curr_code.blocks[chosen_block] = new_block
                case _:
                    pass
            distance -= abs(new_code.distance(curr_code))
            if distance < 0:
                distance += abs(new_code.distance(curr_code))
                temp_dist = abs(new_code.distance(curr_code))
                print(
                    "DEBUG",
                    new_code.blocks,
                    curr_code.blocks,
                    chosen_block,
                    command,
                    distance,
                    temp_dist,
                )
                distance -= temp_dist
            new_code = curr_code
        is_stop = random.randint(0, 2)
    return new_code, new_epoch


class MetaParameter:
    def distance(self, other):
        pass

    def value(self):
        pass


class CodeParameter(MetaParameter):
    block_size = 1
    exp_size = 10

    def __init__(self, s):
        if isinstance(s, CodeParameter):
            s = s.value()
        self.code = s
        self.blocks = [
            self.code[i : i + self.block_size + 1]
            for i in range(0, len(self.code), self.block_size + 1)
        ]

    def distance(self, other):
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

    def value(self):
        return "".join(self.blocks)

    def block_cost(self, block, offset=8):
        return sum(decode(block, block_size=self.block_size, offset=offset)[0])


class EpochParameter(MetaParameter):
    log_value = 1.1
    pow_scale = 3
    min_epoch = 100
    max_epoch = 700

    def __init__(self, epoch):
        self.epoch = epoch

    def distance(self, other):
        if isinstance(other, int):
            return math.log(
                self.epoch**EpochParameter.pow_scale, EpochParameter.log_value
            ) - math.log(other**EpochParameter.pow_scale, EpochParameter.log_value)
        return math.log(
            self.epoch**EpochParameter.pow_scale, EpochParameter.log_value
        ) - math.log(other.epoch**EpochParameter.pow_scale, EpochParameter.log_value)

    def value(self):
        return self.epoch
