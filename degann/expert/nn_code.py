from degann.networks import imodel

act_to_hex = {
    "elu": "0",
    "relu": "1",
    "gelu": "2",
    "selu": "3",
    "exponential": "4",
    "linear": "5",
    "sigmoid": "6",
    "hard_sigmoid": "7",
    "swish": "8",
    "tanh": "9",
    "softplus": "a",
    "softsign": "b",
    "parabolic": "c",
}
reverse_act_to_hex = {v: k for k, v in act_to_hex.items()}

alph_n_full = "0123456789abcdef"
alph_n_div3 = "0369cf"
alph_n_div2 = "02468ace"
alph_n_div4 = "048c"

alphabet_activations_cut = "0689"
alphabet_activations = "0123456789abc"


def encode(nn: imodel.IModel, offset: int = None) -> str:
    blocks = nn.get_shape
    activations = nn.get_activations
    res = ""

    offset = min(blocks) if offset is None else offset
    for layer, act in zip(blocks, activations):
        curr = hex(layer - offset)[2:] + act_to_hex[act]
        res += curr

    return res


def decode(s: str, block_size: int = 1, offset: int = 0):
    blocks = []
    activations = []
    for block in range(0, len(s), block_size + 1):
        temp = s[block : block + block_size]
        blocks.append(int(temp, 16) + offset)
        activations.append(reverse_act_to_hex[s[block + block_size]])
    return blocks, activations
