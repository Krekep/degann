from typing import Union, Tuple, List


def array1d_creator(elem_type: str):
    """
    Return function for creating cpp one dimensional arrays (e.g. float[]) with specified type

    Parameters
    ----------
    elem_type:
        type of arrays

    Returns
    -------
    creator: Callable
        function for create arrays
    """

    def array1d_spec_type_creator(name: str, size: int, initial_value=None) -> str:
        """
        Create string representation of cpp one dimensional array

        Parameters
        ----------
        name: str
            array name
        size: int
            array size
        initial_value: list
            initial value for array. Can be None, list of values (len should be equal to size) or single value

        Returns
        -------
        array: str
        """
        res = f"{elem_type} {name}[{size}]"
        if initial_value is not None:
            if isinstance(initial_value, list):
                if len(initial_value) == size:
                    res += " = {" + ", ".join(map(str, initial_value)) + "}"
                else:
                    raise Exception("Incompatibility size of arrays")
            else:
                res += " = {" + ", ".join([str(initial_value)] * size) + "}"
        res += ";\n"
        return res

    return array1d_spec_type_creator


def array1d_heap_creator(elem_type: str):
    def array1d_heap_spec_type_creator(name: str, size: int) -> str:
        res = (
            f"{elem_type}* {name} = ({elem_type}*)malloc({size} * sizeof({elem_type}))"
        )
        res += ";\n"
        return res

    return array1d_heap_spec_type_creator


def array2d_creator(elem_type: str):
    """
    Return function for creating cpp one dimensional arrays (e.g. float[]) with specified type

    Parameters
    ----------
    elem_type:
        type of arrays

    Returns
    -------
    creator: Callable
        function for create arrays
    """

    def array2d_spec_type_creator(
        name: str, size_x: int, size_y: int, initial_value=None, reverse=False
    ) -> str:
        """
        Create string representation of cpp one dimensional array

        Parameters
        ----------
        name: str
            array name
        size_x: int
            first dimension size
        size_y: int
            second dimension  size
        initial_value: list
            initial value for array. Can be None, list[list] of values (len should be equal to size) or single value

        Returns
        -------
        array: str
        """
        res = f"{elem_type} {name}[{size_x}][{size_y}]"
        if initial_value is not None:
            if isinstance(initial_value, list):
                if len(initial_value) == size_x and len(initial_value[0]) == size_y:
                    if reverse:
                        initial_value = [
                            [initial_value[i][j] for i in range(size_x)]
                            for j in range(size_y - 1, -1, -1)
                        ]
                    temp = "{ "
                    for i in range(size_x - 1):
                        temp += "{" + ", ".join(map(str, initial_value[i])) + "}, \n"
                    temp += "{" + ", ".join(map(str, initial_value[-1])) + "} }"
                    res += " = " + temp
                else:
                    raise Exception("Incompatibility size of arrays")
            else:
                temp = "{ "
                for i in range(size_x - 1):
                    temp += "{" + ", ".join([str(initial_value)] * size_y) + "}, \n"
                temp += "{" + ", ".join([str(initial_value)] * size_y) + "} }"
                res += " = " + temp
        res += ";\n"
        return res

    return array2d_spec_type_creator


def vector1d_creator(elem_type: str):
    def vector1d_spec_type_creator(name: str, size: int, initial_value=0) -> str:
        return f"std::vector<{elem_type}> {name}({size}, {initial_value});\n"

    return vector1d_spec_type_creator


def vector2d_creator(elem_type: str):
    def vector2d_spec_type_creator(
        name: str, size_x: int, size_y: int, initial_value=0
    ) -> str:
        return f"std::vector<std::vector<{elem_type}>> {name}({size_x}, std::vector<{type}>({size_y}, {initial_value}));\n"

    return vector2d_spec_type_creator


def transform_1dvector_to_array(
    elem_type: str, size: Union[int, str], vec_name: str, arr_name: str
) -> str:
    res = f"""
    {elem_type} {arr_name}[{size}];

    for (int i = 0; i < {size}; i++)
    {{
        {arr_name}[i] = {vec_name}[i];
    }}\n
    """
    return res


def transform_1darray_to_vector(
    elem_type: str, size: Union[int, str], vec_name: str, arr_name: str
) -> str:
    res = f"""
    std::vector<{elem_type}> {vec_name}({size});

    for (int i = 0; i < {size}; i++)
    {{
        {vec_name}[i] = {arr_name}[i];
    }}\n
    """
    return res


def copy_1darray_to_array(size: Union[int, str], in_name: str, out_name: str) -> str:
    res = f"""
    for (int i = 0; i < {size}; i++)
    {{
        {out_name}[i] = {in_name}[i];
    }}\n
    """
    return res


def fill_1d_array_by_list_short(
    elem_type: str, size: Union[int, str], arr_name: str, inter_name: str, source: list
) -> str:
    res = f"""{elem_type} {inter_name}[{size}] = """
    temp = "{" + ", ".join(map(str, source)) + "};\n"
    res += temp
    res += f"""
    for (int i = 0; i < {size}; i++)
    {{
        {arr_name}[i] = {inter_name}[i];
    }}\n
    """

    return res


def fill_1d_array_by_list(arr_name: str, source: list) -> str:
    res = ""
    for i, elem in enumerate(source):
        res += f"{arr_name}[{i}] = {elem};\n"

    return res


def fill_2d_array_by_list(arr_name: str, source: list) -> str:
    res = ""
    for i, sub_arr in enumerate(source):
        for j, elem in enumerate(sub_arr):
            res += f"{arr_name}[{i}][{j}] = {elem};\n"

    return res


def feed_forward_step(
    left_name: str,
    left_size: int,
    right_name: str,
    right_size: int,
    weight_name: str,
    bias_name: str,
    activation_func: str,
) -> str:
    res = f"""
    for (int i = 0; i < {right_size}; i++)
    {{
        for (int j = 0; j < {left_size}; j++)
        {{
            {right_name}[i] += {weight_name}[j][i] * {left_name}[j];
        }}
        {right_name}[i] += {bias_name}[i];
        {activation_to_cpp_template(right_name + "[i]", activation_func)}
    }}
    """

    return res


def activation_to_cpp_template(name: str, activation_name: str) -> str:
    """
    Build code for activation function by function name and variable name

    Parameters
    ----------
    name: str
        Name of variable
    activation_name: str
        Name of activation func

    Returns
    -------
    c_activation: str
        Translated activation
    """
    d = {
        "linear": lambda x: f"{x} = {x};\n",
        "elu": lambda x: f"if ({x} >= 0) {x} = {x}; else {x} = 1.0 * (exp({x}) - 1);\n",
        "gelu": lambda x: f"{x} = 0.5 * {x} * (1 + tanh(sqrt(2 / 3.14159265) * ({x} + 0.044715 * {x} * {x} * {x})))",
        "relu": lambda x: f"{x} = std::max({x}, 0.0f);\n",
        "selu": lambda x: f"if ({x} >= 0) {x} = 1.05070098 * {x}; else {x} = 1.05070098 * 1.67326324 * (exp({x}) - 1);\n",
        "exponential": lambda x: f"{x} = exp({x});\n",
        "hard_sigmoid": lambda x: f"if ({x} < -2.5) {x} = 0; else if ({x} > 2.5) {x} = 1; else {x} = 0.2 * {x} + 0.5;\n",
        "sigmoid": lambda x: f"{x} = 1 / (1 + exp(-{x}));\n",
        "softplus": lambda x: f"{x} = log(exp({x}) + 1);\n",
        "softsign": lambda x: f"{x} = {x} / (abs({x}) + 1.0);\n",
        "swish": lambda x: f"{x} = {x} / (1 + exp(-{x}));\n",
        "tanh": lambda x: f"{x} = ((exp({x}) - exp(-{x}))/(exp({x}) + exp(-{x})));\n",
    }

    return d[activation_name](name)


def _fill_values(act_func: str, values: List[Tuple[str, float]]):
    for key, value in values:
        act_func = act_func.replace(key, str(value))
    return act_func
