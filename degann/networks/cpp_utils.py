import cpuinfo # needed to install to get info about available intrinsics information
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
    decorator_params: list = None,
    vectorized_level: str = "none",
) -> str:
    res = f"""
    {vectorized_level}_vectorized({right_name}, {left_name}, {bias_name}, {weight_name});
    """
    if vectorized_level == "none":
        res = f"""
            for (int i = 0; i < {right_size}; i++)
            {{s
                for (int j = 0; j < {left_size}; j++)
                {{
                    {right_name}[i] += {weight_name}[j][i] * {left_name}[j];
                }}
                {right_name}[i] += {bias_name}[i];
                {activation_to_cpp_template(right_name + "[i]", activation_func, vectorized_level, decorator_params)}
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
        "relu": lambda x: f"{x} = max({x}, 0.0f);\n",
        "selu": lambda x: f"if ({x} >= 0) {x} = 1.05070098 * {x}; else {x} = 1.05070098 * 1.67326324 * (exp({x}) - 1);\n",
        "exponential": lambda x: f"{x} = exp({x});\n",
        "hard_sigmoid": lambda x: f"if ({x} < -2.5) {x} = 0; else if ({x} > 2.5) {x} = 1; else {x} = 0.2 * {x} + 0.5;\n",
        "sigmoid": lambda x: f"{x} = 1 / (1 + exp(-{x}));\n",
        "softplus": lambda x: f"{x} = log(exp({x}) + 1);\n",
        "softsign": lambda x: f"{x} = {x} / (abs({x}) + 1.0);\n",
        "swish": lambda x: f"{x} = {x} / (1 + exp(-{x}));\n",
        "tanh": lambda x: f"{x} = ((exp({x}) - exp(-{x}))/(exp({x}) + exp(-{x})));\n",
    }

    if vectorized_level == "none":
            return d[activation_name](name)

    vectorized_d = {
        "linear": f"__m{typename} vans = vsum;\n",
        "elu": f"""__m{typename} vflag = _mm{funcname}_cmp_ps(vsum, _mm{funcname}_set1_ps(0.0f),  _CMP_NLT_US), vans = _mm{funcname}_setzero_ps();
vans = _mm{funcname}_add_ps(vans, _mm{funcname}_and_ps(vflag, vsum));
vans = _mm{funcname}_add_ps(vans, _mm{funcname}_andnot_ps(vflag, _mm{funcname}_mul_ps(_mm{funcname}_set1_ps(1.0f), _mm{funcname}_sub_ps(_mm{funcname}_exp_ps(vsum), _mm{funcname}_set1_ps(1.0f)))));\n""",
        "gelu": f"""__m{typename} part1 = _mm{funcname}_mul_ps(_mm{funcname}_set1_ps(0.5f), vsum); // 0.5 * x
__m{typename} part2 = _mm{funcname}_add_ps(vsum, _mm{funcname}_mul_ps(_mm{funcname}_set1_ps(0.044715f), _mm{funcname}_mul_ps(vsum, _mm{funcname}_mul_ps(vsum, vsum)))); // x + 0.044712 * x * x * x
__m{typename} part3 = _mm{funcname}_add_ps(_mm{funcname}_set1_ps(1.0f), _mm{funcname}_tanh_ps(_mm{funcname}_mul_ps(_mm{funcname}_sqrt_ps(_mm{funcname}_div_ps(_mm{funcname}_set1_ps(2.0f), _mm{funcname}_set1_ps(3.14159265f))), part2))); // 1 + tanh(sqrt * part2)
__m{typename} vans = _mm{funcname}_mul_ps(part1, part3);\n""",
        "relu": f"__m{typename} vans = _mm{funcname}_max_ps(vsum, _mm{funcname}_setzero_ps());\n",
        "selu": f"""__m{typename} vflag = _mm{funcname}_cmp_ps(vsum, _mm{funcname}_set1_ps(0.0f), _CMP_NLT_US), vans = _mm{funcname}_setzero_ps();
        vans = _mm{funcname}_add_ps(vans, _mm{funcname}_and_ps(vflag, _mm{funcname}_mul_ps(_mm{funcname}_set1_ps(1.05070098f), vsum)));
        vans = _mm{funcname}_add_ps(vans, _mm{funcname}_andnot_ps(vflag, _mm{funcname}_mul_ps(_mm{funcname}_set1_ps(1.05070098f), _mm{funcname}_mul_ps(_mm{funcname}_set1_ps(1.67326324f), _mm{funcname}_sub_ps(_mm{funcname}_exp_ps(vsum), _mm{funcname}_set1_ps(1.0f))))));\n""",
        "exponential": f"__m{typename} vans = _mm{funcname}_exp_ps(vsum);\n",
        "hard_sigmoid": f"""__m{typename} vflag_less = _mm{funcname}_cmp_ps(vsum, _mm{funcname}_set1_ps(-2.5f), _CMP_LT_OS), vflag_greater = _mm{funcname}_cmp_ps(vsum, _mm{funcname}_set1_ps(2.5f), _CMP_GT_OS), vans = _mm{funcname}_setzero_ps();
        vans = _mm{funcname}_add_ps(vans, _mm{funcname}_and_ps(vflag_less, _mm{funcname}_set1_ps(0.0f)));
        vans = _mm{funcname}_add_ps(vans, _mm{funcname}_and_ps(vflag_greater, _mm{funcname}_set1_ps(1.0f)));
        vans = _mm{funcname}_add_ps(vans, _mm{funcname}_andnot_ps(_mm{funcname}_or_ps(vflag_greater, vflag_less), _mm{funcname}_add_ps(_mm{funcname}_mul_ps(vsum, _mm{funcname}_set1_ps(0.2f)), _mm{funcname}_set1_ps(0.5f))));\n""",
        "sigmoid": f"""__m{typename} vans = _mm{funcname}_div_ps(_mm{funcname}_set1_ps(1.0f), 
(_mm{funcname}_add_ps(_mm{funcname}_set1_ps(1.0f), _mm{funcname}_exp_ps(_mm{funcname}_mul_ps(_mm{funcname}_set1_ps(-1.0f), vsum)))));\n""",
        "softplus": f"__m{typename} vans = _mm{funcname}_log_ps(_mm{funcname}_add_ps(_mm{funcname}_exp_ps(vsum), _mm{funcname}_set1_ps(1.0f)));\n",
        "softsign": f"""__m{typename} vflag = _mm{funcname}_cmp_ps(vsum, _mm{funcname}_set1_ps(0.0f), _CMP_LT_OS), abs_vsum = _mm{funcname}_setzero_ps();
        abs_vsum = _mm{funcname}_add_ps(abs_vsum, _mm{funcname}_and_ps(vflag, _mm{funcname}_mul_ps(_mm{funcname}_set1_ps(-1.0f), vsum)));
        abs_vsum = _mm{funcname}_add_ps(abs_vsum, _mm{funcname}_andnot_ps(vflag, vsum));
        __m{typename} vans = _mm{funcname}_div_ps(vsum, _mm{funcname}_add_ps(abs_vsum, _mm{funcname}_set1_ps(1.0f)));\n""",
        # "softmax": lambda x: ,
        "swish": f"""__m{typename} vans = _mm{funcname}_div_ps(vsum, 
(_mm{funcname}_add_ps(_mm{funcname}_set1_ps(1.0f), _mm{funcname}_exp_ps(_mm{funcname}_mul_ps(_mm{funcname}_set1_ps(-1.0f), vsum)))));\n""",
        "tanh": f"""__m{typename} vans = _mm{funcname}_div_ps(_mm{funcname}_sub_ps(_mm{funcname}_exp_ps(vsum), _mm{funcname}_exp_ps(_mm{funcname}_mul_ps(_mm{funcname}_set1_ps(-1.0f), vsum))), 
_mm{funcname}_add_ps(_mm{funcname}_exp_ps(vsum), _mm{funcname}_exp_ps(_mm{funcname}_mul_ps(_mm{funcname}_set1_ps(-1.0f), vsum))));\n""",
    } if vectorized_level != "avx512" \
        else {
        "linear": f"__m{typename} vans = vsum;\n",
        "elu": f"""__mmask{int(typename) // 32} vflag = _mm{funcname}_cmp_ps_mask(vsum, _mm{funcname}_set1_ps(0.0f), _CMP_LT_OS);
__m{typename} vans = _mm{funcname}_mask_mul_ps(vsum, vflag, _mm{funcname}_sub_ps(_mm{funcname}_exp_ps(vsum), _mm{funcname}_set1_ps(1.0f)), _mm{funcname}_set1_ps(1.0f));""",
        "gelu": f"""__m{typename} part1 = _mm{funcname}_mul_ps(_mm{funcname}_set1_ps(0.5f), vsum); // 0.5 * x
__m{typename} part2 = _mm{funcname}_add_ps(vsum, _mm{funcname}_mul_ps(_mm{funcname}_set1_ps(0.044715f), _mm{funcname}_mul_ps(vsum, _mm{funcname}_mul_ps(vsum, vsum)))); // x + 0.044712 * x * x * x
__m{typename} part3 = _mm{funcname}_add_ps(_mm{funcname}_set1_ps(1.0f), _mm{funcname}_tanh_ps(_mm{funcname}_mul_ps(_mm{funcname}_sqrt_ps(_mm{funcname}_div_ps(_mm{funcname}_set1_ps(2.0f), _mm{funcname}_set1_ps(3.14159265f))), part2))); // 1 + tanh(sqrt * part2)
__m{typename} vans = _mm{funcname}_mul_ps(part1, part3);\n""",
        "relu": f"vans = _mm{funcname}(vsum, _mm{funcname}_setzero_ps());\n",
        "selu": f"""__mmask{int(typename) // 32} vflag = _mm{funcname}_cmp_ps_mask(vsum, _mm{funcname}_set1_ps(0.0f), _CMP_NLT_US);
__m{typename} vans = _mm{funcname}_mask_mul_ps(_mm{funcname}_mul_ps(_mm{funcname}_mul_ps(_mm{funcname}_set1_ps(1.05070098f), _mm{funcname}_set1_ps(1.67326324f)), _mm{funcname}_sub_ps(_mm{funcname}_exp_ps(vsum), _mm{funcname}_set1_ps(1.0f))), vflag, vsum, _mm{funcname}_set1_ps(1.05070098f));
\n""",
        "exponential": f"__m{typename} vans = _mm{funcname}_exp_ps(vsum);\n",
        "hard_sigmoid": f"""__mmask{int(typename) // 32} vflag_less = _mm{funcname}_cmp_ps_mask(vsum, _mm{funcname}_set1_ps(-2.5f), _CMP_LT_OS);
__mmask{int(typename) // 32} vflag_more = _mm{funcname}_cmp_ps_mask(vsum, _mm{funcname}_set1_ps(2.5f), _CMP_GT_OS);
__m{typename} vans = _mm{funcname}_mask_expand_ps(_mm{funcname}_add_ps(_mm{funcname}_mul_ps(vsum, _mm{funcname}_set1_ps(0.2f)), _mm{funcname}_set1_ps(0.5f)), vflag_less, _mm{funcname}_set1_ps(0.0f));
__m{typename} vans = _mm{funcname}_mask_expand_ps(vans, vflag_more, _mm{funcname}_set1_ps(1.0f));\n""",
        "sigmoid": f"""__m{typename} vans = _mm{funcname}_div_ps(_mm{funcname}_set1_ps(1.0f), 
(_mm{funcname}_add_ps(_mm{funcname}_set1_ps(1.0f), _mm{funcname}_exp_ps(_mm{funcname}_mul_ps(_mm{funcname}_set1_ps(-1.0f), vsum)))));""",
        "softplus": f"__m{typename} vans = _mm{funcname}_log_ps(_mm{funcname}_add_ps(_mm{funcname}_exp_ps(vsum), _mm{funcname}_set1_ps(1.0f)));",
        "softsign": f"""__m{typename} flag = _mm{funcname}_cmplt_ps(vsum, _mm{funcname}_set1_ps(0.0f)), abs_vsum = _mm{funcname}_setzero_ps();
abs_vsum = _mm{funcname}_add_ps(abs_vsum, _mm{funcname}_and_ps(flag, _mm{funcname}_mul_ps(_mm{funcname}_set1_ps(-1.0f), vsum)));
abs_vsum = _mm{funcname}_add_ps(abs_vsum, _mm{funcname}_andnot_ps(flag, vsum));
__m{typename} vans = _mm{funcname}_div_ps(vsum, _mm{funcname}_add_ps(abs_vsum, _mm{funcname}_set1_ps(1.0f)));""",
        # "softmax": lambda x: ,
        "swish": f"""__m{typename} vans = _mm{funcname}_div_ps(vsum, 
(_mm{funcname}_add_ps(_mm{funcname}_set1_ps(1.0f), _mm{funcname}_exp_ps(_mm{funcname}_mul_ps(_mm{funcname}_set1_ps(-1.0f), vsum)))));""",
        "tanh": f"""__m{typename} vans = _mm{funcname}_div_ps(_mm{funcname}_sub_ps(_mm{funcname}_exp_ps(vsum), _mm{funcname}_exp_ps(_mm{funcname}_mul_ps(_mm{funcname}_set1_ps(-1.0f), vsum))), 
_mm{funcname}_add_ps(_mm{funcname}_exp_ps(vsum), _mm{funcname}_exp_ps(_mm{funcname}_mul_ps(_mm{funcname}_set1_ps(-1.0f), vsum))));""",
    }
    # if activation_name == "perceptron_threshold":
    #     d.update(
    #         {
    #             "perceptron_threshold": lambda x: _fill_values(
    #                 f"if ({x} >= threshold) {x} = 1; else {x} = 0;\n", decorator_params
    #             )
    #         }
    #     )
    return vectorized_d[activation_name]

def generate_vectorized_function(
        vectorized_level: str, activation_func: str
) -> str:
    if vectorized_level == "none":
        return ""
    typename, funcname = get_vectorized_names(vectorized_level)
    weights_for_st_setr = ""
    weights_for_nd_setr = ""
    step = f"{int(typename) // 32}"
    for q in range(int(step)):
        weights_for_st_setr += f"weight[j + {q}][i + q], "
        weights_for_nd_setr += f"weight[j + {q}][i], "
    weights_for_st_setr = weights_for_st_setr[:-2:]
    weights_for_nd_setr = weights_for_nd_setr[:-2:]
    res = (f"""\ntemplate <size_t a, size_t b>
void {vectorized_level}_vectorized(float* cur_layer, float* pre_layer, float* bias, float(&weight)[a][b])""" + "{" +
           f"""
                    int ni = b, nj = a;
                    int ni{step} = ni - (ni % {step}), nj{step} = nj - (nj % {step}); // ni{step} and nj{step} was created for vectorized cycles
                    // below executed vectorized i and j cycles
                    for (int i = 0; i < ni{step}; i += {step}) 
               """ + "{" +
           f"""
                        // creating variables containing {step} values 
                        __m{typename} m{typename}_cur_layers[{step}], m{typename}_weight, m{typename}_pre_layer;
                        for (int q = 0; q < {step}; ++q) m{typename}_cur_layers[q] = _mm{funcname}_setzero_ps();
                        // vectorized j-cycle
                        for (int j = 0; j < nj{step}; j += {step}) 
               """ + "{" +
           f"""
                    for (int q = 0; q < {step}; ++q) """ + "{" +
           f"""
                    m{typename}_weight = _mm{funcname}_setr_ps({weights_for_st_setr});
                    m{typename}_pre_layer = _mm{funcname}_load_ps(&pre_layer[j]);
                    m{typename}_cur_layers[q] = _mm{funcname}_add_ps(m{typename}_cur_layers[q], _mm{funcname}_mul_ps(m{typename}_weight, m{typename}_pre_layer));
                        """ + """}
                             }""" +
           f"""
                        // now store results of the vectorized j-cycle
                        float tmp[{step}], sum[{step}];
                        for (int q = 0; q < {step}; ++q)
               """ + "{" +
           f"""
                            _mm{funcname}_store_ps(tmp, m{typename}_cur_layers[q]); sum[q] = 0;
                            for (int j = 0; j < {step}; ++j) sum[q] += tmp[j];\n
                """ + "}" +
           f"""
                        // non-vectorized j-cycle (for less than {step} steps)
                        for (int j = nj{step}; j < nj; ++j)""" + "{" +
           f"""
                            for (int q = 0; q < {step}; ++q) """ + "{" +
           f"""
                                sum[q] += weight[j][i + q] * pre_layer[j];
                                """ + """}
                                }""" +
           f"""
                        // now store results of the vectorized i-cycle
                        __m{typename} vsum = _mm{funcname}_load_ps(&sum[0]);
                        vsum = _mm{funcname}_add_ps(vsum, _mm{funcname}_load_ps(&bias[i]));""" +
           f"""
                        {activation_to_cpp_template("cur_layer[i]", activation_func, vectorized_level=vectorized_level, typename=typename, funcname=funcname)}
                        _mm{funcname}_store_ps(&cur_layer[i], vans);
                """ +
           "}" + f"""

                    // non-vectorized i-cycle (for less than {step} steps)
                    for (int i = ni{step}; i < ni; ++i)
                """ + "{" +
           f"""
                        // vectorized j-cycle again
                        __m{typename} m{typename}_cur_layer = _mm{funcname}_setzero_ps(), m{typename}_weight, m{typename}_pre_layer;
                        for (int j = 0; j < nj{step}; j += {step}) 
                """ + "{" +
           f"""
                            m{typename}_weight = _mm{funcname}_setr_ps({weights_for_nd_setr});
                            m{typename}_pre_layer = _mm{funcname}_load_ps(&pre_layer[j]);
                            m{typename}_cur_layer = _mm{funcname}_add_ps(m{typename}_cur_layer, _mm{funcname}_mul_ps(m{typename}_weight, m{typename}_pre_layer));
                """ + "}" +
           f"""
                        // store results of the vectorized j-cycle again
                        float res_of_cur_layers[{step}];
                        _mm{funcname}_store_ps(&res_of_cur_layers[0], m{typename}_cur_layer);
                        for (int q = 0; q < {step}; ++q) cur_layer[i] += res_of_cur_layers[q];

                        // and non-vectorized j-cycle again (for less than {step} steps)
                        for (int j = nj{step}; j < nj; ++j) """ + "{\n" + f"""
                            cur_layer[i] += weight[j][i] * pre_layer[j];
                        """ +
           "}" + f"""
                        cur_layer[i] += bias[i];
                        {activation_to_cpp_template("cur_layer[i]", activation_func)}
                    """ + """}
                    }\n\n""")
    return res if vectorized_level != "none" else ""


def get_vectorized_names(vectorized_level: str) -> tuple:
    intrisincs = ["sse", "avx", "avx512"]
    vectorized_typename = {"none": "", "sse": "128", "avx": "256", "avx512": "512"}
    vectorized_funcname = {"none": "", "sse": "", "avx": "256", "avx512": "512"}

    return (vectorized_typename[vectorized_level], vectorized_funcname[vectorized_level])


def get_vectorized_level() -> str:
    intrisincs = ["sse", "avx", "avx512"]
    intrisincs_info = cpuinfo.get_cpu_info()
    vectorized_level = ""
    for x in intrisincs:
        if x in intrisincs_info["flags"]:
            vectorized_level = x
    return vectorized_level



def _fill_values(act_func: str, values: List[Tuple[str, float]]):
    for key, value in values:
        act_func = act_func.replace(key, str(value))
    return act_func
