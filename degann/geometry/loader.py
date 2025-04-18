def load_from_file(input_file: str) -> tuple[list[list[float | int]], list[list[float | int]], list[list[float | int]]]:
    """

    Parameters
    ----------
    input_file

    Returns
    -------

    """

    lines: list[str] = open(input_file + ".out", mode="r").readlines()

    row: int = 1
    points, row = load_matrix(lines, row, float)  # rectangle array
    row += 1
    polys, row = load_matrix(lines, row, float)  # rectangle array
    row += 1
    bound, row = load_matrix(lines, row, float)  # rectangle array

    for i in range(len(polys)):
        for j in range(len(polys[i])):
            polys[i][j] -= 1

    for i in range(2):
        for j in range(len(bound[i])):
            bound[i][j] -= 1

    return points, polys, bound


def load_matrix(lines: list[str], start_row: int, value_type: type = float) -> tuple[list[list[float]], int]:
    end_row = start_row
    m = []
    while end_row < len(lines):
        line = lines[end_row]
        if line.startswith("##") or len(line.strip()) == 0:
            break
        values = list(map(value_type, line.split()))
        m.append(values)
        end_row += 1
    return m, end_row
