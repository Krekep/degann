import pytest
import numpy as np
from tests.utils import array_compare
from degann.equations import equation_utils
from degann.equations.system_ode import SystemODE


@pytest.mark.parametrize(
    "eq, interval, points, expected",
    [
        ("y0*2 y0(0)=1", (0, 1), 3, np.array([[0, 1], [0.5, np.e], [1, np.e**2]])),
        (
            "y1*y2 y0(0)=0\n" "-y0*y2 y1(0)=1\n" "-0.5*y0*y1 y2(0)=1",
            (0, 3),
            4,
            np.array([[0, 0], [1, 0.8], [2, 1], [3, 0.7]]),
        ),
    ],
)
def test_solve_system_ode(eq, interval, points, expected):
    parsed = equation_utils.system_ode_from_string(eq)

    s = SystemODE()
    s.prepare_equations(len(parsed), parsed)
    s.solve(interval, points)
    actual = s.build_table([0])

    assert array_compare(actual, expected, eps=-1)
