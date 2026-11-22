"""Run each named .kg example in its own process so one missing primitive cannot hide another."""

import pytest

from .runtime import case_names, evaluate, load


def cases(program):
    return [pytest.param(program, name, id=name) for name in case_names(program)]


@pytest.mark.parametrize("program,name", cases("table_cases.kg") + cases("sql_cases.kg")
                         + cases("group_keys.kg") + cases("numpy_kernels.kg"))
def test_klong_case(host, program, name, tmp_path):
    assert evaluate(host, f"{load(program)};{name}()", tmp_path) == "1"
