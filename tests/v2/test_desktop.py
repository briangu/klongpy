import subprocess

import numpy as np
import pytest

from klongpy import KlongInterpreter
from .runtime import PROGRAMS, environment, evaluate, load


@pytest.mark.parametrize("suite", ["language", "market", "table_values", "tables"])
def test_klong_program(host, tmp_path, suite):
    program = PROGRAMS / f"test_{suite}.kg"
    p = subprocess.run([*host, str(program)], cwd=tmp_path, env=environment(),
                       text=True, capture_output=True, timeout=15)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.strip().splitlines()[-1] == f"PASS {suite}"


@pytest.mark.parametrize("actual,expected", [("2", "3"), ("[1 2]", "[1 2 3]")])
def test_klong_expectation_fails_the_process(host, tmp_path, actual, expected):
    expr = f'{load("expect.kg")};expect("intentional mismatch";{actual};{expected})'
    p = subprocess.run([*host, "-e", expr], cwd=tmp_path, env=environment(),
                       text=True, capture_output=True, timeout=15)
    assert p.returncode != 0
    assert f"intentional mismatch expected {expected} got {actual}" in p.stderr


def test_native_database_runs_without_python(kg, tmp_path):
    # An attempted Python subprocess or normal embedded Python startup fails in this environment.
    empty = tmp_path / "empty"
    empty.mkdir()
    env = {"PATH": str(empty), "PYTHONHOME": str(empty), "PYTHONPATH": str(empty)}
    assert evaluate(kg, f'{load("market.kg")};lastpx()', tmp_path, extra_env=env) == "[103 201 300]"


def test_bad_column_shape_names_the_error(host, tmp_path):
    expr = '(.try({.table(:{;"a";[1 2];"b";[3]})};[])@1)@"kind"'
    assert evaluate(host, expr, tmp_path) == '"shape"'


def test_a_rejected_batch_leaves_table_intact(host, tmp_path):
    expr = (f'{load("tables.kg")};r::.try({{.insert(ledger;[[3 8 100] [4 9]])}};[]);'
            '[;r@0;rows()]')
    assert evaluate(host, expr, tmp_path) == "[0 [[1 7 10] [2 7 20]]]"


def test_empty_table_retains_column_names(host, tmp_path):
    expr = 't::.table(:{;"time";[];"px";[]});d::.db(:{;"ticks";t});[;.schema(t);d("SELECT count(*) FROM ticks")]'
    assert evaluate(host, expr, tmp_path) == '[["time" "px"] 0]'


@pytest.mark.parametrize("writer,reader", [("kg", "kgpy"), ("kgpy", "kg")])
def test_reopen_from_other_process(request, tmp_path, writer, reader):
    # .db(path) is the proposed file-backed form. Each invocation is a fresh process.
    create = 'd::.db("ticks.duckdb");d("CREATE TABLE ticks(seq BIGINT, px BIGINT)");d("INSERT INTO ticks VALUES (1, 9007199254740993), (2, NULL)");d("SELECT count(*) FROM ticks")'
    assert evaluate(request.getfixturevalue(writer), create, tmp_path) == "2"
    assert (tmp_path / "ticks.duckdb").is_file()
    read = 'd::.db("ticks.duckdb");d("SELECT px FROM ticks WHERE seq=1")'
    assert evaluate(request.getfixturevalue(reader), read, tmp_path) == "9007199254740993"
    nulls = 'd::.db("ticks.duckdb");d("SELECT count(*) FROM ticks WHERE px IS NULL")'
    assert evaluate(request.getfixturevalue(reader), nulls, tmp_path) == "1"


def test_same_market_answers_after_restart(host, tmp_path):
    save = (f'{load("market.kg")};market("ATTACH \'day.duckdb\' AS disk");'
            'market("CREATE TABLE disk.trades AS SELECT * FROM trades");'
            'market("CREATE TABLE disk.quotes AS SELECT * FROM quotes");market("DETACH disk");1')
    assert evaluate(host, save, tmp_path) == "1"
    assert (tmp_path / "day.duckdb").is_file()
    reopen = (f'market::.db("day.duckdb");{load("queries.kg")};'
              'market("BEGIN");market("DELETE FROM trades");market("ROLLBACK");'
              '[;lastpx();minutes();bids();unmatched()]')
    assert evaluate(host, reopen, tmp_path) == "[[103 201 300] [[0 7] [1 10]] [98 -1 101 -1 101 198] 2]"


def test_python_array_and_callback_share_a_klong_function():
    kg = KlongInterpreter(backend="numpy")
    kg(load("arrays.kg"))
    source = np.arange(12, dtype=np.int64)[::2]
    kg["samples"] = source
    kg["adjust"] = lambda value: value + 7
    assert kg("adjust(energy(samples))") == int(source @ source) + 7
    np.testing.assert_array_equal(source, [0, 2, 4, 6, 8, 10])
