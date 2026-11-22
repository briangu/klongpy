"""Independent storage readers, recovery after an interrupted writer, and a keyed replay model."""
from contextlib import closing
import gc
from importlib import import_module
from importlib.util import find_spec
import random
import weakref

import numpy as np
import pytest

from klongpy import KlongInterpreter
from klongpy.core import KLONG_UNDEFINED
from .peer import connect, server
from .runtime import evaluate, load, required


driver = import_module("duckdb") if find_spec("duckdb") else None


@pytest.fixture
def storage_driver():
    if driver is None:
        required("The storage interoperability tests require the DuckDB Python client")
    return driver


def test_read_file_from_independent_engine(host, storage_driver, tmp_path):
    with closing(storage_driver.connect(str(tmp_path / "prices.db"))) as db:
        db.execute("CREATE TABLE ticks(id BIGINT, ts TIMESTAMP_NS, px BIGINT)")
        db.execute("INSERT INTO ticks VALUES (1, make_timestamp_ns(946684800000000001), NULL),"
                   "(2, make_timestamp_ns(946684800000000007), 9007199254740993)")
    prefix = 'd::.db("prices.db");'
    assert evaluate(host, prefix + 'd("SELECT epoch_ns(ts) FROM ticks ORDER BY id")', tmp_path) == (
        "[946684800000000001 946684800000000007]")
    assert evaluate(host, prefix + 'd("SELECT px FROM ticks WHERE px IS NOT NULL")', tmp_path) == "9007199254740993"


def test_independent_engine_read_after_commit(host, storage_driver, tmp_path):
    expr = ('d::.db("prices.db");d("CREATE TABLE ledger(id BIGINT PRIMARY KEY, px BIGINT)");'
            'd("BEGIN");d("INSERT INTO ledger VALUES (1, NULL), (2, 9007199254740993)");d("COMMIT");1')
    assert evaluate(host, expr, tmp_path) == "1"
    with closing(storage_driver.connect(str(tmp_path / "prices.db"), read_only=True)) as db:
        assert db.execute("SELECT * FROM ledger ORDER BY id").fetchall() == [(1, None), (2, 2**53 + 1)]


@pytest.mark.parametrize("writer,reader", [("kg", "kgpy"), ("kgpy", "kg")])
def test_schema_migration_survives_host_handoff(writer, reader, request, storage_driver, tmp_path):
    fixture = load("schema_history.kg")
    path = tmp_path / "trade ledger.duckdb"
    migrate = f'{fixture};migrateLedger("{path.name}")'
    append = f'{fixture};appendLedger("{path.name}")'
    assert evaluate(request.getfixturevalue(writer), migrate, tmp_path) == "1"
    assert evaluate(request.getfixturevalue(reader), append, tmp_path) == "[0 3]"

    with closing(storage_driver.connect(str(path), read_only=True)) as db:
        assert db.execute("SHOW TABLES").fetchall() == [("ledger",)]
        result = db.execute("SELECT * FROM ledger ORDER BY id")
        assert [column[0] for column in result.description] == ["id", "price", "venue"]
        assert result.fetchall() == [(1, 2**53 + 1, "XNAS"), (2, None, "XNAS"), (3, -7, "XNAS")]


def test_last_hosted_handle_releases_database(kg, tmp_path):
    interpreter = KlongInterpreter(backend="numpy")
    interpreter["path"] = str(tmp_path / "lifetime.db")
    interpreter('db::.db(path);db("CREATE TABLE answer AS SELECT 42 AS value");read::{db("SELECT value FROM answer")}')
    reader = interpreter["read"]
    owner = weakref.ref(interpreter)
    del interpreter
    gc.collect()
    assert reader() == 42

    del reader
    gc.collect()
    assert evaluate(kg, 'db::.db("lifetime.db");db("SELECT value FROM answer")', tmp_path) == "42"
    assert owner() is None


@pytest.mark.parametrize("kind", ["kg", "kgpy"])
def test_process_death_preserves_only_committed_data(kind, request, tmp_path, storage_driver):
    command = request.getfixturevalue(kind)
    with server(command, kind, tmp_path) as (port, process), connect(port) as peer:
        peer.request('d::.db("journal.db");d("CREATE TABLE journal(i BIGINT)");d("INSERT INTO journal VALUES (3)");1')
        peer.request('d("BEGIN");d("INSERT INTO journal VALUES (99)");1')
        assert peer.request('d("SELECT sum(i) FROM journal")') == 102
        # Stop our test child while its transaction is live. Verify durability using a fresh, independent reader.
        # Popen.kill is portable and sends no application-level shutdown signal on POSIX.
        process.kill()
        process.wait(timeout=5)
    with closing(storage_driver.connect(str(tmp_path / "journal.db"), read_only=True)) as db:
        assert db.execute("SELECT * FROM journal").fetchall() == [(3,)]


@pytest.mark.parametrize("kind", ["kg", "kgpy"])
def test_keyed_replay_against_python_model(kind, request, tmp_path):
    rng = random.Random(93871)
    model = {(0, 0): 7}
    with server(request.getfixturevalue(kind), kind, tmp_path) as (port, _), connect(port) as peer:
        peer.request('t::.table(:{;"acct";,0;"lot";,0;"qty";,7});.index(t;["acct" "lot"]);'
                     'd::.db(:{;"ledger";t});ingest::{.insert(t;x);1};1')
        for step in range(36):
            batch = [[rng.randrange(4), rng.randrange(3), rng.randrange(-150, 151)]
                     for _ in range(1 + step % 7)]
            if step % 3 == 0:
                batch.append([batch[0][0], batch[0][1], step])
            assert peer.call("ingest", np.array(batch, dtype=np.int64)) == 1
            for acct, lot, qty in batch:
                model[acct, lot] = qty
            expected = [[acct, lot, model[acct, lot]] for acct, lot in sorted(model)]
            result = peer.request('d("SELECT acct, lot, qty FROM ledger ORDER BY acct, lot")')
            if len(expected) == 1:  # One row is squeezed by the database interface.
                result = [result]
            np.testing.assert_array_equal(result, expected, err_msg=f"first inconsistent batch {step}")


def test_live_text_columns_across_scan_chunks(host, tmp_path):
    # More than two DuckDB vectors, with a null, short UTF-8 or externally stored
    # string on either side of each boundary. Read again after an unkeyed append.
    values = ["x"] * 4101
    values[3] = values[2048] = values[4099] = None
    values[2047] = "é" * 6
    values[2049] = values[4096] = 'a long "quoted" value'
    literals = [":undefined" if s is None else '"' + s.replace('"', '""') + '"' for s in values]
    source = ('col::[;' + ';'.join(literals) + ']\n'
              't::.table(:{;"text";col})\nd::.db(:{;"words";t})\n'
              'counts::{d("SELECT count(text), sum(length(text)) FROM words")}\n')
    (tmp_path / "words.kg").write_text(source, encoding="utf-8")
    count = sum(s is not None for s in values)
    chars = sum(len(s) for s in values if s is not None)
    expr = ('.l("words.kg");before::counts();.insert(t;,"additional long text");'
            '[;before;counts()]')
    assert evaluate(host, expr, tmp_path) == f"[[{count} {chars}] [{count + 1} {chars + 20}]]"


@pytest.mark.parametrize("kind", ["kg", "kgpy"])
def test_sql_result_text_preserves_embedded_nul_across_chunks(kind, request, tmp_path):
    with server(request.getfixturevalue(kind), kind, tmp_path) as (port, _), connect(port) as peer:
        peer.request('d::.db(:{});1')
        result = peer.request(
            'd("SELECT CASE i WHEN 2047 THEN NULL WHEN 2048 THEN \'a\' || chr(0) || \'é\' '
            'WHEN 4096 THEN \'long multibyte é\' || chr(0) || \'tail\' ELSE \'\' END '
            'FROM range(4098) AS t(i) ORDER BY i")'
        )
        expected = [""] * 4098
        expected[2047] = KLONG_UNDEFINED
        expected[2048] = "a\0é"
        expected[4096] = "long multibyte é\0tail"
        assert result.tolist() == expected


@pytest.mark.parametrize("kind", ["kg", "kgpy"])
def test_sql_result_integers_and_scale_zero_decimals_are_exact(kind, request, tmp_path):
    with server(request.getfixturevalue(kind), kind, tmp_path) as (port, _), connect(port) as peer:
        peer.request('d::.db(:{});1')
        for type_name, number in (
            ("TINYINT", -128),
            ("SMALLINT", -32768),
            ("INTEGER", -(2**31)),
            ("BIGINT", -(2**63)),
            ("UTINYINT", 2**8 - 1),
            ("USMALLINT", 2**16 - 1),
            ("UINTEGER", 2**32 - 1),
            ("UBIGINT", 2**64 - 1),
            ("HUGEINT", -(2**127)),
            ("UHUGEINT", 2**128 - 1),
            ("DECIMAL(4,0)", -9999),
            ("DECIMAL(9,0)", 999999999),
            ("DECIMAL(18,0)", 9007199254740993),
            ("DECIMAL(38,0)", -(10**38 - 1)),
        ):
            assert peer.request(f'$d("SELECT CAST(\'{number}\' AS {type_name})")') == str(number)
