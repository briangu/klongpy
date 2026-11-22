"""Interoperability of KlongPy's kdb+ IPC (``.cli .clid .srv .async``) with PeachQ, an open-source q.

PeachQ (https://github.com/peachq-org/peachq, MIT licence) is an independent implementation of the q
language that speaks the kdb+ wire protocol.  These tests are opt-in: they are skipped unless the
environment variable ``KLONG_TEST_PEACHQ`` names the PeachQ ``q`` executable.  Run them with::

    KLONG_TEST_PEACHQ=/tmp/peachq/v0.92/q python -m unittest tests/test_ipc_peachq.py

Everything PeachQ does here is PeachQ's behaviour, not proof of what KX's kdb+ does.  PeachQ is young: where
the two disagree it can be a bug on either side, and each such case is marked in the test that shows it
(``expectedFailure`` for a KlongPy bug, a skip naming the PeachQ version for a PeachQ one).  The answers to
the VERIFY items of ``docs/ipc_capabilities.md`` are in ``TestVerifyItems``.

Both directions are covered: KlongPy ``.cli`` against ``q -p PORT`` (``TestKlongClient...``) and a q script that
``hopen``s a KlongPy ``.srv`` (``TestPeachQClient...``).  Every step is time-bounded.  Tables from
``klongpy.db`` need pandas and duckdb and are skipped without them.
"""
import json
import math
import os
import socket
import struct
import sys
import time
import unittest
from functools import partial
from unittest.mock import patch

import numpy as np

from ipc_helpers import FakeKdbServer, bounded, free_port, wait_until
from peachq_helpers import (ENV, Q_PRELUDE, TESTED_VERSION, PeachQServer, lan_address, parse_results, peachq_exe,
                            peachq_version, run_q)

from klongpy.core import KGChar, KGSym, KLONG_UNDEFINED
from klongpy.errors import KlongRemoteError, KlongSystemError
from klongpy.ipc import KdbLambda, KdbOpaque, encode, encode_call, decode
from klongpy.ipc.codec import KdbTable
from klongpy.repl import cleanup_repl, create_repl
from klongpy.sys_fn_ipc import HostPortConnectionProvider, KGRemoteFnProxy, KlongIPCTimeoutException

try:
    import pandas  # noqa: F401
    import duckdb  # noqa: F401
    HAVE_DB = True
except ImportError:
    HAVE_DB = False

EXE = peachq_exe()
VERSION = peachq_version(EXE) if EXE else "unknown"
LAN = lan_address() if EXE else None
NEED_PEACHQ = unittest.skipUnless(EXE, f"PeachQ interop is opt-in: set {ENV} to a PeachQ q executable "
                                       f"(tested with {TESTED_VERSION}; see docs/ipc_capabilities.md)")
NEED_LAN = unittest.skipUnless(LAN, "no non-loopback IPv4 address: PeachQ compresses only for non-local peers")
NEED_DB = unittest.skipUnless(HAVE_DB, "klongpy.db needs pandas and duckdb (run with a Python that has both)")

S = KGSym


def obj_array(*items):
    """A 1-D object array holding exactly ``items`` (``np.array`` would try to stack equal-length arrays)."""
    a = np.empty(len(items), dtype=object)
    for i, x in enumerate(items):
        a[i] = x
    return a


NAN = float("nan")
TIMEOUT = 20


def peachq_bug(text):
    """Skip a test for a PeachQ bug or missing feature, naming the version it was seen on."""
    return unittest.skip(f"PeachQ {TESTED_VERSION}: {text}")


class Case(unittest.TestCase):
    """Interpreters that are shut down cleanly, bounded evaluation, a deep comparison."""

    def repl(self):
        klong, loops = create_repl()
        self.backend = klong.backend
        self.public_float = klong._float or float

        def shutdown():
            bounded(klong['.system']['closeEvent'].trigger, 20)
            bounded(lambda: cleanup_repl(loops), 20)
        self.addCleanup(shutdown)
        return klong

    def kg(self, klong, text, timeout=TIMEOUT):
        return bounded(lambda: klong(text), timeout)

    def assertSame(self, got, want, msg=None, path="value", *, host_array=False):
        """Compare values and their public backend representation, including dtype and device."""
        where = f"{msg or ''} {path}"
        if isinstance(want, np.ndarray):
            backend = getattr(self, "backend", None)
            if (want.dtype.kind in "if" and not host_array and backend is not None and backend.name != "numpy"
                    and (want.dtype != np.float64 or backend.supports_float64())):
                self.assertTrue(backend.is_backend_array(got), where)
                device = backend.device
                if device is not None:
                    family, separator, index = str(device).partition(":")
                    self.assertEqual(str(got.device).split(":", 1)[0], family, where)
                    if separator:
                        self.assertEqual(str(got.device), f"{family}:{index}", where)
                got = backend.to_numpy(got)
                self.assertEqual(got.dtype, want.dtype, where)
            self.assertIsInstance(got, np.ndarray, where)
            self.assertEqual(got.shape, want.shape, where)
            if want.dtype == object:
                self.assertEqual(got.dtype, object, where)
                for i, (g, w) in enumerate(zip(got.tolist() if got.ndim == 1 else got, want)):
                    self.assertSame(g, w, msg, f"{path}[{i}]")
            else:
                self.assertEqual(got.dtype.kind, want.dtype.kind, where)
                np.testing.assert_array_equal(got, want, err_msg=where)
        elif isinstance(want, dict):
            self.assertIsInstance(got, dict, where)
            self.assertEqual(list(got.keys()), list(want.keys()), where)
            for k in want:
                self.assertSame(got[k], want[k], msg, f"{path}[{k}]")
        elif isinstance(want, list):
            self.assertEqual(len(got), len(want), where)
            for i, (g, w) in enumerate(zip(got, want)):
                self.assertSame(g, w, msg, f"{path}[{i}]")
        elif isinstance(want, float):
            expected_type = getattr(self, "public_float", float) if path == "value" else float
            self.assertIsInstance(got, expected_type, where)
            with np.errstate(over="ignore"):
                want = expected_type(want)
            if math.isnan(want):
                self.assertTrue(math.isnan(got), where)
            else:
                self.assertEqual(got, want, where)
        elif isinstance(want, KGSym):
            self.assertIsInstance(got, KGSym, where)
            self.assertEqual(str(got), str(want), where)
        elif isinstance(want, KGChar):
            self.assertIsInstance(got, KGChar, where)
            self.assertEqual(str(got), str(want), where)
        elif isinstance(want, bool):
            self.assertEqual(got, want, where)
        elif isinstance(want, int):
            self.assertIs(type(got), int, f"{where}: {got!r}")
            self.assertEqual(got, want, where)
        else:
            self.assertEqual(got, want, where)


# ----------------------------------------------------------------------------------------------------------
# KlongPy client -> PeachQ server
# ----------------------------------------------------------------------------------------------------------

class KlongClientCase(Case):
    """A PeachQ server shared by the tests of a class and a fresh KlongPy client interpreter per test."""

    @classmethod
    def setUpClass(cls):
        cls.peachq = PeachQServer(EXE)
        cls.addClassCleanup(cls.peachq.stop)

    def setUp(self):
        self.assertTrue(self.peachq.alive(), "PeachQ server is not running: " + self.peachq.stdout())
        self.klong = self.repl()
        self.connect()

    def connect(self, host=None, name="c"):
        target = self.peachq.port if host is None else f'"{host}:{self.peachq.port}"'
        self.kg(self.klong, f"{name}::.cli({target})")

    def q(self, *items, name="c"):
        """One sync request: a string is source text, several items are a call (callee first)."""
        self.klong["req"] = items[0] if len(items) == 1 else list(items)
        return self.kg(self.klong, f"{name}(req)")

    def q_error(self, *items):
        with self.assertRaises(KlongRemoteError) as cm:
            self.q(*items)
        return cm.exception


@NEED_PEACHQ
class TestKlongClientReceives(KlongClientCase):
    """What arrives from q for each type of the mapping table."""

    def check(self, table):
        for expr, want in table:
            with self.subTest(expr):
                self.assertSame(self.q(expr), want, expr)

    def test_basic_eval(self):
        self.assertEqual(self.q("1+1"), 2)
        self.assertEqual(self.q("sum 1 2 3"), 6)

    def test_integer_atoms(self):
        self.check([("42", 42), ("42i", 42), ("42h", 42), ("1b", 1), ("0b", 0), ("0x2a", 42), ("-7", -7)])

    def test_float_atoms(self):
        self.check([("3.14", 3.14), ("-2.5", -2.5), ("1e100", 1e100), ("1.5e", 1.5), ("1f", 1.0)])

    def test_real_atom_widens_to_float(self):
        self.assertAlmostEqual(self.q("3.14e"), 3.140000104904175, places=12)    # a real, widened to a double

    def test_integer_vectors(self):
        self.check([("1 2 3", np.array([1, 2, 3])), ("1 2 3i", np.array([1, 2, 3])), ("1 2 3h", np.array([1, 2, 3])),
                    ("101b", np.array([1, 0, 1])), ("0x01020a", np.array([1, 2, 10])), ("til 5", np.arange(5)),
                    ("enlist 7", np.array([7]))])

    def test_float_vectors(self):
        real_dtype = np.float64 if self.backend.supports_float64() else np.float32
        self.check([("1.5 2.5", np.array([1.5, 2.5])), ("1.5 2.5e", np.array([1.5, 2.5], dtype=real_dtype)),
                    ("enlist 1.5", np.array([1.5]))])

    def test_echo_real_and_double_preserve_width_and_device(self):
        narrow = np.array([1.5, -0.0], dtype=np.float32)
        wide = np.array([1 + 2 ** -52, 1e100], dtype=np.float64)
        real_dtype = np.float64 if self.backend.supports_float64() else np.float32
        self.assertEqual(self.q("{type x}", narrow), 8)
        self.assertEqual(self.q("{type x}", wide), 9)
        self.assertSame(self.q("{x}", narrow), narrow.astype(real_dtype))
        self.assertSame(self.q("{x}", wide), wide)
        self.assertSame(self.q("(1.5 2.5e;3.5 4.5e)"), np.array([[1.5, 2.5], [3.5, 4.5]], dtype=real_dtype))
        self.assertSame(self.q("(1.5 2.5e;1.0000000000000002 1e100)"), np.array([[1.5, 2.5], wide]))

    def test_chars_strings_symbols(self):
        self.check([('"a"', KGChar("a")), ('"hello"', "hello"), ('""', ""), ('"h\\303\\251llo"', "héllo"),
                    ('enlist "a"', "a"), ("`abc", S("abc")), ("`", S("")), ('`$"a b"', S("a b")),
                    ("`a`b`c", np.array([S("a"), S("b"), S("c")], dtype=object)),
                    ("enlist `a", np.array([S("a")], dtype=object)),
                    ("`symbol$()", np.array([], dtype=object))])

    def test_general_lists(self):
        self.check([("(1;`a;\"xy\")", np.array([1, S("a"), "xy"], dtype=object)),
                    ("(\"ab\";\"cd\")", np.array(["ab", "cd"], dtype=object)),
                    ("()", np.array([], dtype=np.float64)),
                    ("(1;2.5)", np.array([1, 2.5], dtype=object))])

    def test_nested_lists(self):
        self.check([("(1 2;3 4)", np.array([[1, 2], [3, 4]])),
                    ("((1 2;3 4);(5 6;7 8))", np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])),
                    ("(1 2;(3;4 5))", obj_array(np.array([1, 2]), obj_array(3, np.array([4, 5]))))])

    def test_dictionaries(self):
        self.check([("`a`b!1 2", {S("a"): 1, S("b"): 2}),
                    ("`a`b!(1 2;\"x\")", {S("a"): np.array([1, 2]), S("b"): "x"}),
                    ("`a`b!`c`d", {S("a"): S("c"), S("b"): S("d")}),
                    ("1 2!3 4", {1: 3, 2: 4}),
                    ("`a`b!(`c`d!1 2;3)", {S("a"): {S("c"): 1, S("d"): 2}, S("b"): 3})])

    def test_sorted_dictionary_and_attributes(self):
        self.check([("`s#`a`b!1 2", {S("a"): 1, S("b"): 2}), ("`s#1 2 3", np.array([1, 2, 3])),
                    ("`u#1 2 3", np.array([1, 2, 3]))])

    def test_table_arrives_as_a_table(self):
        t = self.q("([]a:1 2 3;b:1.5 2.5 3.5;s:`x`y`z)")
        self.assertIsInstance(t, dict)                     # KdbTable, or klongpy.db.Table when that is loaded
        self.assertEqual([str(k) for k in t.keys()] if isinstance(t, KdbTable) else list(t.columns), ["a", "b", "s"])
        get = (lambda k: t[S(k)]) if isinstance(t, KdbTable) else t.get
        self.assertSame(np.asarray(get("a")), np.array([1, 2, 3]), host_array=True)
        self.assertSame(np.asarray(get("b")), np.array([1.5, 2.5, 3.5]), host_array=True)
        self.assertEqual([str(x) for x in get("s")], ["x", "y", "z"])

    def test_empty_table_and_table_with_string_column(self):
        empty = self.q("([]a:`long$();b:`symbol$())")
        names, columns, _, _ = empty.__klong_ipc_columns__()
        self.assertEqual([str(name) for name in names], ["a", "b"])
        self.assertEqual([len(column) for column in columns], [0, 0])
        self.assertEqual(self.q("{(type x[`a]),type x[`b]}", empty).tolist(), [7, 11])

        t = self.q('([]a:1 2;c:("ab";"cd"))')
        names, columns, _, _ = t.__klong_ipc_columns__()
        self.assertEqual([str(name) for name in names], ["a", "c"])
        np.testing.assert_array_equal(columns[0], [1, 2])
        self.assertEqual(list(columns[1]), ["ab", "cd"])

    def test_keyed_table_retains_keys_and_values(self):
        r = self.q("([k:1 2]v:3 4)")
        self.assertEqual(self.q("{x~([k:1 2]v:3 4)}", r), 1)

    def test_nulls_and_infinities_are_not_translated(self):
        self.check([("0N", -2 ** 63), ("0Nj", -2 ** 63), ("0Nj", -2 ** 63), ("0Wj", 2 ** 63 - 1), ("-0Wj", -(2 ** 63 - 1)),
                    ("0Ni", -2 ** 31), ("0Wi", 2 ** 31 - 1), ("0Nh", -2 ** 15), ("0n", NAN), ("0w", math.inf),
                    ("-0w", -math.inf), ("0Ne", NAN), ("1 0N 3", np.array([1, -2 ** 63, 3])),
                    ("1.5 0n 2.5", np.array([1.5, NAN, 2.5])), ("0w -0w", np.array([math.inf, -math.inf]))])

    def test_temporals_arrive_as_the_underlying_number(self):
        self.check([("2024.01.15D10:30:00.000000000", 758629800000000000),     # timestamp: ns since 2000
                    ("2024.01.15", 8780),                                      # date: days since 2000
                    ("12:30:00.000", 45000000), ("12:30", 750), ("12:30:15", 45015),
                    ("2024.01m", 288), ("0D01:15:00", 4500000000000),
                    ("2024.01.15T10:30:00.000", 8780.4375),                    # datetime: float days
                    ("0Np", -2 ** 63), ("0Wp", 2 ** 63 - 1), ("0Nd", -2 ** 31), ("0Nz", NAN),
                    ("2024.01.15 2024.01.16", np.array([8780, 8781])),
                    ("12:30:00.000 13:00:00.000", np.array([45000000, 46800000]))])

    def test_guid_arrives_as_sixteen_integers(self):
        self.assertSame(self.q('"G"$"8c6b8b64-6815-6084-0a3e-178401251b68"'),
                        np.array([140, 107, 139, 100, 104, 21, 96, 132, 10, 62, 23, 132, 1, 37, 27, 104]))

    def test_unit_value(self):
        self.assertIs(self.q("(::)"), KLONG_UNDEFINED)
        self.assertIs(self.q("a:1"), KLONG_UNDEFINED)

    def test_q_function_is_opaque_and_can_be_sent_back(self):
        f = self.q("{x*y}")
        self.assertIsInstance(f, KdbLambda)                  # a lambda whose context is not "kg"
        self.assertEqual((f.context, f.text, f.is_klong), ("", "{x*y}", False))
        self.assertEqual(self.q(f, 6, 7), 42)                # sent back as the first item of a call
        for expr in ("+", "neg", "(+/)"):
            with self.subTest(expr):
                self.assertIsInstance(self.q(expr), KdbOpaque)
        self.assertEqual(self.q("+[1]").type, 104)


@NEED_PEACHQ
class TestKlongClientSends(KlongClientCase):
    """What q sees for each Klong value: its type number and the value echoed back."""

    def setUp(self):
        super().setUp()
        self.q("t:{type x}")
        self.q("i:{x}")

    def check(self, table):
        for label, value, qtype, back in table:
            with self.subTest(label):
                self.assertEqual(self.q("t", value), qtype, f"q type of {label}")
                self.assertSame(self.q("i", value), value if back is None else back, label)

    def test_atoms(self):
        self.check([("int", 5, -7, None), ("int64 max", 2 ** 63 - 1, -7, None), ("int64 min", -2 ** 63, -7, None),
                    ("float", 1.5, -9, None), ("nan", NAN, -9, None), ("inf", math.inf, -9, None),
                    ("-inf", -math.inf, -9, None), ("char", KGChar("a"), -10, None), ("symbol", S("abc"), -11, None),
                    ("empty symbol", S(""), -11, None), ("symbol with space", S("a b"), -11, None)])

    def test_strings(self):
        self.check([("string", "abc", 10, None), ("empty string", "", 10, None), ("one char string", "x", 10, None),
                    ("non-ascii string", "héllo ☃", 10, None), ("string with NUL", "a\x00b", 10, None)])

    def test_vectors(self):
        real_dtype = np.float64 if self.backend.supports_float64() else np.float32
        self.check([("long vector", np.array([1, 2, 3]), 7, None), ("float vector", np.array([1.5, 2.5]), 9, None),
                    ("float vector with nan", np.array([1.5, NAN]), 9, None),
                    ("int32 vector", np.array([1, 2, 3], dtype=np.int32), 6, np.array([1, 2, 3])),
                    ("int16 vector", np.array([1, 2, 3], dtype=np.int16), 5, np.array([1, 2, 3])),
                    ("uint8 vector", np.array([1, 2, 3], dtype=np.uint8), 4, np.array([1, 2, 3])),
                    ("bool vector", np.array([True, False]), 1, np.array([1, 0])),
                    ("float32 vector", np.array([1.5, 2.5], dtype=np.float32), 8, np.array([1.5, 2.5], dtype=real_dtype)),
                    ("symbol list", [S("a"), S("b")], 11, np.array([S("a"), S("b")], dtype=object))])

    def test_general_lists(self):
        obj = lambda *xs: np.array(xs, dtype=object)  # noqa: E731
        self.check([("mixed", [1, S("a"), "s"], 0, obj(1, S("a"), "s")),
                    ("strings", ["ab", "cd"], 0, obj("ab", "cd")),
                    ("int and float", [1, 2.5], 0, obj(1, 2.5)),
                    ("nested", [[1, 2], [3, 4]], 0, np.array([[1, 2], [3, 4]])),
                    ("matrix", np.array([[1, 2], [3, 4]]), 0, None),
                    ("ragged", [[1, 2], [3]], 0, obj(np.array([1, 2]), np.array([3]))),
                    ("empty list", [], 0, np.array([], dtype=np.float64)),
                    ("empty array", np.array([]), 0, np.array([], dtype=np.float64))])

    def test_dictionaries(self):
        self.check([("dict", {S("a"): 1, S("b"): 2}, 99, None),
                    ("dict of mixed values", {S("a"): np.array([1, 2]), S("b"): "x"}, 99, None),
                    ("dict with string keys", {"a": 1}, 99, None),
                    ("nested dict", {S("a"): {S("c"): 1}}, 99, None)])

    def test_unit_value(self):
        self.assertEqual(self.q("t", KLONG_UNDEFINED), 101)
        self.assertIs(self.q("i", KLONG_UNDEFINED), KLONG_UNDEFINED)

    def test_klong_function_literal_is_a_lambda_q_can_run(self):
        self.klong["fn"] = self.kg(self.klong, "{x+1}")
        self.assertEqual(self.q("t", self.klong["fn"]), 100)
        self.assertEqual(self.q(self.klong["fn"], 5), 6)                   # applied remotely by q
        self.assertEqual(self.q("{x[5]}", self.klong["fn"]), 6)
        back = self.q("i", self.klong["fn"])                               # q returns it with its "kg" context
        self.klong["back"] = back
        self.assertEqual(self.kg(self.klong, "back(5)"), 6)                # and it is a function KlongPy can call

    def test_klong_function_that_is_not_q_syntax_is_refused_by_q(self):
        """VERIFY 2: PeachQ refuses a ``kg`` lambda whose text it cannot parse, with the signal ``badfunc``."""
        self.klong["fn"] = self.kg(self.klong, "{:[x>1;1;0]}")
        e = self.q_error("i", self.klong["fn"])
        self.assertEqual(e.text, "badfunc")
        self.assertEqual(self.q("1+1"), 2)                                 # the connection survives


@NEED_PEACHQ
class TestKlongClientRequests(KlongClientCase):
    """Remote set and get, calls with arguments, errors, async, size, timeouts."""

    def test_clid_set_and_get(self):
        self.kg(self.klong, "d::.clid(c)")
        self.klong["pair"] = [S("aa"), 42]
        self.kg(self.klong, "d,pair")
        self.assertEqual(self.q("aa"), 42)                       # it was set on the q side
        self.assertEqual(self.kg(self.klong, "d?:aa"), 42)
        self.klong["pair"] = [S("vv"), np.array([1.5, 2.5])]
        self.kg(self.klong, "d,pair")
        self.assertSame(self.q("vv"), np.array([1.5, 2.5]))
        self.assertSame(self.kg(self.klong, "d?:vv"), np.array([1.5, 2.5]))
        self.klong["pair"] = [S("txt"), "text"]
        self.kg(self.klong, "d,pair")
        self.assertEqual(self.q("type txt"), 10)

    def test_clid_get_of_a_q_function(self):
        self.q("fz:{x+1}")
        self.assertIsInstance(self.kg(self.klong, "c(:fz)"), KdbLambda)

    def test_clid_set_of_a_klong_function_comes_back_as_a_proxy(self):
        self.kg(self.klong, "d::.clid(c)")
        self.klong["pair"] = [S("kf"), self.kg(self.klong, "{x+1}")]
        self.kg(self.klong, "d,pair")
        self.assertEqual(self.q("kf 10"), 11)
        proxy = self.kg(self.klong, "d?:kf")                     # a `kg` lambda answered to a lookup becomes a proxy
        self.assertIsInstance(proxy, KGRemoteFnProxy)
        self.assertEqual(self.kg(self.klong, "t::d?:kf;t(10)"), 11)

    def test_set_through_the_call_envelope(self):
        self.assertIs(self.q("set", S("bb"), 7).__class__, KGSym)      # q's set answers the name
        self.assertEqual(self.q("bb"), 7)

    def test_call_by_symbol_name(self):
        self.q("foo:{x*y}")
        self.assertEqual(self.q(S("foo"), 3, 4), 12)
        self.assertEqual(self.q(S("foo"), 1.5, 2), 3.0)

    def test_call_by_name_in_a_string(self):
        self.q("foo:{x*y}")
        self.assertEqual(self.q("foo", 3, 4), 12)
        self.assertEqual(self.q("neg", 5), -5)
        self.assertEqual(self.q("{x+y}", 1, 2), 3)

    def test_call_with_vector_and_string_arguments(self):
        self.assertEqual(self.q("count", np.array([1, 2, 3])), 3)
        self.assertSame(self.q("{x,y}", np.array([1, 2]), np.array([3, 4])), np.array([1, 2, 3, 4]))
        self.assertEqual(self.q("{x,y}", "ab", "cd"), "abcd")
        self.assertSame(self.q("reverse", np.array([1, 2, 3])), np.array([3, 2, 1]))

    def test_call_with_a_dict_argument(self):
        self.assertSame(self.q("key", {S("a"): 1, S("b"): 2}), np.array([S("a"), S("b")], dtype=object))

    def test_call_with_a_symbol_argument_stays_a_symbol(self):
        self.assertEqual(self.q("{type x}", S("abc")), -11)
        self.assertEqual(self.q("{type x}", [S("a"), S("b")]), 11)

    def test_call_niladic_function_sends_unit_argument(self):
        self.q("nil:{42}")
        self.assertEqual(self.q(S("nil"), KLONG_UNDEFINED), 42)

    def test_projection_returned_by_q_is_opaque(self):
        self.q("foo:{x*y}")
        p = self.q(S("foo"), 3)
        self.assertEqual(p.type, 104)
        self.assertEqual(self.q(p, 5), 15)

    def test_q_error_signal(self):
        e = self.q_error("1+`a")
        self.assertEqual((e.kind, e.text), ("remote", "type"))
        self.assertTrue(e.remote)
        e = self.q_error("'oops")
        self.assertEqual((e.kind, e.text), ("remote", "oops"))
        e = self.q_error("nosuchvar")
        self.assertEqual(e.text, "nosuchvar")
        e = self.q_error(S("nosuchfn"), 1)
        self.assertEqual(e.text, "nosuchfn")
        e = self.q_error("1 2 3+1 2")
        self.assertEqual(e.text, "length")
        self.assertEqual(self.q("1+1"), 2)                       # a q error leaves the connection usable

    def test_error_inside_a_called_function(self):
        self.q("bad:{x+`a}")
        e = self.q_error(S("bad"), 1)
        self.assertEqual(e.text, "type")
        self.assertEqual(self.q("2*3"), 6)

    def test_async_callback_gets_the_result(self):
        self.kg(self.klong, "res::0;cb::{res::x};af::.async(c;cb)")
        self.assertEqual(self.kg(self.klong, 'af("2+3")'), 1)
        self.assertTrue(wait_until(lambda: self.klong("res") == 5, 10))
        self.klong["req"] = ["count", np.array([1, 2, 3])]
        self.kg(self.klong, "af(req)")
        self.assertTrue(wait_until(lambda: self.klong("res") == 3, 10))

    def test_async_on_a_proxy_and_on_an_error(self):
        self.kg(self.klong, "d::.clid(c)")
        self.klong["pair"] = [S("kf"), self.kg(self.klong, "{x+1}")]
        self.kg(self.klong, "d,pair")
        self.kg(self.klong, "t::d?:kf;res::0;cb::{res::x};af::.async(t;cb)")
        self.assertEqual(self.kg(self.klong, "af(10)"), 1)
        self.assertTrue(wait_until(lambda: self.klong("res") == 11, 10))
        self.kg(self.klong, "res::0;af2::.async(c;cb)")
        self.klong["req"] = "1+`a"
        self.assertEqual(self.kg(self.klong, "af2(req)"), 1)       # a q error: logged, the callback is not called
        self.klong["req"] = "7+1"
        self.kg(self.klong, "af2(req)")
        self.assertTrue(wait_until(lambda: self.klong("res") == 8, 10))

    def test_async_message_on_the_wire(self):
        """A true async message (type 0), sent by hand: nothing is answered, errors are not reported, order is kept."""
        s = raw_connect(self.peachq.port)
        try:
            send_msg(s, 0, encode("az:5"))
            send_msg(s, 0, encode("1+`a"))                           # fails; no reply, connection stays up
            send_msg(s, 0, encode("az:az+1"))
            send_msg(s, 1, encode("az"))
            kind, value = read_msg(s)
            self.assertEqual((kind, value), (2, 6))                  # one response only, and it is the sync one
        finally:
            s.close()

    def test_large_vectors(self):
        self.q("sm:{sum x}")
        big = np.arange(2_000_000)
        t0 = time.monotonic()
        self.assertEqual(self.q("count", big), 2_000_000)
        self.assertEqual(self.q("sm", big), int(big.sum()))
        self.assertSame(self.q("{x}", big), big)
        floats = np.arange(300_000) * 1.5
        self.assertSame(self.q("{x}", floats), floats)
        r = self.q("til 1000000")
        self.assertEqual((len(r), int(r[-1])), (1000000, 999999))
        self.assertEqual(len(self.q("1000000#`abc")), 1000000)
        self.assertEqual(self.q("count", "x" * 100000), 100000)
        self.assertLess(time.monotonic() - t0, TIMEOUT)

    def test_reply_over_the_message_limit_closes_the_connection(self):
        self.klong[".system"]["ipc_max_message"] = 1000
        self.connect(name="small")
        self.assertSame(self.q("til 10", name="small"), np.arange(10))
        with self.assertRaises(KlongSystemError) as cm:
            self.q("til 1000", name="small")                     # 8014 bytes
        self.assertIn("exceeds the limit", str(cm.exception))
        with self.assertRaises(KlongSystemError):
            self.q("til 10", name="small")                       # the connection is gone
        self.assertEqual(self.q("1+1"), 2)                       # the other connection (made before) is unaffected

    def test_many_requests_in_a_row(self):
        for i in range(200):
            self.assertEqual(self.q("{x+1}", i), i + 1)

    def test_close_handle(self):
        self.assertEqual(self.kg(self.klong, ".clic(c)"), 1)
        self.assertTrue(self.peachq.alive())

    def test_connection_refused_returns_a_disconnected_handle(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))  # Keep the port bound without listening.
            port = reserved.getsockname()[1]
            provider = partial(HostPortConnectionProvider, max_retries=1, retry_delay=0)
            with patch("klongpy.sys_fn_ipc.HostPortConnectionProvider", provider):
                handle = self.kg(self.klong, f'x::.cli("127.0.0.1:{port}")')
            self.assertTrue(callable(handle))
            with self.assertRaises(KlongSystemError) as refused:
                self.kg(self.klong, 'x("1+1")')
            self.assertEqual(str(refused.exception), "connection not established")


@NEED_PEACHQ
class TestKlongClientTimeout(Case):
    """The call timeout, on a PeachQ of its own (q serves one request at a time, so a sleep blocks it)."""

    def test_call_timeout_closes_the_connection(self):
        server = PeachQServer(EXE)
        self.addCleanup(server.stop)
        klong = self.repl()
        self.kg(klong, f"c::.cli({server.port})")
        self.assertEqual(self.kg(klong, ".clit(c;1)"), 1)
        klong["req"] = 'system "sleep 3"'
        t0 = time.monotonic()
        with self.assertRaises(KlongIPCTimeoutException) as cm:
            self.kg(klong, "c(req)")
        self.assertIsInstance(cm.exception, TimeoutError)
        self.assertLess(time.monotonic() - t0, 3)
        klong["req"] = "1+1"
        with self.assertRaises(KlongSystemError):                 # the connection was closed: there are no request ids
            self.kg(klong, "c(req)")
        klong["req"] = "1+1"
        self.kg(klong, f"c2::.cli({server.port})")                 # a new connection works (once q has woken up)
        self.assertEqual(self.kg(klong, "c2(req)", 15), 2)

    def test_call_that_finishes_within_the_timeout(self):
        server = PeachQServer(EXE)
        self.addCleanup(server.stop)
        klong = self.repl()
        self.kg(klong, f"c::.cli({server.port});.clit(c;5)")
        klong["req"] = 'system "sleep 1"'
        self.kg(klong, "c(req)")
        klong["req"] = "1+1"
        self.assertEqual(self.kg(klong, "c(req)"), 2)


@NEED_PEACHQ
class TestKlongClientHandshake(Case):
    """Connection forms and the handshake against PeachQ."""

    @classmethod
    def setUpClass(cls):
        cls.peachq = PeachQServer(EXE)
        cls.addClassCleanup(cls.peachq.stop)

    def test_host_port_and_user_password_forms(self):
        klong = self.repl()
        p = self.peachq.port
        for target in (f'"127.0.0.1:{p}"', f'"localhost:{p}"', f'"127.0.0.1:{p}:user:secret"', f'"127.0.0.1:{p}:user"'):
            with self.subTest(target):
                self.kg(klong, f"h::.cli({target})")
                klong["req"] = "2*3"
                self.assertEqual(self.kg(klong, "h(req)"), 6)
                self.kg(klong, ".clic(h)")

    def test_two_connections_at_once(self):
        a, b = self.repl(), self.repl()
        self.kg(a, f"c::.cli({self.peachq.port})")
        self.kg(b, f"c::.cli({self.peachq.port})")
        a["req"], b["req"] = "10+1", "20+2"
        self.assertEqual(self.kg(a, "c(req)"), 11)
        self.assertEqual(self.kg(b, "c(req)"), 22)

    def test_password_file_server(self):
        """A PeachQ started with ``-u file`` checks ``user:password``; KlongPy's credentials are what it expects."""
        import tempfile
        d = tempfile.mkdtemp(prefix="peachq-pw-")
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        path = os.path.join(d, "pw.txt")
        with open(path, "w") as f:
            f.write("bob:pw\n")
        server = PeachQServer(EXE, extra_args=("-u", path))
        self.addCleanup(server.stop)
        klong = self.repl()
        p = server.port
        self.kg(klong, f'h::.cli("127.0.0.1:{p}:bob:pw")')
        klong["req"] = "1+1"
        self.assertEqual(self.kg(klong, "h(req)"), 2)
        for target in (f'"127.0.0.1:{p}:bob:bad"', f'"127.0.0.1:{p}:alice:pw"', f'"127.0.0.1:{p}"', str(p)):
            with self.subTest(target), self.assertRaises(KlongSystemError):
                self.kg(klong, f"x::.cli({target});x(req)", 60)

    def test_what_klongpy_sends_in_its_handshake(self):
        """Run against a scripted peer, to show PeachQ is answering what KlongPy really sent (user, password, 3)."""
        fake = FakeKdbServer(lambda v: v)
        self.addCleanup(fake.stop)
        klong = self.repl()
        self.kg(klong, f'c::.cli("127.0.0.1:{fake.port}:bob:pw")')
        self.assertEqual(fake.greetings, [("bob", "pw", 3)])


@NEED_PEACHQ
@NEED_LAN
class TestCompressionWithPeachQ(KlongClientCase):
    """PeachQ, like kdb+, compresses large messages to peers that are not on this host (loopback is never compressed)."""

    def connect(self, host=None, name="c"):
        super().connect(host or LAN, name)

    def spy(self):
        """Count what the session layer compresses and decompresses (``klongpy.ipc.session`` binds both by name)."""
        ses = sys.modules["klongpy.ipc.session"]
        calls = {"compress": [], "decompress": []}
        real_c, real_d = ses.compress, ses.decompress

        def compress(msg, *a, **k):
            out = real_c(msg, *a, **k)
            calls["compress"].append((len(msg), len(out), out[2]))
            return out

        def decompress(msg, *a, **k):
            calls["decompress"].append(len(msg))
            return real_d(msg, *a, **k)
        ses.compress, ses.decompress = compress, decompress
        self.addCleanup(lambda: (setattr(ses, "compress", real_c), setattr(ses, "decompress", real_d)))
        return calls

    def test_compressed_messages_from_peachq_are_understood(self):
        calls = self.spy()
        r = self.q("til 100000")
        self.assertSame(r, np.arange(100000))
        self.assertTrue(calls["decompress"], "PeachQ did not compress a 800 kB reply to a non-local peer")
        self.assertSame(self.q("100000#0"), np.zeros(100000, dtype=np.int64))
        self.assertEqual(len(self.q("10000#`abc")), 10000)
        self.assertEqual(self.q("2500#\"a\""), "a" * 2500)

    def test_compressed_messages_from_klongpy_are_understood_by_peachq(self):
        calls = self.spy()
        self.klong.ipc_compress = True
        self.q("sm:{sum x}")
        w = np.arange(100000)
        self.assertEqual(self.q("sm", w), int(w.sum()))
        self.assertEqual(self.q("sm", np.zeros(100000, dtype=np.int64)), 0)
        f = np.random.default_rng(1).random(10000)
        self.assertSame(self.q("{x}", f), f)
        self.assertTrue([c for c in calls["compress"] if c[2] == 1], f"nothing was compressed: {calls['compress']}")

    def test_many_shapes_and_sizes_agree_both_ways(self):
        """Round trip through PeachQ at sizes around the compression threshold and well above it."""
        self.klong.ipc_compress = True
        calls = self.spy()
        self.q("i:{x}")
        rng = np.random.default_rng(7)
        sym = lambda n: np.array([S("s%d" % (i % 17)) for i in range(n // 4 + 1)], dtype=object)  # noqa: E731
        cases = []
        for n in (300, 2001, 2500, 70000):
            cases += [(f"zeros{n}", np.zeros(n, dtype=np.int64)), (f"arange{n}", np.arange(n)),
                      (f"rand{n}", rng.integers(0, 2 ** 62, n)), (f"randf{n}", rng.random(n)),
                      (f"small{n}", rng.integers(0, 5, n)), (f"str{n}", "the quick brown fox " * (n // 20 + 1)),
                      (f"syms{n}", sym(n)), (f"mixed{n}", [1, "ab", 2.5] * (n // 30 + 1))]
        for name, v in cases:
            with self.subTest(name):
                got = self.q("i", v)
                if isinstance(v, list):
                    self.assertEqual([x if not isinstance(x, np.generic) else x.item() for x in got.tolist()], v)
                elif isinstance(v, str):
                    self.assertEqual(got, v)
                else:
                    self.assertSame(got, v)
        self.assertTrue(calls["decompress"] and [c for c in calls["compress"] if c[2] == 1])
        for expr in ("1000000?100", "1000000#0", "1000000?1.0", "100000?`a`b`c`d", "100000?0x00ff",
                     "{x}each 3000#enlist (1;`a;\"xyz\")", "raze 1000#enlist \"hello world \""):
            with self.subTest(expr):
                self.assertGreater(len(self.q(expr)), 1000)

    def test_loopback_is_not_compressed_by_peachq(self):
        s = raw_connect(self.peachq.port)                      # 127.0.0.1
        try:
            send_msg(s, 1, encode("til 100000"))
            head, body = read_raw(s)
            self.assertEqual(head[2], 0)                       # not compressed
            self.assertEqual(len(body), 800006)
        finally:
            s.close()


VECTORS = os.path.join(os.path.dirname(__file__), "ipc_vectors", "vectors.json")


def _body_and_order(raw):
    """(body, little_endian) of a vector message, decompressed if it is compressed."""
    if raw[2] == 1:
        from klongpy.ipc.compress import decompress
        raw = decompress(raw)
    return raw[8:], raw[0] == 1


@NEED_PEACHQ
class TestSharedVectors(KlongClientCase):
    """PeachQ's ``-9!`` and ``-8!`` against ``tests/ipc_vectors/vectors.json`` (119 messages, 4 published by KX).

    Every message that is not a rejection case goes through ``-8!-9!`` in PeachQ.  What comes back must be the same
    bytes after the header (PeachQ always writes the async message type, so the header is not compared) for the
    round-trip entries, and the same value for the decode-only ones (big-endian, compressed, odd widths).
    Two kinds of entry are not acceptable to PeachQ, by design on both sides: an error object (``-9!`` of ``80...``
    is itself an error) and a lambda with context ``kg`` and display text (``badfunc``)."""

    def test_vectors_agree(self):
        with open(VECTORS) as f:
            entries = [e for e in json.load(f) if e["mode"] != "reject"]
        self.assertGreater(len(entries), 50)
        self.q("rt:{-8!-9!x}")
        checked = 0
        for e in entries:
            raw = bytes.fromhex(e["hex"])
            body, little = _body_and_order(raw)
            with self.subTest(e["name"]):
                if body[:1] == b"\x80" or body[:3] == b"dkg":
                    with self.assertRaises(KlongRemoteError):
                        self.q("rt", np.frombuffer(raw, dtype=np.uint8).copy())
                    continue
                replay = self.q("rt", np.frombuffer(raw, dtype=np.uint8).copy())
                out = bytes(np.asarray(self.backend.to_numpy(replay), dtype=np.uint8))
                if e["mode"] == "roundtrip":
                    self.assertEqual(out[8:].hex(), body.hex())
                else:
                    self.assertEqual(repr(decode(out[8:], True)), repr(decode(body, little)))
                checked += 1
        self.assertGreater(checked, 50)


# ----------------------------------------------------------------------------------------------------------
# PeachQ client -> KlongPy server
# ----------------------------------------------------------------------------------------------------------

#: name -> value the KlongPy server holds, and what q's ``-3!`` shows for it with its type number
SERVED = [
    ("vint", 5, "-7|5"), ("vfloat", 1.5, "-9|1.5"), ("vnan", NAN, "-9|0n"), ("vinf", math.inf, "-9|0w"),
    ("vninf", -math.inf, "-9|-0w"), ("vchar", KGChar("a"), '-10|"a"'), ("vstr", "abc", '10|"abc"'),
    ("vempty", "", '10|""'), ("vuni", "héllo", '10|"h\\303\\251llo"'), ("vsym", S("abc"), "-11|`abc"),
    ("vund", KLONG_UNDEFINED, "101|::"),
    ("vivec", np.array([1, 2, 3]), "7|1 2 3"), ("vfvec", np.array([1.5, 2.5]), "9|1.5 2.5"),
    ("vsvec", [S("a"), S("b")], "11|`a`b"), ("vmixed", [1, S("a"), "s"], '0|(1;`a;,"s")'),
    ("vstrs", ["ab", "cd"], '0|("ab";"cd")'), ("vnest", np.array([[1, 2], [3, 4]]), "0|(1 2;3 4)"),
    ("vrag", [[1, 2], [3]], "0|(1 2;,3)"), ("vlist0", [], "0|()"),
    ("vdict", {S("a"): 1, S("b"): 2}, "99|`a`b!1 2"),
    ("vdict2", {S("a"): np.array([1, 2]), S("b"): "x"}, '99|`a`b!(1 2;,"x")'),
    ("vnull", -2 ** 63, "-7|0N"), ("vfnan", np.array([1.5, NAN]), "9|1.5 0n"),
    ("vmax", 2 ** 63 - 1, "-7|0W"),
]


class PeachQClientCase(Case):
    """A KlongPy server in this process and q scripts, run by PeachQ, that connect to it."""

    def serve(self, setup=None, system=None, **values):
        self.server = self.repl()
        for k, v in (system or {}).items():
            self.server[".system"][k] = v
        for k, v in {**{n: v for n, v, _ in SERVED}, **values}.items():
            self.server[k] = v
        self.kg(self.server, "idf::{x}; sq::{x*x}; add::{x+y}; nil::{42}; mk::{[a];a::x;a}")
        for line in (setup or "").split(";;"):
            if line.strip():
                self.kg(self.server, line)
        self.port = free_port()
        self.assertEqual(self.kg(self.server, f".srv({self.port})"), 1)

    def run_q(self, body, timeout=40, host="127.0.0.1", handle="hopen %PORT%"):
        script = f"h:{handle};\n".replace("%PORT%", str(self.port)).replace("%HOST%", host) + Q_PRELUDE + body + "\n\\\\\n"
        code, out, err = run_q(script, timeout)
        return parse_results(out), out + err


@NEED_PEACHQ
class TestPeachQClientEvaluates(PeachQClientCase):
    def test_string_is_evaluated(self):
        self.serve()
        r, out = self.run_q('''
rep["sum";"1+1"];
rep["fn";"sq(7)"];
rep["str";"\\"hello\\""];
rep["vec";"!5"];
rep["define";"newfn::{x+100}"];
rep["use";"newfn(1)"];
''')
        self.assertEqual(r["sum"], ("ok", -7, "2"), out)
        self.assertEqual(r["fn"], ("ok", -7, "49"), out)
        self.assertEqual(r["str"], ("ok", 10, '"hello"'), out)
        self.assertEqual(r["vec"], ("ok", 7, "0 1 2 3 4"), out)
        self.assertEqual(r["use"], ("ok", -7, "101"), out)
        self.assertEqual(self.kg(self.server, "newfn(1)"), 101)             # it ran in the KlongPy server

    def test_call_as_a_list(self):
        self.serve()
        r, out = self.run_q('''
rep["sq";(`sq;7)];
rep["add";(`add;3;4)];
rep["add str";("add";3;4)];
rep["vec";(`sq;1 2 3)];
rep["float";(`add;1.5;2.5)];
rep["strarg";(`idf;"abc")];
rep["symarg";(`idf;`abc)];
rep["symvec arg";(`idf;`a`b)];
rep["niladic";(`nil;::)];
rep["lookup";`sq];
rep["lookup value";`vint];
rep["set";(`set;`gv;1 2 3)];
rep["set sym";(`set;`gs;`abc)];
rep["get back";`gv];
''')
        self.assertEqual(r["sq"], ("ok", -7, "49"), out)
        self.assertEqual(r["add"], ("ok", -7, "7"), out)
        self.assertEqual(r["add str"], ("ok", -7, "7"), out)
        self.assertEqual(r["vec"], ("ok", 7, "1 4 9"), out)
        self.assertEqual(r["float"], ("ok", -9, "4f"), out)
        self.assertEqual(r["strarg"], ("ok", 10, '"abc"'), out)
        self.assertEqual(r["symarg"], ("ok", -11, "`abc"), out)
        self.assertEqual(r["symvec arg"], ("ok", 11, "`a`b"), out)
        self.assertEqual(r["niladic"], ("ok", -7, "42"), out)
        self.assertEqual(r["lookup"], ("ok", 100, "{x*x}"), out)
        self.assertEqual(r["lookup value"], ("ok", -7, "5"), out)
        self.assertEqual(r["get back"], ("ok", 7, "1 2 3"), out)
        self.assertSame(self.server["gv"], np.array([1, 2, 3]))
        self.assertSame(self.server["gs"], S("abc"))

    def test_a_list_of_symbols_is_a_call(self):
        """q sends ```f`a`` (a symbol vector) for ``(`f;`a)``: KlongPy treats a symbol vector of two or more as a call."""
        self.serve()
        r, out = self.run_q('rep["ok";(`idf;`abc)]; rep["set";(`set;`two;`xyz)];')
        self.assertEqual(r["ok"], ("ok", -11, "`abc"), out)
        self.assertEqual(r["set"][0], "ok", out)
        self.assertSame(self.server["two"], S("xyz"))

    def test_klong_error_arrives_as_a_q_signal_with_klongpys_text(self):
        self.serve()
        r, out = self.run_q('''
rep["type";"1+\\"a\\""];
rep["name";"nosuchvar+1"];
rep["name call";(`nosuchfn;1)];
rep["parse";"1+"];
rep["after";"1+1"];
''')
        self.assertEqual(r["type"][0], "err", out)
        self.assertTrue(r["type"][1].startswith("type: cannot apply + (Plus) to integer and string"), r["type"])
        self.assertEqual(r["name call"], ("err", "name: symbol not found: nosuchfn"), out)
        self.assertEqual(r["name"][0], "err", out)
        self.assertEqual(r["parse"][0], "err", out)
        self.assertEqual(r["after"], ("ok", -7, "2"), out)              # the connection is still good

    def test_async_messages(self):
        self.serve()
        r, out = self.run_q('''
asy["ga::5"];
asy["ga::ga+1"];
asy[(`set;`gb;`abc)];
asy["1+\\"a\\""];
rep["order";"ga"];
''')
        self.assertEqual(r["order"], ("ok", -7, "6"), out)               # applied in order, before the sync request
        self.assertSame(self.server["gb"], S("abc"))

    def test_many_async_and_sync_messages_keep_their_order(self):
        self.serve()
        self.kg(self.server, "cnt::0")
        r, out = self.run_q('''
do[500; asy["cnt::cnt+1"]];
rep["count";"cnt"];
tot:0; do[300; tot+:h(`add;1;2)];
-1 "R|loop|ok|-7|",string tot;
''')
        self.assertEqual(r["count"], ("ok", -7, "500"), out)
        self.assertEqual(r["loop"], ("ok", -7, "900"), out)

    def test_request_over_the_message_limit_closes_the_connection(self):
        self.serve(system={"ipc_max_message": 1000})
        r, out = self.run_q('''
rep["small";(`idf;til 10)];
rep["big";(`idf;til 1000)];
''')
        self.assertEqual(r["small"][:2], ("ok", 7), out)
        self.assertEqual(r["big"][0], "err", out)

    def test_async_error_is_reported_to_the_srv_e_hook(self):
        self.serve(".srv.e::{err::y}")
        self.kg(self.server, 'err::""')
        r, out = self.run_q('asy["1+\\"a\\""]; rep["sync";"1+1"];')
        self.assertEqual(r["sync"], ("ok", -7, "2"), out)
        self.assertTrue(wait_until(lambda: str(self.server("err")).startswith("type: cannot apply +"), 10),
                        self.server("err"))

    def test_hooks_see_the_connection(self):
        self.serve(".srv.o::{opened::opened+1;x};;.srv.c::{closed::closed+1;x}")
        self.kg(self.server, "opened::0;closed::0")
        r, out = self.run_q('rep["x";"1+1"]; hclose h;')
        self.assertEqual(r["x"], ("ok", -7, "2"), out)
        self.assertTrue(wait_until(lambda: self.server("opened") == 1 and self.server("closed") == 1, 10),
                        (self.server("opened"), self.server("closed")))

    def test_credentials_checked_by_srv_auth(self):
        self.serve('.srv.auth::{:[(x="bob")&(y="pw");1;0]}')
        r, out = self.run_q('rep["ok";"1+1"]', handle="hopen `:127.0.0.1:%PORT%:bob:pw")
        self.assertEqual(r["ok"], ("ok", -7, "2"), out)
        for creds in (":bob:bad", ""):
            with self.subTest(creds):
                code, text, err = run_q(f"r:@[hopen;`:127.0.0.1:{self.port}{creds};{{\"REFUSED \",x}}];\n"
                                        f"-1 (-3!r);\n\\\\\n", 30)
                self.assertIn("REFUSED", text + err)

    def test_q_lambda_as_callee_is_not_applied(self):
        """(c) only a lambda with context ``kg`` is parsed and applied; q's own lambda comes back unchanged
        (kdb+ would apply it: ``h({x*y};6;7)`` is 42 there)."""
        self.serve()
        r, out = self.run_q('rep["lam";({x*y};6;7)]; rep["lam as arg";(`idf;{x*y})];')
        self.assertEqual(r["lam"], ("ok", 0, "({x*y};6;7)"), out)
        self.assertEqual(r["lam as arg"], ("ok", 100, "{x*y}"), out)


@NEED_PEACHQ
class TestPeachQClientReceives(PeachQClientCase):
    """Each value kind returned by the KlongPy server, as q sees it."""

    def test_values_returned_to_q(self):
        self.serve()
        lines = "".join(f'rep["{name}";`{name}];\n' for name, _, _ in SERVED)
        r, out = self.run_q(lines)
        for name, _, want in SERVED:
            with self.subTest(name):
                typ, _, text = want.partition("|")
                self.assertEqual(r[name], ("ok", int(typ), text), out)

    def test_big_vector_returned(self):
        self.serve(vbig=np.arange(300000))
        r, out = self.run_q('x:h`vbig; -1 "R|big|ok|",(string type x),"|",string[count x]," ",string sum x;')
        self.assertEqual(r["big"], ("ok", 7, "300000 44999850000"), out)

    def test_functions_returned_to_q(self):
        """A function literal crosses as source text with context ``kg``; q accepts it when the text is valid q."""
        self.serve()
        r, out = self.run_q('''
f:h"sq"; -1 "R|call|ok|",(string type f),"|",string f 5;
g:h"add"; -1 "R|call2|ok|0|",string g[3;4];
rep["fn";"sq"];
rep["fn2";"add"];
rep["compose";"idf(sq)"];
''')
        self.assertEqual(r["call"], ("ok", 100, "25"), out)
        self.assertEqual(r["call2"], ("ok", 0, "7"), out)
        self.assertEqual(r["fn"], ("ok", 100, "{x*x}"), out)
        self.assertEqual(r["fn2"], ("ok", 100, "{x+y}"), out)

    def test_functions_q_cannot_parse_signal_badfunc(self):
        """VERIFY 2 (from the other side): a Klong function whose text is not q syntax, or a projection (display form
        text), is refused by PeachQ when it reads the reply; the signal is ``badfunc`` and the handle stays usable."""
        self.serve("cnd::{:[x>1;1;0]};;prj::add(1;)")
        r, out = self.run_q('''
rep["cond";"cnd"];
rep["proj";"prj"];
rep["after";"1+1"];
''')
        self.assertEqual(r["cond"], ("err", "badfunc"), out)
        self.assertEqual(r["proj"], ("err", "badfunc"), out)
        self.assertEqual(r["after"], ("ok", -7, "2"), out)

    def test_values_sent_by_q_arrive_as_klong_values(self):
        self.serve()
        cases = [("5", 5), ("5i", 5), ("5h", 5), ("1b", 1), ("0x2a", 42), ("1.5", 1.5), ("2.5e", 2.5), ("0n", NAN),
                 ("0w", math.inf), ("0N", -2 ** 63), ("0Wj", 2 ** 63 - 1), ("`abc", S("abc")), ('"a"', KGChar("a")),
                 ('"hello"', "hello"), ('""', ""), ("1 2 3", np.array([1, 2, 3])), ("1 2 3i", np.array([1, 2, 3])),
                 ("101b", np.array([1, 0, 1])), ("1.5 2.5", np.array([1.5, 2.5])),
                 ("`a`b", np.array([S("a"), S("b")], dtype=object)),
                 ("(1;`a;\"xy\")", np.array([1, S("a"), "xy"], dtype=object)),
                 ("(1 2;3 4)", np.array([[1, 2], [3, 4]])),
                 ("`a`b!1 2", {S("a"): 1, S("b"): 2}),
                 ("2024.01.15", 8780), ("2024.01.15D10:30:00.000000000", 758629800000000000),
                 ("12:30:00.000", 45000000), ("(::)", KLONG_UNDEFINED)]
        body = "".join(f'asy[(`set;`$"g{i}";{expr})];\n' for i, (expr, _) in enumerate(cases)) + 'rep["done";"1+1"];\n'
        r, out = self.run_q(body)
        self.assertEqual(r["done"], ("ok", -7, "2"), out)
        for i, (expr, want) in enumerate(cases):
            with self.subTest(expr):
                self.assertSame(self.server[f"g{i}"], want, expr)

    def test_values_sent_by_q_come_back_unchanged(self):
        self.serve()
        exprs = ["5", "5i", "5h", "1b", "0x2a", "1.5", "2.5e", "0n", "0w", "-0w", "0N", "0Wj", "`abc", '"a"', '"hello"',
                 '""', "1 2 3", "1.5 2.5", "`a`b`c", '(1;`a;"s")', "(1 2;3 4)", "(1 2;enlist 3)", "()", "(1;(2;(3;4)))",
                 '("ab";"cd")', "`a`b!1 2", '`a`b!(1 2;"x")', "1 2!3 4", "2024.01.15D10:30:00.000000000",
                 "2024.01.15", "12:30:00.000", "2024.01m", "12:30", "12:30:15", "0D01:15:00", "(::)", "{x+1}", "+", "(+/)"]
        body = "".join(f'rep["{i}";(`idf;{e})];\n' for i, e in enumerate(exprs))
        r, out = self.run_q(body)
        widened = {"5i": "-7|5", "5h": "-7|5", "1b": "-7|1", "0x2a": "-7|42", "2.5e": "-9|2.5",
                   "2024.01.15D10:30:00.000000000": "-7|758629800000000000", "2024.01.15": "-7|8780",
                   "12:30:00.000": "-7|45000000", "2024.01m": "-7|288", "12:30": "-7|750", "12:30:15": "-7|45015",
                   "0D01:15:00": "-7|4500000000000", "1 2 3i": "7|1 2 3"}
        for i, e in enumerate(exprs):
            with self.subTest(e):
                got = r[str(i)]
                self.assertEqual(got[0], "ok", out)
                if e in widened:
                    self.assertEqual(f"{got[1]}|{got[2]}", widened[e], out)
                elif e in ("{x+1}", "+", "(+/)"):
                    pass
                else:
                    self.assertEqual(got[2], run_show(e), out)

    def test_keyed_table_sent_by_q_comes_back_keyed(self):
        self.serve()
        r, out = self.run_q('rep["keyed";(`idf;([k:1 2]v:3 4))];')
        self.assertEqual(r["keyed"][:2], ("ok", 99), out)
        self.assertEqual(r["keyed"][2], run_show("([k:1 2]v:3 4)"), out)

    def test_q_table_sent_to_klongpy_comes_back_as_a_table(self):
        self.serve()
        r, out = self.run_q('rep["t";(`idf;([]a:1 2 3;b:1.5 2.5 3.5;s:`x`y`z))];')
        self.assertEqual(r["t"][:2], ("ok", 98), out)
        self.assertIn("`a`b`s!(1 2 3;1.5 2.5 3.5;", r["t"][2], out)

    def test_q_projection_and_derived_verbs_are_opaque_but_return(self):
        self.serve()
        r, out = self.run_q('rep["proj";(`idf;+[1])]; rep["derived";(`idf;(+/))]; rep["neg";(`idf;neg)];')
        self.assertEqual(r["proj"], ("ok", 104, "+[1]"), out)
        self.assertEqual(r["derived"], ("ok", 107, "+/"), out)
        self.assertEqual(r["neg"], ("ok", 101, "-:"), out)


def run_show(expr):
    """What q's own ``-3!`` shows for ``expr``, from a throw-away q script (so expected texts are not hand-written)."""
    code, out, err = run_q(f'-1 "SHOW|",(-3!{expr});\n\\\\\n', 20)
    for line in out.splitlines():
        if line.startswith("SHOW|"):
            return line[5:]
    raise AssertionError(f"no output for {expr}: {out!r} {err!r}")


@NEED_PEACHQ
@NEED_DB
class TestTablesWithKlongDb(PeachQClientCase):
    """klongpy.db tables (pandas) as kdb+ tables, both ways."""

    def setUp(self):
        import pandas as pd
        from klongpy.db import sys_fn_db
        from klongpy.ipc import codec
        previous = codec._table_factory
        codec.set_table_factory(sys_fn_db._table_from_wire)
        self.addCleanup(codec.set_table_factory, previous)
        self.db = sys_fn_db
        self.table = sys_fn_db.Table(pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "s": ["x", "y", "z"]}))

    def test_klongpy_table_received_by_q_as_a_table(self):
        self.serve(tbl=self.table)
        r, out = self.run_q('''
rep["tbl";`tbl];
rep["count";"#tbl"];
x:h`tbl; -1 "R|cols|ok|0|",(-3!cols x);
-1 "R|rows|ok|0|",(-3!x[`a]);
-1 "R|sel|ok|0|",(-3!select from x where a>1);
''')
        self.assertEqual(r["tbl"][:2], ("ok", 98), out)
        self.assertEqual(r["cols"][2], "`a`b`s", out)
        self.assertEqual(r["rows"][2], "1 2 3", out)
        self.assertIn("2 3", r["sel"][2], out)

    def test_table_received_from_q_is_a_klongpy_table(self):
        import pandas  # noqa: F401
        klong_table = self.table
        server = PeachQServer(EXE)
        self.addCleanup(server.stop)
        klong = self.repl()
        self.kg(klong, f"c::.cli({server.port})")
        klong["req"] = "([]a:1 2 3;b:1.5 2.5 3.5;s:`x`y`z)"
        t = self.kg(klong, "c(req)")
        self.assertIsInstance(t, self.db.Table)
        self.assertEqual(t.columns, ["a", "b", "s"])
        np.testing.assert_array_equal(t.get("a"), [1, 2, 3])
        np.testing.assert_array_equal(t.get("b"), [1.5, 2.5, 3.5])
        self.assertEqual([str(x) for x in t.get("s")], ["x", "y", "z"])
        klong["req"] = ["{type x}", klong_table]                          # and back to q as a table
        self.assertEqual(self.kg(klong, "c(req)"), 98)
        klong["req"] = ["{sum x`a}", klong_table]
        self.assertEqual(self.kg(klong, "c(req)"), 6)
        klong["req"] = ["{select from x where a>1}", klong_table]
        back = self.kg(klong, "c(req)")
        np.testing.assert_array_equal(back.get("a"), [2, 3])

    def test_datetime_column_is_a_timestamp_column(self):
        """A pandas datetime64 column is sent as a timestamp vector (type 12, int64 ns since 2000), NaT as 0Np.
        It comes back as the underlying integers: temporals decode to numbers."""
        import pandas as pd
        t = self.db.Table(pd.DataFrame({"a": [1, 2, 3], "t": pd.to_datetime(["2000-01-01", "2024-01-15 10:30:00", None],
                                                                            format="ISO8601")}))
        self.serve(tdt=t)
        r, out = self.run_q('''
rep["tdt";`tdt];
rep["after";"1+1"];
x:h`tdt;
-1 "R|ty|ok|0|",(-3!type x`t);
-1 "R|ts|ok|0|",(-3!x`t);
-1 "R|null|ok|0|",(-3!null x`t);
-1 "R|cast|ok|0|",(-3!`long$x`t);
-1 "R|date|ok|0|",(-3!`date$x`t);
''')
        self.assertEqual(r["tdt"][:2], ("ok", 98), out)
        self.assertEqual(r["ty"][2], "12h", out)
        self.assertEqual(r["ts"][2], "2000.01.01D00:00:00.000000000 2024.01.15D10:30:00.000000000 0N", out)      # PeachQ prints a null timestamp as 0N
        self.assertEqual(r["null"][2], "001b", out)
        self.assertEqual(r["cast"][2], "0 758629800000000000 0N", out)
        self.assertEqual(r["date"][2], "2000.01.01 2024.01.15 0N", out)
        self.assertEqual(r["after"], ("ok", -7, "2"), out)

    def test_timestamp_column_comes_back_intact(self):
        import pandas as pd
        sent = pd.to_datetime(["2000-01-01", "2024-01-15 10:30:00.123456789", "1999-12-31 23:59:59"], format="ISO8601")
        klong = self.repl()
        server = PeachQServer(EXE)
        self.addCleanup(server.stop)
        self.kg(klong, f"c::.cli({server.port})")
        klong["req"] = ["{x}", self.db.Table(pd.DataFrame({"t": sent}))]            # q echoes the table it was sent
        back = self.kg(klong, "c(req)")
        self.assertIsInstance(back, self.db.Table)
        col = np.asarray(back.get("t"))
        self.assertEqual(col.dtype, np.int64)
        self.assertEqual(col.tolist(), [0, 758629800123456789, -1000000000])
        recovered = pd.to_datetime(col) + (pd.Timestamp("2000-01-01") - pd.Timestamp(0))
        self.assertTrue((recovered == sent).all())
        klong["req"] = ["{type x`t}", self.db.Table(pd.DataFrame({"t": sent}))]
        self.assertEqual(self.kg(klong, "c(req)"), 12)
        klong["req"] = ["{type x}", np.array(["2000-01-02"], dtype="datetime64[D]")]    # a plain array too
        self.assertEqual(self.kg(klong, "c(req)"), 12)
        klong["req"] = ["{x}", np.array([1, 2], dtype="timedelta64[s]")]
        self.assertSame(self.kg(klong, "c(req)"), np.array([10 ** 9, 2 * 10 ** 9]))
        klong["req"] = ["{type x}", np.array([1, 2], dtype="timedelta64[s]")]
        self.assertEqual(self.kg(klong, "c(req)"), 16)


@NEED_PEACHQ
@NEED_LAN
class TestPeachQClientCompression(PeachQClientCase):
    def test_large_messages_both_ways_over_a_non_loopback_address(self):
        ses = sys.modules["klongpy.ipc.session"]
        seen = {"compress": [], "decompress": []}
        real_c, real_d = ses.compress, ses.decompress

        def compress(msg, *a, **k):
            out = real_c(msg, *a, **k)
            seen["compress"].append(out[2])
            return out

        def decompress(msg, *a, **k):
            seen["decompress"].append(len(msg))
            return real_d(msg, *a, **k)
        ses.compress, ses.decompress = compress, decompress
        self.addCleanup(lambda: (setattr(ses, "compress", real_c), setattr(ses, "decompress", real_d)))
        self.serve(vbig=np.arange(200000), vzero=np.zeros(200000, dtype=np.int64))
        self.server.ipc_compress = True
        self.kg(self.server, "sm::{+/x}")
        r, out = self.run_q('''
v:til 200000;
rep["sum";(`sm;v)];
rep["echo";(`idf;v)];
x:h`vbig; -1 "R|big|ok|",(string type x),"|",string[count x]," ",string sum x;
y:h`vzero; -1 "R|zero|ok|",(string type y),"|",string[count y]," ",string sum y;
''', handle='hopen `$":%HOST%:%PORT%"', host=LAN)
        self.assertEqual(r["sum"], ("ok", -7, "19999900000"), out)
        self.assertEqual(r["big"], ("ok", 7, "200000 19999900000"), out)
        self.assertEqual(r["zero"], ("ok", 7, "200000 0"), out)
        self.assertEqual(r["echo"][:2], ("ok", 7), out)
        self.assertTrue(seen["decompress"], "PeachQ sent nothing compressed")
        self.assertIn(1, seen["compress"], "KlongPy did not compress a large reply to a non-local peer")


@NEED_PEACHQ
class TestPeachQClientHandshake(Case):
    """What PeachQ sends as a client, seen by a scripted peer."""

    def test_greeting_and_message_kinds(self):
        fake = FakeKdbServer(lambda v: 7)
        self.addCleanup(fake.stop)
        code, out, err = run_q(f'h:hopen {fake.port};\nr:h"1+1";\nneg[h] (`g;`a);\nr2:h(`f;1 2);\n-1 "R|r|ok|",(-3!r);\n'
                               f'-1 "R|r2|ok|",(-3!r2);\n\\\\\n', 30)
        self.assertEqual(len(fake.greetings), 1, out + err)
        user, password, capability = fake.greetings[0]
        self.assertEqual((user, password, capability), ("", "", 3))
        kinds = [(k, type(v).__name__) for k, v in fake.requests]
        self.assertEqual(kinds[0], ("sync", "str"))
        self.assertEqual(fake.requests[0][1], "1+1")
        self.assertEqual(fake.requests[1][0], "async")                        # neg[h]: a two-symbol list is a symbol vector
        self.assertEqual([str(x) for x in fake.requests[1][1]], ["g", "a"])
        self.assertEqual(fake.requests[2][0], "sync")
        self.assertEqual(str(fake.requests[2][1][0]), "f")
        self.assertIn("R|r|ok|7", out)


class TestSetWithASymbolCallee(Case):
    """Found by the PeachQ client tests, needs no PeachQ.  A q client writes ``h(`set;`a;1)`` (a symbol vector, or a
    general list whose first item is the symbol ``set``) as readily as ``h("set";`a;1)``; both assign on a KlongPy
    server.  Until the check in ``evaluate_request`` compared ``KGSym("set") == "set"`` (always False) only the char
    vector form assigned and the symbol form failed with ``name: symbol not found: set``."""

    def call(self, port, items, as_symbol_vector=False):
        s = raw_connect(port)
        try:
            send_msg(s, 1, encode(items, 64) if as_symbol_vector else encode_call(items, 64))
            head, body = read_raw(s)
            return head[1], decode(body, True)
        finally:
            s.close()

    def test_set_with_a_symbol_or_a_string_callee(self):
        server = self.repl()
        port = free_port()
        self.assertEqual(self.kg(server, f".srv({port})"), 1)
        for callee in (S("set"), "set"):
            with self.subTest(type(callee).__name__):
                kind, value = self.call(port, [callee, S("gx"), 5])
                self.assertEqual(kind, 2)
                self.assertIs(value, KLONG_UNDEFINED)
                self.assertEqual(server["gx"], 5)
                server["gx"] = 0
        kind, value = self.call(port, [S("set"), S("gy"), S("abc")], as_symbol_vector=True)     # as q sends it
        self.assertEqual((kind, str(server["gy"])), (2, "abc"))

    def test_a_program_that_defines_set_keeps_its_own(self):
        server = self.repl()
        self.kg(server, "set::{x+y}")
        port = free_port()
        self.assertEqual(self.kg(server, f".srv({port})"), 1)
        self.assertEqual(self.call(port, [S("set"), 3, 4]), (2, 7))


class TestWireFloatPrecision(PeachQClientCase):
    def test_forwarding_double_preserves_bits(self):
        wide = 1.0 + 2 ** -52
        vector = np.array([wide, 1e100, -0.0], dtype=np.float64)
        self.serve("last::{[a];a::0;x}", vwide=wide, vwides=vector)
        expected_atom = struct.pack("<bd", -9, wide)
        expected_vector = struct.pack("<BBiddd", 9, 0, 3, wide, 1e100, -0.0)
        conn = raw_connect(self.port)
        self.addCleanup(conn.close)

        for request, expected in (
            (encode(S("vwide")), expected_atom),
            (encode(S("vwides")), expected_vector),
            (encode_call([S("idf"), wide]), expected_atom),
            (encode_call([S("idf"), vector]), expected_vector),
            (encode_call([S("last"), wide]), expected_atom),
            (encode_call([S("set"), S("from_peer"), wide]), encode(KLONG_UNDEFINED)),
            (encode(S("from_peer")), expected_atom),
        ):
            with self.subTest(request=request):
                send_msg(conn, 1, request)
                header, response = read_raw(conn)
                self.assertEqual(header[1], 2)
                self.assertEqual(response, expected)

    def test_arithmetic_uses_backend_precision_and_device(self):
        offset = 2 ** -25
        self.serve("delta::{x-y}", nearone=1.0 + offset)
        expected = 0.0 if self.public_float is np.float32 else offset
        # These are ordinary Klong evaluations with bound inputs, so the
        # operand reads still obey the configured compute precision.
        self.assertSame(self.server["nearone"], 1.0 + offset)
        self.assertSame(self.kg(self.server, "nearone-1.0"), expected)
        dtype = np.dtype(self.public_float)
        self.assertSame(
            self.kg(self.server, "[1.0 1.0]+nearone-1.0"),
            np.array([1.0 + expected, 1.0 + expected], dtype=dtype),
        )
        conn = raw_connect(self.port)
        self.addCleanup(conn.close)
        send_msg(conn, 1, encode("nearone-1.0"))
        header, response = read_raw(conn)
        self.assertEqual(header[1], 2)
        fmt, code = ("<bf", -8) if self.public_float is np.float32 else ("<bd", -9)
        self.assertEqual(response, struct.pack(fmt, code, expected))
        send_msg(conn, 1, encode_call([S("delta"), 1.0 + offset, 1.0]))
        header, response = read_raw(conn)
        self.assertEqual(header[1], 2)
        self.assertEqual(response, struct.pack("<bd", -9, expected))


# ----------------------------------------------------------------------------------------------------------
# VERIFY items (docs/ipc_capabilities.md, docs/ipc_capabilities.md).  PeachQ's answers, not KX's.
# ----------------------------------------------------------------------------------------------------------

class RawConn:
    """A socket after the handshake; ``reply`` is the capability byte the peer answered (empty if it closed)."""

    def __init__(self, sock, reply):
        self._sock = sock
        self.reply = reply

    def __getattr__(self, name):
        return getattr(self._sock, name)


def raw_connect(port, host="127.0.0.1", capability=3, hello=None, timeout=10):
    s = socket.create_connection((host, port), timeout=timeout)
    s.settimeout(timeout)
    s.sendall(hello if hello is not None else b"user:pw" + bytes([capability, 0]))
    return RawConn(s, s.recv(1))


def send_msg(s, kind, body, little=True):
    s.sendall(bytes([1 if little else 0, kind, 0, 0]) + (8 + len(body)).to_bytes(4, "little" if little else "big") + body)


def read_exact(s, n):
    b = b""
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise EOFError("connection closed")
        b += c
    return b


def read_raw(s):
    head = read_exact(s, 8)
    n = int.from_bytes(head[4:], "little" if head[0] == 1 else "big")
    return head, read_exact(s, n - 8)


def read_msg(s):
    head, body = read_raw(s)
    if head[2] == 1:
        from klongpy.ipc.compress import decompress
        full = decompress(head + body)
        head, body = full[:8], full[8:]
    return head[1], decode(body, head[0] == 1)


@NEED_PEACHQ
class TestVerifyItems(Case):
    """Answers to the VERIFY list, as PeachQ answers them.  Labelled in each docstring; none of this is KX's behaviour."""

    @classmethod
    def setUpClass(cls):
        cls.peachq = PeachQServer(EXE)
        cls.addClassCleanup(cls.peachq.stop)

    def ask(self, s, text):
        send_msg(s, 1, encode(text))
        head, body = read_raw(s)
        return head, body

    def test_1_set_through_a_call_list_assigns(self):
        """PeachQ: ``h("set";`a;1)`` assigns, and answers the symbol ``a`` (as q's set does), not ``::``."""
        s = raw_connect(self.peachq.port)
        try:
            send_msg(s, 1, encode_call(["set", S("va"), 1], 64))
            head, body = read_raw(s)
            self.assertEqual(head[1], 2)
            self.assertEqual(bytes(body), bytes([0xf5]) + b"va\x00")
            self.assertEqual(self.ask(s, "va")[1], bytes([0xf9]) + (1).to_bytes(8, "little"))
        finally:
            s.close()

    def test_1b_symbol_names_that_are_q_keywords_do_not_resolve(self):
        """PeachQ: ``(`neg;5)`` and ``(`set;`a;1)`` (the callee a symbol, a keyword) are 'neg and 'set errors; a char
        vector callee works.  (c) Real kdb+ may differ; KlongPy sends the call as ``("set";...)``, which works."""
        s = raw_connect(self.peachq.port)
        try:
            send_msg(s, 1, encode_call([S("neg"), 5], 64))
            head, body = read_raw(s)
            self.assertEqual(bytes(body), b"\x80neg\x00")
            send_msg(s, 1, encode_call(["neg", 5], 64))
            self.assertEqual(read_raw(s)[1], bytes([0xf9]) + (-5).to_bytes(8, "little", signed=True))
        finally:
            s.close()

    def test_2_a_kg_context_lambda_is_accepted_only_if_its_text_is_q(self):
        """PeachQ: a type 100 lambda with context ``kg`` whose text is valid q is a function (``type`` 100, callable);
        any other text (Klong syntax, a display form such as ``:monad``, empty text) is ``badfunc`` and the connection
        stays up.  The context is kept: a lambda with context ``kg`` comes back with context ``kg``."""
        s = raw_connect(self.peachq.port)
        try:
            def lam(ctx, text):
                return bytes([100]) + ctx.encode() + b"\x00" + bytes([10, 0]) + len(text).to_bytes(4, "little") + text.encode()

            def call(*items):
                body = bytes([0, 0]) + len(items).to_bytes(4, "little") + b"".join(items)
                send_msg(s, 1, body)
                return read_raw(s)[1]
            long = lambda n: bytes([0xf9]) + n.to_bytes(8, "little")  # noqa: E731
            self.assertEqual(call(lam("kg", "{x+1}"), long(5)), long(6))
            self.assertEqual(call(lam("", "{x+1}"), long(5)), long(6))
            self.assertEqual(call(lam("zz", "{x+1}"), long(5)), long(6))                      # any context
            self.assertEqual(call(lam("", "{x}"), lam("kg", "{x+1}")), lam("kg", "{x+1}"))     # the context is kept
            for text in ("{:[x>1;1;0]}", ":monad", ""):
                self.assertEqual(call(lam("kg", text), long(5)), b"\x80badfunc\x00", text)
            self.assertEqual(self.ask(s, "1+1")[1], long(2))
        finally:
            s.close()

    def test_3_compression_in_both_directions(self):
        """PeachQ decompresses a request that KlongPy compressed (even on loopback), and answers a loopback request
        uncompressed.  (Compressed replies to a non-local peer are in ``TestCompressionWithPeachQ``.)"""
        from klongpy.ipc.compress import compress
        v = encode_call(["count", np.zeros(10000, dtype=np.int64)], 64)
        msg = bytes([1, 1, 0, 0]) + (8 + len(v)).to_bytes(4, "little") + v
        packed = compress(msg)
        self.assertEqual(packed[2], 1)
        s = raw_connect(self.peachq.port)
        try:
            s.sendall(packed)
            head, body = read_raw(s)
            self.assertEqual((head[2], bytes(body)), (0, bytes([0xf9]) + (10000).to_bytes(8, "little")))
            send_msg(s, 1, encode("1000#0"))
            self.assertEqual(read_raw(s)[0][2], 0)                                         # loopback: not compressed
        finally:
            s.close()

    def test_4_assignment_answers_the_generic_null(self):
        """PeachQ: ``h"a:1"`` answers ``(::)``, bytes ``65 00``."""
        s = raw_connect(self.peachq.port)
        try:
            head, body = self.ask(s, "a:1")
            self.assertEqual(bytes(body), b"\x65\x00")
            self.assertEqual(self.ask(s, "(::)")[1], b"\x65\x00")
        finally:
            s.close()

    def test_5_null_and_infinity_bit_patterns(self):
        """PeachQ sends ``0n`` as ``fff8000000000000`` (negative quiet NaN, the x86 ``0/0.0``), ``0w`` as ``7ff0...``,
        ``0N`` as INT64_MIN, ``0Wj`` as INT64_MAX, int and short nulls as the type's minimum, a real NaN as ``ffc00000``
        (``0Ne``) -- as the table in ref section 4 says (the NaN pattern there was marked unconfirmed)."""
        s = raw_connect(self.peachq.port)
        try:
            want = {"0n": "f7" + "000000000000f8ff", "0w": "f7" + "000000000000f07f", "-0w": "f7" + "000000000000f0ff",
                    "0N": "f9" + "0000000000000080", "0Wj": "f9" + "ffffffffffffff7f", "-0Wj": "f9" + "0100000000000080",
                    "0Ni": "fa" + "00000080", "0Nh": "fb" + "0080", "0Np": "f4" + "0000000000000080",
                    "0Wp": "f4" + "ffffffffffffff7f", "0Nd": "f2" + "00000080", "0Nz": "f1" + "000000000000f8ff"}
            for expr, hexed in want.items():
                with self.subTest(expr):
                    self.assertEqual(self.ask(s, expr)[1].hex(), hexed)
            real_nan = self.ask(s, "0Ne")[1].hex()
            self.assertEqual(real_nan[:2], "f8")
            self.assertTrue(math.isnan(struct.unpack("<f", bytes.fromhex(real_nan[2:]))[0]))
        finally:
            s.close()

    def test_6_q_as_a_client_calls_klongpy(self):
        """PeachQ as client: ``hopen``, ``h"1+1"`` against ``.srv`` is the long 2; ``h(`f;1;2)`` calls a Klong function."""
        case = PeachQClientCase()
        case.addCleanup = self.addCleanup
        case.repl, case.kg, case.assertEqual, case.assertSame = self.repl, self.kg, self.assertEqual, self.assertSame
        case.serve()
        r, out = case.run_q('rep["sum";"1+1"]; rep["call";(`add;1;2)];')
        self.assertEqual(r["sum"], ("ok", -7, "2"), out)
        self.assertEqual(r["call"], ("ok", -7, "3"), out)

    def test_7_klong_error_arrives_in_q_as_a_signal_with_klongpys_text(self):
        case = PeachQClientCase()
        case.addCleanup = self.addCleanup
        case.repl, case.kg, case.assertEqual, case.assertSame = self.repl, self.kg, self.assertEqual, self.assertSame
        case.serve()
        r, out = case.run_q('rep["e";"1+\\"a\\""];')
        self.assertEqual(r["e"], ("err", "type: cannot apply + (Plus) to integer and string"), out)

    # ---- items marked UNCONFIRMED in docs/ipc_capabilities.md that PeachQ can answer ----

    def test_handshake_answers_the_smaller_capability_up_to_3(self):
        """PeachQ: an offer of 1, 2, 3 is answered with the same, an offer of 4, 5 or 6 with 3 (its maximum:
        no large-message extension); an offer above 6 is answered 0 (but the connection works)."""
        for offer, answer in ((1, 1), (2, 2), (3, 3), (4, 3), (5, 3), (6, 3), (7, 0), (255, 0)):
            with self.subTest(offer):
                s = raw_connect(self.peachq.port, capability=offer)
                try:
                    self.assertEqual(s.reply, bytes([answer]))
                    self.assertEqual(self.ask(s, "1+1")[1], bytes([0xf9]) + (2).to_bytes(8, "little"))
                finally:
                    s.close()

    def test_handshake_capability_zero_forms(self):
        """PeachQ: the qPython capability-0 form (``user:pw`` then a single NUL) is answered 0 and works; the
        two-NUL form is answered 0 but the connection then answers nothing (ref section 1: unconfirmed for kdb+).
        Credentials may be empty, the user alone, or contain a colon in the password."""
        s = raw_connect(self.peachq.port, hello=b"u:p\x00")
        try:
            self.assertEqual(s.reply, b"\x00")
            self.assertEqual(self.ask(s, "1+1")[1], bytes([0xf9]) + (2).to_bytes(8, "little"))
        finally:
            s.close()
        s = raw_connect(self.peachq.port, hello=b"u:p\x00\x00")
        try:
            self.assertEqual(s.reply, b"\x00")
            send_msg(s, 1, encode("1+1"))
            s.settimeout(1.5)
            with self.assertRaises(socket.timeout):
                s.recv(1)
        finally:
            s.close()
        for hello in (b":\x03\x00", b"\x03\x00", b"user\x03\x00", b"u:p:q\x03\x00"):
            with self.subTest(hello):
                s = raw_connect(self.peachq.port, hello=hello)
                try:
                    self.assertEqual(s.reply, b"\x03")
                finally:
                    s.close()

    def test_async_error_is_silent_and_the_connection_survives(self):
        """PeachQ: an async message that fails sends nothing back (ref 6.2: unconfirmed for kdb+)."""
        s = raw_connect(self.peachq.port)
        try:
            send_msg(s, 0, encode("1+`a"))
            send_msg(s, 0, encode("nosuchname"))
            head, body = self.ask(s, "1+1")
            self.assertEqual((head[1], bytes(body)), (2, bytes([0xf9]) + (2).to_bytes(8, "little")))
        finally:
            s.close()

    def test_error_response_bytes(self):
        """PeachQ: an error is a type 2 message with ``80 text 00`` (the example of ref 6.1)."""
        s = raw_connect(self.peachq.port)
        try:
            head, body = self.ask(s, "1+`a")
            self.assertEqual((bytes(head[:4]), bytes(body)), (bytes([1, 2, 0, 0]), b"\x80type\x00"))
        finally:
            s.close()

    def test_general_list_of_same_type_atoms_is_unified_on_receipt(self):
        """(c) PeachQ's ``-9!`` turns a type 0 list whose items are all atoms of one type into the vector of that type
        (two symbols give type 11, two longs 7, two chars 10; one item too; mixed or empty stays 0).  KlongPy sends a call
        envelope as type 0 on purpose so that ``(`f;`a)`` is not a symbol vector, but PeachQ
        reads it as one -- harmless, since a symbol vector is accepted as a call, and unknown
        for kdb+."""
        def lst(*items):
            return bytes([0, 0]) + len(items).to_bytes(4, "little") + b"".join(items)
        sym = lambda t: bytes([0xf5]) + t.encode() + b"\x00"  # noqa: E731
        lng = lambda n: bytes([0xf9]) + n.to_bytes(8, "little")  # noqa: E731
        chr_ = lambda c: bytes([0xf6]) + c.encode()  # noqa: E731
        s = raw_connect(self.peachq.port)
        try:
            for label, body, want in (("two symbols", lst(sym("f"), sym("a")), 11), ("two longs", lst(lng(1), lng(2)), 7),
                                      ("two chars", lst(chr_("a"), chr_("b")), 10), ("symbol and long", lst(sym("f"), lng(1)), 0),
                                      ("empty", lst(), 0)):
                with self.subTest(label):
                    self.assertEqual(encode_call([S("f"), S("a")], 64)[:2], bytes([0, 0]))      # what KlongPy sends
                    send_msg(s, 1, encode_call(["{type -9!x}", np.frombuffer(bytes([1, 0, 0, 0]) + (8 + len(body)).to_bytes(4, "little") + body, dtype=np.uint8)], 64))
                    self.assertEqual(read_raw(s)[1], bytes([0xfb]) + want.to_bytes(2, "little"))      # type is a short
        finally:
            s.close()

    def test_big_endian_requests_are_accepted_and_answered_little_endian(self):
        s = raw_connect(self.peachq.port)
        try:
            text = b"1+1"
            send_msg(s, 1, bytes([10, 0]) + len(text).to_bytes(4, "big") + text, little=False)
            head, body = read_raw(s)
            self.assertEqual((head[0], bytes(body)), (1, bytes([0xf9]) + (2).to_bytes(8, "little")))
        finally:
            s.close()

    def test_framing_pipelined_and_split_messages(self):
        s = raw_connect(self.peachq.port)
        try:
            def frame(text, kind=1):
                b = encode(text)
                return bytes([1, kind, 0, 0]) + (8 + len(b)).to_bytes(4, "little") + b
            s.sendall(frame("1") + frame("2") + frame("3"))                      # three requests in one segment
            self.assertEqual([read_raw(s)[1][1] for _ in range(3)], [1, 2, 3])
            for i, byte in enumerate(frame("5+5")):                                # one byte at a time
                s.sendall(bytes([byte]))
                time.sleep(0.002)
            self.assertEqual(read_raw(s)[1][1], 10)
        finally:
            s.close()

    def test_unknown_object_type_is_an_error_reply(self):
        """PeachQ answers an object with an unknown type byte with the signal ``badmsg`` and then closes the connection."""
        s = raw_connect(self.peachq.port)
        try:
            send_msg(s, 1, bytes([200, 0]) + b"zz")
            head, body = read_raw(s)
            self.assertEqual(bytes(body), b"\x80badmsg\x00")
            send_msg(s, 1, encode("1+1"))
            with self.assertRaises((EOFError, ConnectionError)):
                read_raw(s)
        finally:
            s.close()

    def test_root_lambdas_have_an_empty_context(self):
        s = raw_connect(self.peachq.port)
        try:
            head, body = self.ask(s, "{x+y}")
            self.assertEqual(bytes(body), bytes([100, 0, 10, 0]) + (5).to_bytes(4, "little") + b"{x+y}")
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
