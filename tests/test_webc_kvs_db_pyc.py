"""
Library and system-function fixes: .webc stops the server; a missing .kvs key is :undefined; .db from
a dictionary literal; .pyc passes positional arguments; KGLambda subclasses
that do not call the constructor have the `user` flag.
"""
import importlib.util
from pathlib import Path
import socket
import tempfile
import time
import unittest
import urllib.request

from klongpy import KlongInterpreter
from klongpy.repl import cleanup_repl, create_repl
from klongpy.core import KGLambda, KLONG_UNDEFINED


def have(*names):
    return all(importlib.util.find_spec(n) is not None for n in names)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


@unittest.skipUnless(have("aiohttp"), "aiohttp is not installed")
class TestWebc(unittest.TestCase):
    def test_webc_stops_server(self):
        port = free_port()
        klong, loops = create_repl()
        try:
            klong('.py("klongpy.web")')
            klong('get:::{}; get,"/",{x;"hi"}; post:::{}')
            klong(f'h::.web("127.0.0.1:{port}";get;post)')
            for _ in range(50):
                if port_open(port):
                    break
                time.sleep(0.1)
            self.assertTrue(port_open(port))
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                self.assertEqual(r.read().decode(), "hi")
            self.assertEqual(klong(".webc(h)"), 1)
            self.assertFalse(port_open(port))
            # a second stop has nothing to stop
            self.assertEqual(klong(".webc(h)"), 0)
        finally:
            try:
                klong(".webc(h)")
            except Exception:
                pass
            cleanup_repl(loops)


@unittest.skipUnless(have("pandas", "duckdb"), "pandas and duckdb are required")
class TestKvs(unittest.TestCase):
    def test_missing_key_is_undefined(self):
        with tempfile.TemporaryDirectory() as d:
            klong = KlongInterpreter()
            klong('.py("klongpy.db")')
            klong(f'kvs::.kvs("{d}")')
            self.assertIs(klong('kvs?"nothing"'), KLONG_UNDEFINED)
            klong('kvs,["k" "v"]')
            self.assertEqual(klong('kvs?"k"'), "v")
            self.assertIs(klong('kvs?"nothing"'), KLONG_UNDEFINED)


@unittest.skipUnless(have("pandas", "duckdb"), "pandas and duckdb are required")
class TestDb(unittest.TestCase):
    def setUp(self):
        self.klong = KlongInterpreter()
        self.klong('.py("klongpy.db")')
        self.klong('T::.table([["a" [1 2 3]] ["b" [4 5 6]]])')

    def test_db_from_dict_literal(self):
        self.klong('db1::.db(:{},"T",,T)')
        self.klong('db2::.db(:{["T" T]})')
        q = '"select sum(a) from T"'
        self.assertEqual(self.klong(f"db1({q})"), 6)
        self.assertEqual(self.klong(f"db2({q})"), 6)

    def test_database_has_user_flag(self):
        from klongpy.db.sys_fn_db import Database
        self.assertFalse(Database({}).user)


class TestSiblingLambdas(unittest.TestCase):
    def test_subclasses_have_user_flag(self):
        # a subclass that does not call KGLambda.__init__ must still have `user`
        # (the interpreter reads it when the call raises)
        classes = []
        import klongpy.sys_fn_ipc as ipc
        classes.append(ipc.KGAsyncCall)
        if have("websockets"):
            import klongpy.ws.sys_fn_ws as ws
            classes.append(ws.NetworkClient)
        for cls in classes:
            self.assertIs(getattr(cls, "user", None), False, cls.__name__)


class TestPyc(unittest.TestCase):
    def test_builtin_with_variadic_arguments(self):
        klong = KlongInterpreter()
        klong('.pyf("math";"hypot")')
        self.assertEqual(klong('.pyc("hypot";[3 4];:{})'), 5.0)
        self.assertEqual(klong('hypot(3;4)'), 5.0)

    def test_imported_variadic_argument_can_have_any_name(self):
        with tempfile.TemporaryDirectory() as directory:
            module = Path(directory) / "variadic.py"
            module.write_text("def total(first, *rest):\n    return first + sum(rest)\n", encoding="utf-8")
            klong = KlongInterpreter()
            klong(f'.pyf("{module.as_posix()}";"total")')
            self.assertEqual(klong('.pyc("total";[1 2 3 4];:{})'), 10)
            self.assertEqual(klong('total(1;2;3;4)'), 10)

    def test_class_with_positional_and_keyword_arguments(self):
        klong = KlongInterpreter()
        klong('.pyf("datetime";"datetime")')
        r = klong('.pyc(datetime;[2012 1 1];:{["hour" 12] ["minute" 30]})')
        self.assertEqual(str(r), "2012-01-01 12:30:00")

    def test_class_with_positional_arguments_only(self):
        klong = KlongInterpreter()
        klong('.pyf("datetime";"datetime")')
        self.assertEqual(str(klong('.pyc(datetime;[2012 1 2];:{})')), "2012-01-02 00:00:00")


if __name__ == "__main__":
    unittest.main()
