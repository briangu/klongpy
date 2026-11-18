"""
Fixes from writing programs against the docs: JSON, .fmt, .timer, --version, the escaped-quote hint,
SQL results, optional-module messages, and (torch runs only) .bkf scalars, .gradcheck, optimizers.
"""
import importlib.util
import io
import os
import subprocess
import sys
import threading
import time
import unittest
import warnings
from contextlib import redirect_stderr

import numpy as np

import klongpy
from klongpy import KlongInterpreter
from klongpy.core import KlongException
from klongpy.repl import cleanup_repl, create_repl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DB = importlib.util.find_spec("duckdb") is not None and importlib.util.find_spec("pandas") is not None
IS_TORCH_RUN = os.environ.get("_KLONGPY_TEST_BACKEND", "numpy") == "torch"
DEVICE = os.environ.get("_KLONGPY_TEST_DEVICE") or "cpu"


def run_py(code, *args, hide=()):
    """Run Python code in a subprocess (modules in hide cannot be imported); (status, stdout, stderr)."""
    pre = "import sys\n" + "".join(f"sys.modules[{m!r}] = None\n" for m in hide)
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-W", "ignore", "-c", pre + code, *args], capture_output=True, text=True,
                       timeout=60, env=env, cwd=ROOT)
    return p.returncode, p.stdout, p.stderr


class TestJson(unittest.TestCase):
    def setUp(self):
        self.k = KlongInterpreter()

    def test_value_to_text(self):
        k = self.k
        self.assertEqual(k('.json(:{;"a";1;"b";[1 2 3];"c";"hi";"d";:undefined;"e";[1.5 2.5];"f";[[1 2] [3 4]];"g";[]})'),
                         '{"a":1,"b":[1,2,3],"c":"hi","d":null,"e":[1.5,2.5],"f":[[1,2],[3,4]],"g":[]}')
        self.assertEqual(k('.json(1+!3)'), '[1,2,3]')
        self.assertEqual(k('.json(3.5)'), '3.5')
        self.assertEqual(k('.json("a""b")'), '"a\\"b"')
        self.assertEqual(k('.json(0cx)'), '"x"')
        self.assertEqual(k('.json(+/1.5+!3)'), '7.5')          # a NumPy or torch 0-d number
        self.assertEqual(k('.json([;1;:undefined])'), '[1,null]')
        self.assertEqual(k('.json(0%0)'), 'null')

    def test_text_to_value(self):
        k = self.k
        d = k('.unjson("{""a"":1,""b"":[1,2,3],""c"":null,""d"":[[1,2],[3,4]],""e"":""x"",""f"":true,""g"":{},""h"":[]}")')
        self.assertEqual(d["a"], 1)
        self.assertEqual([int(v) for v in d["b"]], [1, 2, 3])
        self.assertEqual(str(d["c"]), ":undefined")
        self.assertEqual(d["e"], "x")
        self.assertEqual(d["f"], 1)
        self.assertEqual(d["g"], {})
        self.assertEqual(len(d["d"]), 2)

    def test_round_trip(self):
        k = self.k
        text = '{"name":"ada","scores":[1,2,3],"mean":2.5,"note":null}'
        k('t::"%s"' % text.replace('"', '""'))
        self.assertEqual(k('.json(.unjson(t))'), text)

    def test_unsupported_values_are_type_errors(self):
        for src, word in [('.json(:{;"s";:sym})', "symbol"), ('.json({x})', "function"),
                          ('.json(:{;"f";{x}})', 'key "f"'), ('.json(:{;1;2})', "key"), ('.unjson(3)', "string")]:
            with self.subTest(src=src):
                with self.assertRaises(KlongException) as cm:
                    self.k(src)
                self.assertEqual(cm.exception.kind, "type")
                self.assertIn(word, str(cm.exception))

    def test_bad_text_is_a_range_error_with_the_position(self):
        for src in ['.unjson("[1,")', '.unjson("NaN")', '.unjson("")', '.unjson("[1] 2")']:
            with self.subTest(src=src):
                with self.assertRaises(KlongException) as cm:
                    self.k(src)
                self.assertEqual(cm.exception.kind, "range")
        with self.assertRaises(KlongException) as cm:
            self.k('.unjson("[1,")')
        self.assertIn("line 1 column 4 (character 3)", str(cm.exception))


class TestFmtZeroPadding(unittest.TestCase):
    def test_leading_zero_is_a_clear_error(self):
        k = KlongInterpreter()
        for spec in ("02", "-02", "05.2"):
            with self.subTest(spec=spec), self.assertRaises(KlongException) as cm:
                k('.fmt("{:%s}";5)' % spec)
            self.assertIn("zero padding is not supported", str(cm.exception))

    def test_other_formats_unchanged(self):
        k = KlongInterpreter()
        self.assertEqual(k('.fmt("{:8}|";42)'), "42      |")
        self.assertEqual(k('.fmt("{:-8}|";42)'), "      42|")
        self.assertEqual(k('.fmt("{:0}|";42)'), "42|")


class TestTimer(unittest.TestCase):
    def setUp(self):
        self.klong, self.loops = create_repl()

    def tearDown(self):
        cleanup_repl(self.loops)

    def test_callback_with_arguments_says_what_is_wrong(self):
        for src in ('.timer("t";1;{x+1})', '.timer("t";1;{x;1})'):
            with self.subTest(src=src), self.assertRaises(KlongException) as cm:
                self.klong(src)
            self.assertIn("takes no arguments", str(cm.exception))
            self.assertIn("cb::{...}", cm.exception.hint)
        self.klong('cb1::{x}')
        with self.assertRaises(KlongException) as cm:
            self.klong('.timer("t";1;cb1)')
        self.assertIn("takes no arguments, not one of 1", str(cm.exception))

    def test_nilad_literal_is_evaluated_at_the_call_site(self):
        self.klong('hits::0')
        with self.assertRaises(KlongException) as cm:
            self.klong('.timer("t";1;{hits::hits+1;1})')
        self.assertEqual(str(cm.exception), "z must be a function")
        self.assertEqual(self.klong['hits'], 1)           # ran before .timer was called
        self.assertIn("run before .timer sees it", cm.exception.hint)

    def test_named_nilad_is_accepted(self):
        self.klong('cb::{0}')
        th = self.klong('.timer("t";5;cb)')
        self.assertEqual(th.cancel(), 1)

    def test_timer_made_on_another_thread_fires_and_cancels(self):
        # the loop is idle in select(): only call_soon_threadsafe wakes it
        self.klong('n::0')
        self.klong('cb::{n::n+1;1}')
        out = {}
        t = threading.Thread(target=lambda: out.update(h=self.klong('.timer("t";0.05;cb)')))
        t.start()
        t.join()
        end = time.time() + 5
        while time.time() < end and self.klong['n'] < 2:
            time.sleep(0.02)
        self.assertGreaterEqual(self.klong['n'], 2)
        self.assertEqual(out["h"].cancel(), 1)
        self.assertEqual(out["h"].cancel(), 0)
        time.sleep(0.2)
        seen = self.klong['n']
        time.sleep(0.2)
        self.assertEqual(self.klong['n'], seen)


class TestVersionFlag(unittest.TestCase):
    def test_version(self):
        code, out, err = run_py("from klongpy.cli import main; main()", "--version")
        # run_py passes args after -c; main reads sys.argv
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), f"kgpy {klongpy.__version__}")

    def test_version_via_module(self):
        p = subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", "--version"], capture_output=True,
                           text=True, timeout=60, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT))
        self.assertEqual((p.returncode, p.stdout.strip()), (0, f"kgpy {klongpy.__version__}"))


class TestEscapedQuoteHint(unittest.TestCase):
    HINT = 'a quote inside a string is written twice: ""'

    def test_hint_on_syntax_errors(self):
        for strict in (0, 2):
            for src in ('f::{"a\\"b"}', 'f::{.p("a\\"b")}'):
                with self.subTest(strict=strict, src=src):
                    k = KlongInterpreter(strict=strict)
                    with self.assertRaises(KlongException) as cm:
                        k(src)
                    self.assertIn(self.HINT, cm.exception.hint or "")

    def test_hint_on_unterminated_string_warning_and_error(self):
        k = KlongInterpreter(strict=2)
        with self.assertRaises(KlongException) as cm:
            k('a::"x \\" y";b::{1}')
        self.assertIn(self.HINT, cm.exception.hint)

    def test_no_hint_for_other_errors(self):
        k = KlongInterpreter()
        with self.assertRaises(KlongException) as cm:
            k('f::{1+')
        self.assertIsNone(cm.exception.hint)


@unittest.skipUnless(HAS_DB, "requires duckdb and pandas")
class TestSqlResults(unittest.TestCase):
    def setUp(self):
        self.k = KlongInterpreter(backend="numpy")      # tables hold NumPy columns
        self.k('.py("klongpy.db")')
        self.k('T::.table(:{;"id";[1 2 3];"name";["a" "b" "c"];"x";[1.5 2.5 3.5]})')
        self.k('db::.db(:{;"T";T})')

    def test_single_string_cell(self):
        self.assertEqual(self.k('db("select name from T where id=2")'), "b")
        self.assertEqual(self.k("db(\"select 'a'\")"), "a")

    def test_displaying_a_single_string_result(self):
        from klongpy.core import kg_write
        v = self.k('db("select name from T where id=2")')
        self.assertEqual(kg_write(v, self.k._backend, display=False), '"b"')

    def test_null_is_undefined(self):
        self.assertEqual(str(self.k('db("select null")')), ":undefined")

    def test_shapes_are_unchanged(self):
        self.assertEqual(int(self.k('db("select id from T where id=2")')), 2)
        self.assertEqual(list(self.k('db("select id from T")')), [1, 2, 3])
        self.assertEqual(list(self.k('db("select name, x from T where id=2")')), ["b", 2.5])
        self.assertEqual(np.asarray(self.k('db("select id, x from T")')).shape, (3, 2))
        self.assertEqual(len(self.k('db("select * from T where id>9")')), 0)

    def test_undefined_in_a_column_makes_it_text(self):
        self.k('U::.table(:{;"f";[1.5 2.5 :undefined]})')
        self.k('e::.db(:{;"U";U})')
        self.assertEqual(self.k('e("select typeof(f) from U limit 1")'), "VARCHAR")
        self.k('V::.table(:{;"f";[1.5 2.5 -1]})')
        self.k('g::.db(:{;"V";V})')
        self.assertEqual(float(self.k('g("select avg(nullif(f,-1)) from V")')), 2.0)

    def test_error_rendering_has_no_doubled_word(self):
        with self.assertRaises(KlongException) as cm:
            self.k('db("select nope from")')
        text = cm.exception.render()
        self.assertIn("error: Parser Error", text)
        self.assertNotIn("error error", text)

    def test_kvs_key_error_rendering(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.k('kv::.kvs("%s")' % d)
            with self.assertRaises(KlongException) as cm:
                self.k('kv@5')
        self.assertNotIn("error error", cm.exception.render())
        self.assertIn("key must be a str", cm.exception.render())


class TestMissingExtras(unittest.TestCase):
    def run_klong(self, hide, src):
        return run_py("import sys\nfrom klongpy import KlongInterpreter\nk=KlongInterpreter()\n"
                      "from klongpy.errors import KlongError\n"
                      "try:\n    k(%r)\nexcept KlongError as e:\n    print(e.message)\n" % src, hide=hide)

    def test_db_names_its_extra(self):
        for hide in (("duckdb",), ("pandas",)):
            code, out, err = self.run_klong(hide, '.py("klongpy.db")')
            self.assertIn('pip install "klongpy[db]"', out, (out, err))

    def test_web_names_its_extra(self):
        code, out, err = self.run_klong(("aiohttp",), '.py("klongpy.web")')
        self.assertIn('pip install "klongpy[web]"', out, (out, err))

    def test_ws_names_its_extra(self):
        code, out, err = self.run_klong(("websockets",), '.py("klongpy.ws")')
        self.assertIn('pip install "klongpy[ws]"', out, (out, err))

    def test_torch_backend_names_its_extra(self):
        code, out, err = run_py("from klongpy import KlongInterpreter\ntry:\n    KlongInterpreter(backend='torch')\n"
                                "except ValueError as e:\n    print(e)\n", hide=("torch",))
        self.assertIn('pip install "klongpy[torch]"', out, (out, err))

    def test_cli_backend_torch_names_its_extra(self):
        code, out, err = run_py("from klongpy.cli import main; main()", "--backend", "torch", "-e", "1", hide=("torch",))
        self.assertNotEqual(code, 0)
        self.assertIn('pip install "klongpy[torch]"', err)

    def test_other_import_failures_are_unchanged(self):
        code, out, err = self.run_klong(("nonesuch",), '.py("nonesuch")')
        self.assertIn("failed to load module: nonesuch", out)


@unittest.skipUnless(HAS_TORCH and IS_TORCH_RUN, "runs with --backend torch")
class TestTorchFixes(unittest.TestCase):
    def setUp(self):
        self.k = KlongInterpreter(backend="torch", device=DEVICE)

    def test_every_documented_bkf_function_takes_scalars_and_arrays(self):
        names = "exp sin cos tan tanh sqrt abs log log10 floor ceil round".split()
        self.k('.bkf([%s])' % " ".join('"%s"' % n for n in names))
        ref = KlongInterpreter(backend="numpy")
        ref('.bkf([%s])' % " ".join('"%s"' % n for n in names))
        for n in names:
            for arg in ("0.5", "2", "-1.5", "[0.5 1.5 2.5]", "[1 2 3]"):
                with self.subTest(fn=n, arg=arg):
                    got = np.asarray(self.k._backend.to_numpy(self.k(f"{n}({arg})")), dtype=float)
                    want = np.asarray(ref(f"{n}({arg})"), dtype=float)
                    if n in ("log", "log10", "sqrt") and arg == "-1.5":
                        self.assertTrue(np.isnan(got).all())
                        continue
                    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_gradcheck_with_a_constant_matrix(self):
        if DEVICE != "cpu":
            self.skipTest(".gradcheck needs the CPU")
        self.k('.bkf(["tanh" "matmul"])')
        self.k('net(v)::{+/tanh(matmul([[1.0 2.0] [3.0 4.0]];v))}')
        self.assertEqual(self.k('.gradcheck(net;[0.1 0.2])'), 1)

    def test_mixed_float_widths_in_a_torch_function(self):
        if DEVICE == "mps":
            self.skipTest("MPS has no float64")
        self.k('.bkf("matmul")')
        self.k('W::[[1.0 2.0] [3.0 4.0]]')
        import torch
        self.k['v'] = torch.tensor([1.0, 2.0], dtype=torch.float64, device=self.k._backend.device)
        got = self.k._backend.to_numpy(self.k('matmul(W;v)'))
        np.testing.assert_allclose(got, [5.0, 11.0])

    def test_optimizers_keep_working_on_the_device(self):
        prog = ('.pyf("klongpy.optimizers";"SGDOptimizer")\n'
                'a::(!4)%4.0\nX::[;a;a]\nW::[[0.5 0.1] [0.2 0.3]]\n.bkf("matmul")\n'
                'loss::{+/+/matmul(W;X)}\nopt::SGDOptimizer(["W"];:{;"lr";0.01})\n'
                'opt(loss)\nopt(loss)\nr::loss()')
        self.k(prog)
        got = float(self.k._backend.to_numpy(self.k['r']))
        self.assertAlmostEqual(got, 1.47, places=3)
        self.assertEqual(self.k['W'].device.type, self.k._backend.device.type)

    def test_adam_on_the_device(self):
        self.k('.pyf("klongpy.optimizers";"AdamOptimizer")\nw::1.0;b::0.0\nX::[1 2 3];Y::[3 5 7]\n'
               'loss::{(+/((w*X)+b-Y)^2)%3}\nopt::AdamOptimizer(["w" "b"];:{;"lr";0.1})\n{opt(loss)}\'!300\nr::loss()')
        self.assertLess(float(self.k._backend.to_numpy(self.k['r'])), 0.5)

    def test_gradient_of_a_square_at_zero_is_zero(self):
        g = self.k('{+/x^2}:>[0.0 1.0]')
        np.testing.assert_allclose(self.k._backend.to_numpy(g), [0.0, 2.0])

    def test_optimizers_module_is_importable_without_torch_at_import(self):
        code, out, err = run_py("import sys\nimport klongpy.optimizers\nprint(sys.modules.get('torch') is not None)", hide=("torch",))
        self.assertEqual((code, out.strip()), (0, "False"), err)


class TestOptimizersModule(unittest.TestCase):
    def test_package_has_the_optimizers_and_the_example_reexports_them(self):
        import klongpy.optimizers as o
        self.assertEqual(sorted(o.klongpy_exports), ["AdamOptimizer", "SGDOptimizer"])
        spec = importlib.util.spec_from_file_location("ex_optimizers", os.path.join(ROOT, "examples", "autograd", "optimizers.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.assertIs(m.SGDOptimizer, o.SGDOptimizer)
        self.assertIs(m.AdamOptimizer, o.AdamOptimizer)

    def test_sgd_on_numpy(self):
        k = KlongInterpreter()
        k('.pyf("klongpy.optimizers";"SGDOptimizer")\nw::10.0\nloss::{w^2}\nopt::SGDOptimizer(["w"];:{;"lr";0.1})\n{opt(loss)}\'!50')
        self.assertLess(abs(float(k['w'])), 1e-3)


if __name__ == "__main__":
    unittest.main()
