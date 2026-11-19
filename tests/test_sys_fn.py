import os
import tempfile
import unittest


from klongpy import KlongInterpreter, KlongException
from klongpy.sys_fn import *


class TestSysFn(unittest.TestCase):

    def run_in(self, klong, **names):
        for k, v in names.items():
            klong[k] = v
        return klong

    def test_channel_integration(self):
        data = '"hello"'
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            klong["fname"] = fname
            klong('o::.oc(fname);.tc(o);.d("""hello""");.cc(o)')
            with open(fname) as f:
                self.assertEqual(f.read(), data)
            klong('i::.ic(fname);.fc(i)')
            self.assertEqual(klong(".rl()"), data)
            klong(".fc(0);.cc(i)")

    def test_append_channel(self):
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write("123")
            klong["fname"] = fname
            klong('a::.ac(fname);.tc(a);.d("456");.cc(a)')
            with open(fname) as f:
                self.assertEqual(f.read(), "123456")

    def test_output_channel_truncates(self):
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write("123")
            klong["fname"] = fname
            klong('o::.oc(fname);.tc(o);.d("hi");.cc(o)')
            with open(fname) as f:
                self.assertEqual(f.read(), "hi")

    def test_delete_file(self):
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            klong["fname"] = fname
            klong('o::.oc(fname);.tc(o);.d("hello");.cc(o)')
            self.assertTrue(os.path.exists(fname))
            klong(".df(fname)")
            self.assertFalse(os.path.exists(fname))
            with self.assertRaises(KlongException):
                klong(".df(fname)")

    def test_evaluate(self):
        self.assertEqual(KlongInterpreter()('.E("A::1+1;A+2")'), 4)

    def test_from_channel_and_to_channel_give_the_previous_channel(self):
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write("1+1")
            klong["fname"] = fname
            klong("i::.ic(fname)")
            self.assertEqual(str(klong(".fc(i)")), str(klong[".cin"]))
            self.assertEqual(str(klong(".fc(0)")), str(klong["i"]))
            self.assertEqual(str(klong(".tc(.cout)")), str(klong[".cout"]))

    def test_input_channel_of_a_missing_file_is_an_error(self):
        with self.assertRaises(KlongException):
            KlongInterpreter()('.ic("doesntexist")')

    def test_load(self):
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write("fn::{1+1};fn()")
            klong = KlongInterpreter()
            klong["fname"] = fname
            self.assertEqual(klong(".l(fname)"), 2)
            self.assertEqual(klong("fn()"), 2)

    def test_more_input_and_read_line(self):
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write("line 1\nline 2\nline 3\n")
            klong = KlongInterpreter()
            klong["fname"] = fname
            klong(".fc(.ic(fname))")
            got = []
            while klong(".mi()"):
                got.append(klong(".rl()"))
            self.assertEqual(got, ["line 1", "line 2", "line 3", ""])       # .mi stays true until a read meets the end

    def test_read(self):
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write('[1 2 3 4];"hello"')
            klong = KlongInterpreter()
            klong["fname"] = fname
            klong(".fc(.ic(fname))")
            self.assertTrue(klong._backend.kg_equal(klong(".r()"), [1, 2, 3, 4]))
            self.assertEqual(klong(".rl()"), ';"hello"')        # .r read one object and left the rest

    def test_read_string(self):
        klong = KlongInterpreter()
        self.assertTrue(klong._backend.kg_equal(klong('.rs("[1 2 3 4]")'), [1, 2, 3, 4]))

    def test_print_and_write_go_to_the_channel(self):
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            klong["fname"] = fname
            klong('o::.oc(fname);.tc(o);.p("hello");.w("hello");.cc(o)')
            with open(fname) as f:
                self.assertEqual(f.read(), 'hello\n"hello"')

    def test_random_number(self):
        klong = KlongInterpreter()
        r, r2 = klong(".rn()"), klong(".rn()")
        i = 0
        while r == r2 and i < 3:
            i += 1
            r2 = klong(".rn()")
        self.assertNotEqual(r, r2)

    def test_autoclose_channel(self):
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            f = open(fname, "w")
            c = KGChannel(f, channel_dir=KGChannelDir.OUTPUT)
            self.assertFalse(f.closed)
            del c
            self.assertTrue(f.closed)


    def test_sys_display_result(self):
        klong = KlongInterpreter()
        r = klong(".d(1)")
        self.assertEqual(r,"1")


    def test_sys_print_result(self):
        klong = KlongInterpreter()
        r = klong(".p(1)")
        self.assertEqual(r,"1")

    def test_sys_python_load_module(self):
        klong = KlongInterpreter()
        r = klong(f'.py("math")')
        self.assertEqual(r,1)
        r = klong('sqrt(64)')
        self.assertEqual(r,8)
        r = klong('fsum(1+!10)')
        self.assertEqual(r,55)

    def test_sys_python_load_sub_module(self):
        klong = KlongInterpreter()
        with self.assertRaises(KlongException):
            r = klong('sqrt(64)')
        r = klong(f'.pyf("math";"sqrt")')
        self.assertEqual(r,1)
        r = klong('sqrt(64)')
        self.assertEqual(r,8)
        with self.assertRaises(KlongException):
            r = klong('fsum(1+!10)')
        r = klong(f'.pyf("math";"fsum")')
        self.assertEqual(r,1)
        r = klong('fsum(1+!10)')
        self.assertEqual(r,55)

    def test_sys_python_load_multiple_sub_modules(self):
        klong = KlongInterpreter()
        with self.assertRaises(KlongException):
            r = klong('sqrt(64)')
        with self.assertRaises(KlongException):
            r = klong('fsum(1+!10)')
        r = klong(f'.pyf("math";["sqrt" "fsum"])')
        self.assertEqual(r,1)
        r = klong('sqrt(64)')
        self.assertEqual(r,8)
        r = klong('fsum(1+!10)')
        self.assertEqual(r,55)

    def test_sys_python_load_custom_module(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.join(tests_dir, "plugins")
        tests = []
        tests.append(os.path.join("tests", os.path.join("plugins", "greetings")))
        tests.append(os.path.join(plugins_dir, "greetings"))
        tests.append(os.path.join(plugins_dir, "greetings/__init__.py"))
        tests.append(os.path.join(plugins_dir, "greetings/hello_world.py"))
        tests.append("greetings")
        try:
            sys.path.append(plugins_dir)
            for fpath in tests:
                klong = KlongInterpreter()
                self.assertEqual(klong(f'.py("{fpath}")'),1)

                self.assertEqual(klong('nilad()'),"hello, world!")
                self.assertEqual(klong('monad(1)'),"1+1")
                self.assertEqual(klong('dyad(1;2)'),"1*2+1")
                self.assertEqual(klong('triad(1;2;3)'),"1*2+3+1")
                self.assertEqual(klong('knilad()'),4)
                self.assertEqual(klong('kmonad(1)'),2)
                self.assertEqual(klong('kdyad(1;2)'),1*2+1)
                self.assertEqual(klong('ktriad(1;2;3)'),1*2+3+1)

                if fpath.endswith("hello_world.py"):
                    r = klong['not_exported']
                    self.assertTrue(r is not None)
                else:
                    with self.assertRaises(KeyError):
                        klong['not_exported']
        finally:
            sys.path.pop()

    def test_sys_python_load_custom_export(self):
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.join(tests_dir, "plugins")
        fpath = os.path.join(plugins_dir, "custom_export")
        klong = KlongInterpreter()
        self.assertEqual(klong(f'.py("{fpath}")'),1)
        self.assertEqual(klong('.hello()'),"hello, world!")


    def test_sys_write_result(self):
        klong = KlongInterpreter()
        r = klong(".w(1)")
        self.assertEqual(r,1)

    def test_sys_write_dict(self):
        klong = KlongInterpreter()
        r = klong(".w(:{[1 2]})")
        # Convert to comparable values (handles both numpy and torch backends)
        keys = list(r.keys())
        values = list(r.values())
        key = keys[0].item() if hasattr(keys[0], 'item') else keys[0]
        value = values[0].item() if hasattr(values[0], 'item') else values[0]
        self.assertEqual(key, 1)
        self.assertEqual(value, 2)


    def test_simple_io(self):
        t = """
        foo::{.tc(T::.oc(x));.p("hello!");.cc(T)}
        bar::{.fc(T::.ic(x));R::.rl();.cc(T);R}
        """
        klong = KlongInterpreter()
        klong(t)
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            klong['fname'] = fname
            klong('foo(fname)')
            with open(fname, 'r') as f:
                r = f.read()
                self.assertEqual(r, 'hello!\n')
            r = klong('bar(fname)')
            self.assertEqual(r, 'hello!')

    def test_simple_cat(self):
        t = """
        cat::{.mi{.p(x);.rl()}:~.rl()}
        type::{.fc(.ic(x));cat()}
        copy::{[of];.tc(of::.oc(y));type(x);.cc(of)}
        """
        klong = KlongInterpreter()
        klong(t)
        with tempfile.TemporaryDirectory() as td:
            fname_src = os.path.join(td, "source.txt")
            fname_dest = os.path.join(td, "dest.txt")
            klong['src'] = fname_src
            klong['dest'] = fname_dest
            data = "this is a test"
            with open(fname_src, 'w') as f:
                f.write(data)
            klong('copy(src;dest)')
            with open(fname_dest, 'r') as f:
                r = f.read()
                self.assertEqual(r, "this is a test\n")
