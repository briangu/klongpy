import gc
import os
import tempfile
import unittest

from utils import array_equal

from klongpy import KlongInterpreter
from klongpy.sys_fn import *


class TestSysFn(unittest.TestCase):

    def test_autoclose_channel(self):
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            f = open(fname, "w")
            c = KGChannel(f, channel_dir=KGChannelDir.OUTPUT)
            self.assertFalse(f.closed)
            del c
            self.assertTrue(f.closed)

    def test_channel_integration(self):
        data = '"hello"'
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with eval_sys_output_channel(fname) as f:
                o = eval_sys_to_channel(klong, f)
                self.assertEqual(o, klong['.cout'])
                self.assertEqual(klong['.sys.cout'], f)
                eval_sys_display(klong, data)
            with eval_sys_input_channel(fname) as f:
                o = eval_sys_from_channel(klong, f)
                self.assertEqual(o, klong['.cin'])
                self.assertEqual(klong['.sys.cin'], f)
                r = f.raw.read()
                self.assertEqual(r, data)
                f.raw.seek(0,0)
                r = eval_sys_read_line(klong)
                self.assertEqual(r, data)
                a = eval_sys_read_string(klong, r)
                self.assertEqual(a, klong(data))
                f.raw.seek(0,0)
                r = eval_sys_read(klong)
                self.assertEqual(r, klong(data))

    def test_eval_sys_append_channel(self):
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write("123")
            with eval_sys_append_channel(fname) as f:
                f.raw.write("456")
            with open(fname, "r") as f:
                r = f.read()
                self.assertEqual(r, "123456")

    def test_eval_sys_close_channel(self):
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            f = eval_sys_append_channel(fname)
            self.assertFalse(f.raw.closed)
            eval_sys_close_channel(f)
            self.assertTrue(f.raw.closed)

    def test_eval_sys_display(self):
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with eval_sys_output_channel(fname) as f:
                eval_sys_to_channel(klong, f)
                eval_sys_display(klong, "hello")
            with eval_sys_input_channel(fname) as f:
                r = f.raw.read()
                self.assertEqual(r, "hello")

    def test_eval_sys_delete_file(self):
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with eval_sys_output_channel(fname) as f:
                eval_sys_to_channel(klong, f)
                eval_sys_display(klong, "hello")
            self.assertTrue(os.path.exists(fname))
            eval_sys_delete_file(fname)
            self.assertFalse(os.path.exists(fname))

            with self.assertRaises(RuntimeError):
                eval_sys_delete_file(fname)

    def test_eval_sys_evaluate(self):
        klong = KlongInterpreter()
        r = eval_sys_evaluate(klong, "A::1+1;A+2")
        self.assertEqual(r, 4)

    def test_eval_sys_from_channel(self):
        data = '1+1'
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write(data)
            with eval_sys_input_channel(fname) as f:
                self.assertEqual(klong['.sys.cin'], klong['.cin'])
                o = eval_sys_from_channel(klong, f)
                self.assertEqual(o, klong['.cin'])
                self.assertEqual(klong['.sys.cin'], f)
                o = eval_sys_from_channel(klong, 0)
                self.assertEqual(o, f)
                self.assertEqual(klong['.sys.cin'], klong['.cin'])

    def test_eval_sys_flush(self):
        pass

    def test_eval_sys_input_channel(self):
        data = '1+1'
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write(data)
            with eval_sys_input_channel(fname) as f:
                r = f.raw.read()
                self.assertEqual(r, data)
        with self.assertRaises(FileNotFoundError):
            with eval_sys_input_channel("doesntexist"):
                pass

    def test_eval_sys_load(self):
        data = '1+1'
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write(data)
            klong = KlongInterpreter()
            r = eval_sys_load(klong, fname)
            self.assertEqual(r, 2)
            klong = KlongInterpreter()
            r = klong(f'.l("{fname}")')
            self.assertEqual(r, 2)

    def test_eval_sys_load_fn(self):
        data = 'fn::{1+1};fn()'
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write(data)
            klong = KlongInterpreter()
            r = eval_sys_load(klong, fname)
            self.assertEqual(r, 2)
            self.assertEqual(klong('fn()'), 2)
            klong = KlongInterpreter()
            r = klong(f'.l("{fname}")')
            self.assertEqual(r, 2)
            self.assertEqual(klong('fn()'), 2)

    def test_eval_sys_more_input(self):
        data = ' ' * 100
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write(data)
            with eval_sys_input_channel(fname) as f:
                klong = KlongInterpreter()
                eval_sys_from_channel(klong, f)
                self.assertTrue(eval_sys_more_input(klong))
                eval_sys_read_line(klong)
                self.assertTrue(eval_sys_more_input(klong))
                eval_sys_read_line(klong)
                self.assertFalse(eval_sys_more_input(klong))

    def test_eval_sys_module(self):
        pass

    def test_eval_sys_output_channel(self):
        data = '"hello"'
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write("123")
            with eval_sys_output_channel(fname) as f:
                f.raw.write(data)
            with open(fname, "r") as f:
                s = f.read()
                self.assertFalse(s.startswith("123"))
                self.assertNotEqual(f.read(), data)

    def test_eval_sys_process_clock(self):
        pass

    def test_eval_sys_print(self):
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with eval_sys_output_channel(fname) as f:
                eval_sys_to_channel(klong, f)
                o = eval_sys_print(klong, "hello")
                self.assertEqual(o, "hello")
            with eval_sys_input_channel(fname) as f:
                r = f.raw.read()
                self.assertEqual(r, "hello\n")

    def test_eval_sys_random_number(self):
        r = eval_sys_random_number()
        r2 = eval_sys_random_number()
        i = 0
        while r == r2 and i < 3:
            i += 1
            r2 = eval_sys_random_number()
        self.assertNotEqual(r, r2)

    def test_eval_sys_read(self):
        # TODO: test all variants of kg_read
        data = '[1 2 3 4];"hello"'
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write(data)
            with eval_sys_input_channel(fname) as f:
                eval_sys_from_channel(klong, f)
                r = eval_sys_read(klong)
                self.assertTrue(array_equal(r, [1,2,3,4]))
                c = f.raw.read(1)
                self.assertEqual(c,';')

    def test_eval_sys_read_line(self):
        data = """line 1
        line 2
        line 3
        """
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with open(fname, "w") as f:
                f.write(data)
            with eval_sys_input_channel(fname) as f:
                eval_sys_from_channel(klong, f)
                r = eval_sys_read_line(klong)
                self.assertEqual(r, "line 1")
                r = eval_sys_read_line(klong)
                self.assertEqual(r.strip(), "line 2")
                r = eval_sys_read_line(klong)
                self.assertEqual(r.strip(), "line 3")
                r = eval_sys_read_line(klong)
                self.assertEqual(len(r), 0)

    def test_eval_sys_read_string(self):
        # TODO:
        data = '[1 2 3 4]'
        klong = KlongInterpreter()
        self.assertTrue(array_equal(eval_sys_read_string(klong, data), [1,2,3,4]))

    def test_eval_sys_system(self):
        pass

    def test_eval_sys_to_channel(self):
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with eval_sys_output_channel(fname) as f:
                self.assertEqual(klong['.sys.cout'], klong['.cout'])
                o = eval_sys_to_channel(klong, f)
                self.assertEqual(o, klong['.cout'])
                self.assertEqual(f, klong['.sys.cout'])
                o = eval_sys_to_channel(klong, 0)
                self.assertEqual(o, f)
                self.assertEqual(klong['.sys.cout'], klong['.cout'])

    def test_eval_sys_write(self):
        # TODO: test all variants of kg_write
        klong = KlongInterpreter()
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "data.txt")
            with eval_sys_output_channel(fname) as f:
                eval_sys_to_channel(klong, f)
                eval_sys_write(klong, "hello")
            with eval_sys_input_channel(fname) as f:
                r = f.raw.read()
                self.assertEqual(r, '"hello"')

    def test_eval_sys_exit(self):
        pass

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
