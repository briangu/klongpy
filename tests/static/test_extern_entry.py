"""
.extern and .entry under KlongPy :
the binding API (klong.externs), the checks at declaration and at call time,
and what `kgpy check` says about the same mistakes.

Binding API, as specified here:

    klong.externs[name] = callable      a Python stand-in, looked up at each call
    klong.externs.declared[name]        ExternDecl(name, signature, module), set by .extern
    klong.externs.entries               [EntryDecl(name, signature)], appended by .entry
    load_program(path, externs={...})   run a file with stand-ins bound
    call_entry(klong, name, *args)      call an entry with arguments and result checked
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.errors import (KlongArityError, KlongError, KlongNameError, KlongPythonError, KlongShapeError,
                            KlongSystemError, KlongTypeError)
from klongpy.static import call_entry, check_source, load_program
from klongpy.sys_fn_static import ExternTable
from klongpy.types import KLONG_UNDEFINED

from tests.backend_compat import on_rust_engine

from .hostutil import to_np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADC = '.extern("read.adc";"int -> float")\n'


def interp(**externs):
    klong = KlongInterpreter(strict=2)
    klong.externs.update({k.replace("_", "."): v for k, v in externs.items()})
    return klong


class TestExtern(unittest.TestCase):
    def test_the_stand_in_is_called(self):
        seen = []
        klong = interp(read_adc=lambda ch: seen.append(ch) or 0.25)
        self.assertEqual(klong(ADC + "read.adc(3)"), 0.25)
        self.assertEqual(seen, [3])
        self.assertIs(type(seen[0]), int)

    def test_externs_is_a_table_on_every_interpreter(self):
        a, b = KlongInterpreter(), KlongInterpreter()
        self.assertIsInstance(a.externs, ExternTable)
        self.assertIsInstance(a.externs, dict)
        self.assertIsNot(a.externs, b.externs)
        self.assertEqual((a.externs, a.externs.declared, a.externs.entries), ({}, {}, []))

    def test_the_declaration_is_recorded(self):
        klong = interp()
        self.assertEqual(klong(ADC), "int -> float")        # the canonical signature
        self.assertEqual(klong('.extern("fill";"float[4]->float[4]";"dsp")'), "float[4] -> float[4]")
        d = klong.externs.declared
        self.assertEqual(sorted(d), ["fill", "read.adc"])
        self.assertEqual((d["read.adc"].name, str(d["read.adc"].signature), d["read.adc"].module), ("read.adc", "int -> float", "env"))
        self.assertEqual(d["fill"].module, "dsp")
        self.assertEqual(d["read.adc"].c_name, "read_adc")

    def test_the_stand_in_is_looked_up_at_each_call(self):
        klong = interp()
        klong(ADC + "f::{read.adc(x)*2}")                  # declared and used before any stand-in exists
        klong.externs["read.adc"] = lambda ch: 1.0
        self.assertEqual(klong("f(0)"), 2.0)
        klong.externs["read.adc"] = lambda ch: 4.0         # and can be replaced
        self.assertEqual(klong("f(0)"), 8.0)

    def test_missing_stand_in_is_a_system_error_naming_the_extern(self):
        klong = interp()
        with self.assertRaises(KlongSystemError) as cm:
            klong(ADC + "x::1\nread.adc(0)")
        e = cm.exception
        self.assertIn("read.adc", e.message)
        self.assertIn('klong.externs["read.adc"]', e.hint)
        self.assertEqual(e.position()[1:], (3, 1))
        self.assertEqual(e.source.text[e.span[0]:e.span[1]], "read.adc(0)")

    def test_wrong_argument_type(self):
        klong = interp(read_adc=lambda ch: 0.0)
        with self.assertRaises(KlongTypeError) as cm:
            klong(ADC + "read.adc(0.5)")
        self.assertEqual(cm.exception.message, 'argument 1 of read.adc must be int, got a float (declared "int -> float")')
        self.assertEqual(cm.exception.position()[1:], (2, 1))
        for bad in ('"s"', "[1 2]", ":sym"):
            with self.subTest(arg=bad), self.assertRaises(KlongTypeError):
                klong(f"read.adc({bad})")

    def test_an_int_is_passed_as_a_float_where_a_float_is_declared(self):
        seen = []
        klong = interp(set_duty=lambda v: seen.append(v) or 0)
        klong('.extern("set.duty";"float -> int")\nset.duty(1)')
        self.assertEqual(seen, [1.0])
        self.assertIs(type(seen[0]), float)

    def test_wrong_array_shape(self):
        klong = interp(fill=lambda a: a * 2.0)
        klong('.extern("fill";"float[3] -> float[3]")')
        r = to_np(klong, klong("fill([1 2 3])"))
        self.assertEqual((r.dtype.kind, r.tolist()), ("f", [2.0, 4.0, 6.0]))
        with self.assertRaises(KlongShapeError) as cm:
            klong("fill([1 2])")
        self.assertIn("argument 1 of fill must be float[3], got int[2]", cm.exception.message)
        with self.assertRaises(KlongTypeError):
            klong("fill(1.0)")

    def test_the_stand_in_gets_numpy_arrays_of_the_declared_type(self):
        seen = []
        klong = interp(dot=lambda a, b: seen.append((a, b)) or float(a @ b))
        klong('.extern("dot";"float[3] int[3] -> float")')
        self.assertEqual(klong("dot([1 2 3];[4 5 6])"), 32.0)
        a, b = seen[0]
        self.assertEqual((type(a), a.dtype, type(b), b.dtype), (np.ndarray, np.float64, np.ndarray, np.int64))

    def test_the_result_is_checked_too(self):
        klong = interp(read_adc=lambda ch: "high")
        with self.assertRaises(KlongTypeError) as cm:
            klong(ADC + "read.adc(0)")
        self.assertIn("the result of the stand-in of read.adc must be float, got a string", cm.exception.message)
        klong = interp(cnt=lambda: 1.5, vec=lambda: [1.0, 2.0])
        klong('.extern("cnt";"-> int")\n.extern("vec";"-> float[3]")')
        with self.assertRaises(KlongTypeError):
            klong("cnt()")
        with self.assertRaises(KlongShapeError):
            klong("vec()")

    def test_zero_parameter_extern(self):
        klong = interp(now=lambda: 42)
        self.assertEqual(klong('.extern("now";"-> int")\nnow()'), 42)
        with self.assertRaises(KlongArityError) as cm:
            klong("now(1)")
        self.assertEqual(cm.exception.message, 'now is declared "-> int" and takes 0 arguments, got 1')

    def test_wrong_number_of_arguments(self):
        klong = interp(read_adc=lambda ch: 0.0)
        klong(ADC)
        for call in ("read.adc()", "read.adc(1;2)"):
            with self.subTest(call=call), self.assertRaises(KlongArityError) as cm:
                klong(call)
            self.assertIn("takes 1 argument", cm.exception.message)

    def test_more_than_three_parameters(self):
        klong = interp(xfer=lambda bus, cs, addr, val: bus * 1000 + cs * 100 + addr * 10 + val)
        klong('.extern("xfer";"int int int int -> int")')
        self.assertEqual(klong("xfer(1;2;3;4)"), 1234)
        self.assertEqual(klong("f::{xfer(x;x;x;x)};f(2)"), 2222)
        with self.assertRaises(KlongArityError):
            klong("xfer(1;2;3;4;5)")
        with self.assertRaises(KlongTypeError) as cm:
            klong("xfer(1;2;3;4.5)")
        self.assertIn("argument 4 of xfer", cm.exception.message)

    def test_void_and_str(self):
        seen = []
        klong = interp(log_msg=lambda text, n: seen.append((text, n)))
        r = klong('.extern("log.msg";"str int -> void")\nlog.msg("tick";3)')
        self.assertIs(r, KLONG_UNDEFINED)                   # a void extern has no value
        self.assertEqual(seen, [("tick", 3)])
        self.assertIs(type(seen[0][0]), str)
        with self.assertRaises(KlongTypeError):
            klong("log.msg(:tick;3)")                       # a symbol is not a string

    def test_an_exception_in_the_stand_in_is_a_python_error(self):
        def broken(ch):
            raise RuntimeError("sensor offline")
        klong = interp(read_adc=broken)
        with self.assertRaises(KlongPythonError) as cm:
            klong(ADC + "read.adc(0)")
        self.assertIn("sensor offline", cm.exception.message)
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)

    def test_an_extern_is_a_function_like_any_other_under_klongpy(self):
        klong = interp(read_adc=lambda ch: ch * 0.5)
        klong(ADC)
        self.assertEqual(klong("read.adc'[0 1 2]").tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(klong("+/read.adc'!4"), 3.0)

    def test_declaring_twice(self):
        klong = interp()
        klong(ADC)
        self.assertEqual(klong('.extern("read.adc";"int->float")'), "int -> float")    # the same declaration again
        with self.assertRaises(KlongTypeError) as cm:
            klong('.extern("read.adc";"int -> int")')
        self.assertIn('already declared "int -> float"', cm.exception.message)
        with self.assertRaises(KlongTypeError):
            klong('.extern("read.adc";"int -> float";"other")')

    def test_bad_declarations(self):
        cases = [
            ('.extern("read.adc")', KlongArityError, "2 or 3 arguments"),
            ('.extern(:a;"int -> int")', KlongTypeError, "name must be a string"),
            ('.extern("a";1)', KlongTypeError, "signature must be a string"),
            ('.extern("a";"int -> int";3)', KlongTypeError, "module must be a string"),
            ('.extern("a";"int => int")', KlongTypeError, "'->'"),
            ('.extern("a";"int -> flot")', KlongTypeError, "unknown type 'flot'"),
            ('.extern("a";"int -> str")', KlongTypeError, "str cannot be a result"),
            ('.extern("kg.read";"int -> int")', KlongTypeError, "kg_"),
            ('.extern("while";"int -> int")', KlongTypeError, "reserved in C"),
            ('.extern("a_b";"int -> int")', KlongTypeError, "Klong name"),
            ('.extern("x";"int -> int")', KlongTypeError, "function argument"),
            ('taken::1\n.extern("taken";"int -> int")', KlongNameError, "already defined"),
        ]
        for text, cls, word in cases:
            with self.subTest(text=text):
                with self.assertRaises(cls) as cm:
                    interp()(text)
                self.assertIn(word, cm.exception.message)
                self.assertIsNotNone(cm.exception.span, "positioned at the declaration")

    def test_a_signature_error_says_where_in_the_signature(self):
        with self.assertRaises(KlongTypeError) as cm:
            interp()('.extern("a";"int -> flot")')
        self.assertIn('(at column 8 of "int -> flot")', cm.exception.message)

    def test_declared_inside_a_function_and_at_strict_levels(self):
        for level in (0, 1, 2):
            with self.subTest(strict=level):
                klong = KlongInterpreter(strict=level)
                klong.externs["read.adc"] = lambda ch: 0.5
                klong('setup::{.extern("read.adc";"int -> float")};setup()')
                self.assertEqual(klong("read.adc(0)"), 0.5)

    def test_export_is_left_alone(self):
        klong = KlongInterpreter()
        self.assertIsNot(klong(".export"), klong(".extern"))
        if not on_rust_engine():        # the Python engine's function object
            self.assertIn("export", klong(".export").a.fn.__name__)


class TestEntry(unittest.TestCase):
    def test_xyz_function(self):
        klong = interp()
        self.assertEqual(klong('f::{x+y}\n.entry("f";"int  int->int")'), "int int -> int")
        (e,) = klong.externs.entries
        self.assertEqual((e.name, str(e.signature)), ("f", "int int -> int"))
        self.assertEqual(klong("f(1;2)"), 3)                # otherwise a no-op

    def test_arity_is_the_highest_argument_used(self):
        klong = interp()
        klong('g::{z}\n.entry("g";"int int int -> int")')  # {z} takes three
        with self.assertRaises(KlongArityError) as cm:
            klong('.entry("g";"int -> int")')
        self.assertEqual(cm.exception.message, '.entry: g takes 3 arguments but the signature "int -> int" has 1 parameter')
        klong('h::{-x}\n.entry("h";"float -> float")')     # an argument under a monad counts

    def test_named_parameter_function(self):
        klong = interp()
        klong('mix(a;b;c;d)::{(a+b)*c-d}\n.entry("mix";"int int float int -> float")')
        with self.assertRaises(KlongArityError):
            klong('.entry("mix";"int int -> float")')
        klong('k()::{42}\n.entry("k";"-> int")')
        with self.assertRaises(KlongArityError):
            klong('.entry("k";"int -> int")')

    def test_the_function_must_be_defined_first(self):
        with self.assertRaises(KlongNameError) as cm:
            interp()('.entry("f";"int -> int")\nf::{x}')
        self.assertIn("f is not defined", cm.exception.message)
        self.assertEqual(cm.exception.position()[1:], (1, 1))

    def test_only_a_function_of_the_program_can_be_an_entry(self):
        for text in ('v::3\n.entry("v";"-> int")', 'p::{x+y}(1;)\n.entry("p";"int -> int")',
                     ADC + '.entry("read.adc";"int -> float")'):
            with self.subTest(text=text), self.assertRaises(KlongTypeError) as cm:
                interp()(text)
            self.assertIn("is not a function defined in the program", cm.exception.message)

    def test_collision_with_generated_names(self):
        for name in ("init", "t", "size"):
            with self.subTest(name=name), self.assertRaises(KlongTypeError) as cm:
                interp()(f'{name}::{{x}}\n.entry("{name}";"int -> int")')
            self.assertIn(f"prog_{name}", cm.exception.message)

    def test_bad_signatures(self):
        klong = interp()
        klong("f::{x}")
        for sig, word in (("str -> int", "externs only"), ("int", "'->'"), ("int -> str", "str cannot be a result")):
            with self.subTest(sig=sig), self.assertRaises(KlongTypeError) as cm:
                klong(f'.entry("f";"{sig}")')
            self.assertIn(word, cm.exception.message)
        with self.assertRaises(KlongTypeError):
            klong('.entry(:f;"int -> int")')

    def test_several_entries_with_different_shapes(self):
        klong = interp()
        klong('f::{x*2}\n.entry("f";"float[64] -> float[64]")\n.entry("f";"float[8] -> float[8]")\n.entry("f";"float[8]->float[8]")')
        self.assertEqual([str(e.signature) for e in klong.externs.entries], ["float[64] -> float[64]", "float[8] -> float[8]"])
        self.assertEqual(str(klong.externs.entry("f", "float[8] -> float[8]").signature), "float[8] -> float[8]")
        self.assertEqual(call_entry(klong, "f", np.ones(8)).shape, (8,))
        self.assertEqual(call_entry(klong, "f", np.ones(64)).shape, (64,))
        with self.assertRaises(KlongShapeError):
            call_entry(klong, "f", np.ones(5))


class TestHostHelpers(unittest.TestCase):
    PROG = ADC + 'acc::0.0\nstep::{acc::acc+x*read.adc(0);acc}\n.entry("step";"float -> float")\n'

    def test_load_program_and_call_entry(self):
        klong = load_program("prog.kg", text=self.PROG, externs={"read.adc": lambda ch: 0.5})
        self.assertEqual(klong.strict, 2)
        self.assertEqual(call_entry(klong, "step", 2), 1.0)             # the int argument is converted
        self.assertEqual(call_entry(klong, "step", 2.0), 2.0)           # state is kept between calls
        self.assertIs(type(call_entry(klong, "step", 0.0)), float)

    def test_call_entry_checks_arguments_and_result(self):
        klong = load_program("prog.kg", text=self.PROG, externs={"read.adc": lambda ch: 0.5})
        with self.assertRaises(KlongTypeError):
            call_entry(klong, "step", [1.0, 2.0])
        with self.assertRaises(KlongTypeError):
            call_entry(klong, "step")
        with self.assertRaises(KlongNameError):
            call_entry(klong, "nosuch", 1.0)
        klong = load_program("p.kg", text='f::{x%2}\n.entry("f";"int -> int")')    # the function returns a float
        with self.assertRaises(KlongTypeError) as cm:
            call_entry(klong, "f", 3)
        self.assertIn("the result of entry f must be int, got a float", cm.exception.message)

    def test_load_program_from_a_file_positions_errors_in_it(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.kg")
            with open(path, "w") as f:
                f.write(ADC + "v::read.adc(0)\n")
            with self.assertRaises(KlongSystemError) as cm:
                load_program(path)                          # a compile-time call of an extern without a stand-in
            self.assertEqual(cm.exception.position(), (path, 2, 4))
            self.assertEqual(load_program(path, externs={"read.adc": lambda ch: 9.0})("v"), 9.0)

    def test_into_an_existing_interpreter(self):
        klong = KlongInterpreter(strict=2)
        self.assertIs(load_program("prog.kg", text=self.PROG, externs={"read.adc": lambda ch: 1.0}, klong=klong), klong)
        self.assertEqual(klong("step(3.0)"), 3.0)


HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestHostBackends(unittest.TestCase):
    """
    A static program runs on any backend. The interior is the backend's own (the
    torch backend computes in float32); the boundary is numpy: an extern stand-in
    gets Python numbers and numpy int64/float64 arrays, call_entry returns the same,
    and signatures are checked on those, whatever the backend.
    """
    PROG = ('.extern("probe";"float[3] -> float[3]")\n'
            '.extern("tick";"int -> int")\n'
            'w::[1.5 2.5 3.5]\n'
            'f::{[a];a::probe(x+w);(a*2.0)+tick(1)}\n'
            '.entry("f";"float[3] -> float[3]")\n'
            'g::{+/x}\n'
            '.entry("g";"int[4] -> int")\n')

    def run_on(self, backend):
        seen = []

        def probe(a):
            seen.append(a)
            return a * 0.5

        def tick(n):
            seen.append(n)
            return n
        klong = load_program("p.kg", text=self.PROG, externs={"probe": probe, "tick": tick}, backend=backend)
        out = call_entry(klong, "f", [1.0, 2.0, 3.0])
        total = call_entry(klong, "g", [1, 2, 3, 4])
        return klong, seen, out, total

    def check_boundary(self, backend):
        klong, seen, out, total = self.run_on(backend)
        self.assertEqual(klong._backend.name, backend)
        arr, n = seen
        self.assertIs(type(arr), np.ndarray)
        self.assertEqual((arr.dtype, arr.shape), (np.float64, (3,)))
        self.assertEqual(arr.tolist(), [2.5, 4.5, 6.5])
        self.assertIs(type(n), int)
        self.assertIs(type(out), np.ndarray)
        self.assertEqual(out.dtype, np.float64)
        self.assertEqual(out.tolist(), [3.5, 5.5, 7.5])
        self.assertIs(type(total), int)
        self.assertEqual(total, 10)
        # the signature is checked on the numpy value: a wrong shape or type is refused on every backend
        with self.assertRaises(KlongShapeError):
            call_entry(klong, "f", [1.0, 2.0])
        with self.assertRaises(KlongTypeError):
            call_entry(klong, "g", [1.0, 2.0, 3.0, 4.0])
        return out

    def test_numpy(self):
        self.check_boundary("numpy")

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_torch(self):
        self.check_boundary("torch")

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_a_torch_result_that_is_not_the_declared_type_is_refused(self):
        klong = load_program("p.kg", text='f::{x%2}\n.entry("f";"int -> int")', backend="torch")
        with self.assertRaises(KlongTypeError):
            call_entry(klong, "f", 3)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed")
    def test_the_checker_does_not_depend_on_the_backend(self):
        """check_source gives the same diagnostics and types whichever backend evaluated the globals."""
        text = 'w::[1.5 2.5 3.5]\nk::[[1 2] [3 4]]\nf::{x+w}\n.entry("f";"float[3] -> float[3]")\n'
        base = check_source(text, "t.kg")
        self.assertEqual([str(d) for d in base.diagnostics], [])
        want = {n: str(g.ty) for n, g in base.globals.items()}
        self.assertEqual(want, {"w": "float[3]", "k": "int[2,2]"})
        for backend in ("numpy", "torch"):
            klong = load_program("t.kg", text=text, backend=backend)
            r = check_source(text, "t.kg", globals={n: klong(n) for n in want})
            self.assertEqual([str(d) for d in r.diagnostics], [])
            self.assertEqual({n: str(g.ty) for n, g in r.globals.items()}, want)
            self.assertEqual({f: {tuple(map(str, k)): str(v) for k, v in i.specializations.items()} for f, i in r.functions.items()},
                             {f: {tuple(map(str, k)): str(v) for k, v in i.specializations.items()} for f, i in base.functions.items()})


class TestSameMistakeHostAndChecker(unittest.TestCase):
    """A wrong argument type or a missing stand-in is a positioned error naming the extern on both sides."""

    PROG = ADC + 'step::{read.adc(x)}\n.entry("step";"float -> float")\n'

    def test_wrong_argument_type(self):
        r = check_source(self.PROG, "prog.kg")
        (d,) = r.diagnostics
        self.assertEqual((d.rule, d.line, d.col), ("extern-call", 2, 17))
        self.assertIn("read.adc", d.message)
        klong = load_program("prog.kg", text=self.PROG, externs={"read.adc": lambda ch: 0.0})
        with self.assertRaises(KlongTypeError) as cm:
            call_entry(klong, "step", 0.5)
        self.assertIn("read.adc", cm.exception.message)
        self.assertEqual(cm.exception.position(), ("prog.kg", 2, 8))

    def test_the_checker_and_the_host_read_a_signature_alike(self):
        bad = '.extern("a";"int -> flot")\n'
        (d,) = check_source(bad, "p.kg").diagnostics
        self.assertEqual(d.rule, "signature")
        with self.assertRaises(KlongTypeError) as cm:
            KlongInterpreter()(bad)
        self.assertIn(d.message.split(": ", 1)[1], cm.exception.message)

    def test_kgpy_run_reports_the_missing_stand_in(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.kg")
            with open(path, "w") as f:
                f.write(ADC + "v::read.adc(0)\n")
            p = subprocess.run([sys.executable, "-W", "ignore", "-m", "klongpy.cli", path], cwd=ROOT, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 1)
            self.assertIn("p.kg:2:4: system error: extern read.adc has no stand-in on the host", p.stderr)


if __name__ == "__main__":
    unittest.main()
