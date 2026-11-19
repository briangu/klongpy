"""
Calling a value that is not a function is a type error, at every strict level.

avg::(+/x)%#x (the braces forgotten) made avg the number 1.0, and avg(list)
printed 1.0 and dropped the argument. A call now fails with a KlongTypeError
that names what was called and what it is. What is callable stays callable:
Klong functions, projections, operators, Python callables, IPC connections,
databases. A name that is not defined keeps its undefined-function error.

The suite runs this file on each backend (--backend numpy|torch).
"""
import os
import tempfile
import unittest

from klongpy import KlongInterpreter
from klongpy.errors import KlongError, KlongNameError, KlongTypeError

LEVELS = (0, 1, 2)

# (what the value is, how a name is defined to hold it, the kind in the message)
VALUES = [
    ("integer", "a::3", "an integer"),
    ("float", "a::1.0", "a float"),
    ("string", 'a::"text"', "a string"),
    ("char", "a::0cx", "a char"),
    ("list", "a::[1 2 3]", "a list of 3"),
    ("empty list", "a::[]", "an empty list"),
    ("matrix", "a::[[1 2] [3 4]]", "a 2x2 matrix"),
    ("dictionary", "a:::{[1 2]}", "a dict"),
    ("undefined", "a::1%0", ":undefined"),
]

# the forms a call takes
CALLS = [
    "a(3)", "a([3 4])", "a(1;2)", "a(1;2;3)", "a()", "1+a(2)",
    "{a(x)}(1)", "{[b];b::a;b(x)}(1)", "{x(2)}(a)", "{y(x)}(1;a)", "{[b];b::a;b()}()",
]


def run(level, *lines):
    k = KlongInterpreter(strict=level)
    r = None
    for line in lines:
        r = k(line)
    return r


def error_of(level, *lines):
    k = KlongInterpreter(strict=level)
    for line in lines[:-1]:
        k(line)
    try:
        k(lines[-1])
    except KlongError as e:
        return e
    raise AssertionError(f"{lines[-1]!r} did not raise at level {level}")


class TestEveryKindOfValue(unittest.TestCase):
    def test_every_value_every_form_every_level(self):
        for level in LEVELS:
            for name, define, _ in VALUES:
                for call in CALLS:
                    with self.subTest(level=level, value=name, call=call):
                        e = error_of(level, define, call)
                        self.assertIsInstance(e, KlongTypeError)
                        self.assertEqual(e.kind, "type")

    def test_message_names_the_callee_and_the_kind(self):
        for level in LEVELS:
            for name, define, kind in VALUES:
                with self.subTest(level=level, value=name):
                    e = error_of(level, define, "a(1)")
                    self.assertEqual(str(e), f"cannot call a: it is {kind}, not a function")

    def test_hint(self):
        e = error_of(0, "avg::1.0", "avg([3 3 3 4 5 7])")
        self.assertEqual(str(e), "cannot call avg: it is a float, not a function")
        self.assertEqual(e.hint, "a function is written {...}")

    def test_hint_for_a_list_suggests_indexing(self):
        e = error_of(0, "a::[1 2 3]", "a(1)")
        self.assertEqual(e.hint, "a function is written {...}; to index, write a@i")
        e = error_of(0, 'a::"abc"', "a(1)")
        self.assertIn("a@i", e.hint)
        e = error_of(0, "a::5", "a(1)")
        self.assertNotIn("a@i", e.hint)

    def test_symbol_value(self):
        # a symbol that names a function is still called through; one that
        # names nothing, or a value, is an error
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "f::{x+1}", "g:::f", "g(2)"), 3)
                self.assertEqual(run(level, "f::{x+1}", "g:::f", "h:::g", "h(2)"), 3)
                e = error_of(level, "g:::f", "f::7", "g(2)")
                self.assertIsInstance(e, KlongTypeError)
                self.assertEqual(str(e), "cannot call g: it is a symbol, not a function")

    def test_symbol_that_names_nothing(self):
        e = error_of(0, "a:::sym", "a(1)")
        self.assertEqual(str(e), "cannot call a: it is a symbol, not a function")

    def test_dictionary_lookup_is_at(self):
        # d(k) was never a lookup (it returned the dictionary); d@k is
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "d:::{[1 2]}", "d@1"), 2)
                self.assertIsInstance(error_of(level, "d:::{[1 2]}", "d(1)"), KlongTypeError)
                self.assertIsInstance(error_of(level, "d:::{[1 2]}", "d()"), KlongTypeError)

    def test_channel(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.txt")
            for level in LEVELS:
                with self.subTest(level=level):
                    k = KlongInterpreter(strict=level)
                    k(f'c::.oc("{path}")')
                    try:
                        for call in ("c(1)", "c()"):
                            with self.assertRaises(KlongTypeError) as cm:
                                k(call)
                            self.assertEqual(str(cm.exception), "cannot call c: it is a channel, not a function")
                    finally:
                        k("n::.cc(c)")

    def test_result_of_a_call_that_is_not_a_function(self):
        # a function that returns a number, defined without braces around the call
        for level in LEVELS:
            with self.subTest(level=level):
                e = error_of(level, "f::{x+1}", "g::f(3)", "g(1)")
                self.assertEqual(str(e), "cannot call g: it is an integer, not a function")

    def test_local_variable_and_parameters(self):
        for level in LEVELS:
            with self.subTest(level=level):
                e = error_of(level, "f::{[a];a::3;a(1)}", "f()")
                self.assertEqual(str(e), "cannot call a: it is an integer, not a function")
                e = error_of(level, "f::{x(1)}", "f(5)")
                self.assertEqual(str(e), "cannot call x: it is an integer, not a function")
                e = error_of(level, "f::{y(1)}", "f(5;6)")
                self.assertEqual(str(e), "cannot call y: it is an integer, not a function")

    def test_name_defined_later_as_a_number_replaces_the_function(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "f::{x+1}", "f(1)"), 2)
                e = error_of(level, "f::{x+1}", "f::5", "f(1)")
                self.assertIsInstance(e, KlongTypeError)


class TestAdverbs(unittest.TestCase):
    def test_non_function_under_an_adverb(self):
        for level in LEVELS:
            for src in ("a'[1 2]", "a/[1 2]", "a\\[1 2]", "[1 2]a'[3 4]", "a:~5"):
                with self.subTest(level=level, src=src):
                    e = error_of(level, "a::1", src)
                    self.assertIsInstance(e, KlongTypeError)
                    self.assertEqual(str(e), "cannot call a: it is an integer, not a function")

    def test_function_under_an_adverb(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(list(run(level, "f::{x+1}", "f'[1 2]")), [2, 3])
                self.assertEqual(run(level, "f::{x+y}", "f/[1 2 3]"), 6)

    def test_function_literal_argument_is_a_value_not_a_function(self):
        # a nilad literal passed as an argument is evaluated where it stands,
        # so x() calls its value; pass a name to pass the function
        for level in LEVELS:
            with self.subTest(level=level):
                e = error_of(level, "ap::{x()}", "ap({7})")
                self.assertEqual(str(e), "cannot call x: it is an integer, not a function")
                self.assertEqual(run(level, "ap::{x()}", "g::{7}", "ap(g)"), 7)


class TestSpanAndLevels(unittest.TestCase):
    def test_span_is_the_call(self):
        for level in LEVELS:
            with self.subTest(level=level):
                text = "avg::1.0;1+avg([3 3 3 4 5 7])"
                k = KlongInterpreter(strict=level)
                with self.assertRaises(KlongTypeError) as cm:
                    k(text)
                e = cm.exception
                self.assertEqual(e.source.text[e.span[0]:e.span[1]], "avg([3 3 3 4 5 7])")
                self.assertIn("^^^", e.render())
                self.assertIn("hint: a function is written {...}", e.render())

    def test_span_inside_a_function(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = KlongInterpreter(strict=level)
                k("avg::1.0")
                k("f::{avg(x)}")
                with self.assertRaises(KlongTypeError) as cm:
                    k("f(3)")
                e = cm.exception
                self.assertEqual(e.source.text[e.span[0]:e.span[1]], "avg(x)")

    def test_undefined_name_keeps_its_error(self):
        for level in LEVELS:
            with self.subTest(level=level):
                e = error_of(level, "nope(1)")
                self.assertIsInstance(e, KlongNameError)
                self.assertEqual(str(e), "undefined function: nope")

    def test_nilad_function_still_called(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "f::{7}", "f()"), 7)
                self.assertEqual(run(level, "f::{a}", "a::4", "f()"), 4)
                # a body that is a bare name is not a call of that name
                self.assertEqual(run(level, "f::{a}", "a::4", "f()"), 4)
                self.assertEqual(run(level, "f::{x}", "f(9)"), 9)
                self.assertEqual(run(level, "f::{{x}(1)}", "f(5)"), 1)

    def test_a_value_is_still_a_value(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "a::3", "a"), 3)
                self.assertEqual(run(level, "a::3", "a+1"), 4)
                self.assertEqual(run(level, "a::[1 2 3]", "a@1"), 2)

    def test_eval_function(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = KlongInterpreter(strict=level)
                with self.assertRaises(KlongTypeError):
                    k('.E("a::1;a(2)")')
                self.assertEqual(k('.E("a::{x+1};a(2)")'), 3)


class TestCallableKindsStayCallable(unittest.TestCase):
    def test_klong_functions(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "f::{x+1}", "f(2)"), 3)
                self.assertEqual(run(level, "f::{x+y}", "f(2;3)"), 5)
                self.assertEqual(run(level, "f::{x+y+z}", "f(1;2;3)"), 6)
                self.assertEqual(run(level, "f::{7}", "f()"), 7)
                self.assertEqual(run(level, "{x*2}(4)"), 8)

    def test_named_parameters(self):
        for level in (1, 2):
            with self.subTest(level=level):
                self.assertEqual(run(level, "f(a;b)::{a+b}", "f(2;3)"), 5)

    def test_projections(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "f::{x+y}", "p::f(1;)", "p(5)"), 6)
                self.assertEqual(run(level, "f::{x+y}", "p::f(;1)", "p(5)"), 6)
                self.assertEqual(run(level, "p::{x+y}(1;)", "p(5)"), 6)

    def test_function_passed_by_name(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "ap::{x(y)}", "g::{x+1}", "ap(g;5)"), 6)
                self.assertEqual(run(level, "ap::{x(y)}", "ap({x+1};5)"), 6)

    def test_recursion_through_dot_f(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "fact::{:[x<2;1;x*.f(x-1)]}", "fact(5)"), 120)

    def test_operator_as_verb(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "+/[1 2 3]"), 6)
                self.assertEqual(run(level, "f::{x+y}", "f/[1 2 3]"), 6)

    def test_python_callable_stored_from_python(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = KlongInterpreter(strict=level)
                k["pf"] = lambda x: x + 1
                k["pg"] = lambda x, y: x + y
                self.assertEqual(k("pf(2)"), 3)
                self.assertEqual(k("pg(2;3)"), 5)

    def test_python_imports(self):
        for level in LEVELS:
            with self.subTest(level=level):
                k = KlongInterpreter(strict=level)
                k('.py("math")')
                self.assertEqual(k("sqrt(16)"), 4.0)
                k = KlongInterpreter(strict=level)
                k('.pyf("math";"sqrt")')
                self.assertEqual(k("sqrt(16)"), 4.0)

    def test_system_function_stored_under_a_name(self):
        k = KlongInterpreter()
        k("r::.rn")
        self.assertTrue(0.0 <= float(k("r()")) < 1.0)

    def test_wrapper_from_python(self):
        k = KlongInterpreter()
        k("f::{x+1}")
        self.assertEqual(k["f"](2), 3)

    def test_function_stored_in_a_symbol(self):
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(run(level, "f::{x+1}", "g:::f", "g(2)"), 3)
                self.assertEqual(list(run(level, "f::{x+1}", "{x(1)}'[:f]")), [2])


    def test_ipc_connection(self):
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from ipc_helpers import free_port
        from klongpy.repl import create_repl, cleanup_repl
        k, loops = create_repl()
        try:
            port = free_port()
            k("sq::{x*x}")
            k(f".srv({port})")
            k(f"h::.cli({port})")
            self.assertEqual(k("h(:sq,,7)"), 49)
        finally:
            cleanup_repl(loops)

    def test_database(self):
        try:
            import pandas  # noqa: F401
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("needs pandas and duckdb")
        k = KlongInterpreter()
        k('.py("klongpy.db")')
        k('T::.table(,"a",,[1 2 3])')
        k('db::.db(:{},"T",,T)')
        r = k('db("select * from T")')            # a database is a function
        self.assertIsNotNone(r)
        with self.assertRaises(KlongTypeError):
            k('T(1)')                              # a table is not

    def test_kglambda_subclass_is_callable(self):
        from klongpy.types import KGLambda

        class Triple(KGLambda):
            def __init__(self):
                super().__init__(lambda x: 3 * x)

        k = KlongInterpreter()
        k["t3"] = Triple()
        self.assertEqual(k("t3(4)"), 12)


if __name__ == "__main__":
    unittest.main()
