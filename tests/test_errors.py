"""
One KlongError hierarchy, source positions and rendering.
"""
import os
import pickle
import tempfile
import unittest
import warnings

import klongpy
import klongpy.interpreter
import klongpy.types
from klongpy.errors import KlongSyntaxWarning
from klongpy import KlongInterpreter, KlongException
from klongpy.errors import (
    ERROR_KINDS, KlongArityError, KlongError, KlongInternalError, KlongInterrupt, KlongMemoryError,
    KlongNameError, KlongPythonError, KlongRangeError, KlongRemoteError, KlongShapeError,
    KlongStaticError, KlongSyntaxError, KlongSystemError, KlongTypeError, UnexpectedChar,
    UnexpectedEOF, compat,
)
from klongpy.source import Source
from klongpy.types import KGCall, KGFn, KGLambda, KGOp, KGSym
from tests.backend_compat import requires_python_engine


def raised(klong, text, cls=KlongError):
    try:
        klong(text)
    except cls as e:
        return e
    raise AssertionError(f"{text!r} did not raise {cls.__name__}")


class TestHierarchy(unittest.TestCase):
    def test_kinds_and_wire_names(self):
        table = [
            (KlongSyntaxError, 'syntax', 'parse'), (KlongNameError, 'name', 'name'),
            (KlongArityError, 'arity', 'rank'), (KlongTypeError, 'type', 'type'),
            (KlongShapeError, 'shape', 'length'), (KlongRangeError, 'range', 'index'),
            (KlongSystemError, 'system', 'os'), (KlongPythonError, 'python', 'python'),
            (KlongInternalError, 'internal', 'internal'), (KlongMemoryError, 'memory', 'wsfull'),
            (KlongInterrupt, 'interrupt', 'stop'), (KlongStaticError, 'static', None),
            (KlongRemoteError, 'remote', None),
        ]
        for cls, kind, wire in table:
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, KlongError))
                self.assertEqual(cls.kind, kind)
                self.assertEqual(cls.wire, wire)
                self.assertIs(ERROR_KINDS[kind], cls)
        self.assertEqual(len(ERROR_KINDS), len(table))

    def test_klong_exception_is_the_same_class_everywhere(self):
        self.assertIs(KlongException, KlongError)
        self.assertIs(klongpy.KlongException, KlongError)
        self.assertIs(klongpy.types.KlongException, KlongError)
        self.assertIs(klongpy.interpreter.KlongException, KlongError)
        from klongpy.core import KlongException as from_core
        self.assertIs(from_core, KlongError)
        self.assertIs(klongpy.KlongTypeError, KlongTypeError)

    def test_range_error_is_replaced(self):
        self.assertIs(klongpy.types.RangeError, KlongRangeError)

    def test_fields_and_str(self):
        e = KlongTypeError("no good")
        self.assertEqual(str(e), "no good")
        self.assertEqual(e.message, "no good")
        self.assertEqual(e.args, ("no good",))
        self.assertIsNone(e.span)
        self.assertIsNone(e.source)
        self.assertIsNone(e.hint)
        src = Source("f.kg", "abc")
        e = KlongTypeError("no good", span=(0, 1), source=src, hint="try harder")
        self.assertEqual((e.span, e.source, e.hint), ((0, 1), src, "try harder"))
        self.assertEqual(str(e), "no good")

    def test_parser_classes_are_syntax_errors(self):
        self.assertIsInstance(UnexpectedChar("(1+2", 4, ")"), KlongSyntaxError)
        self.assertIsInstance(UnexpectedEOF("0c", 2), KlongSyntaxError)
        self.assertIsInstance(UnexpectedChar("(1+2", 4, ")"), KlongException)

    def test_compat_class_is_both(self):
        cls = compat(KlongRangeError, IndexError)
        self.assertIs(cls, compat(KlongRangeError, IndexError))
        self.assertTrue(issubclass(cls, KlongRangeError))
        self.assertTrue(issubclass(cls, IndexError))
        self.assertEqual(cls.__name__, "KlongRangeError")
        self.assertEqual(cls.kind, "range")
        self.assertEqual(str(cls("m")), "m")
        self.assertEqual(str(compat(KlongSyntaxError, KeyError)("m")), "m")     # not KeyError's repr
        self.assertIs(compat(KlongRangeError, None), KlongRangeError)

    def test_pickle_round_trip(self):
        for e in (KlongTypeError("m", span=(1, 2), source=Source("f.kg", "abc"), hint="h"),
                  UnexpectedChar("(1+2", 4, ")"),
                  compat(KlongRangeError, IndexError)("m")):
            with self.subTest(e=type(e).__name__):
                c = pickle.loads(pickle.dumps(e))
                self.assertIs(type(c), type(e))
                self.assertEqual((str(c), c.span, c.source, c.hint), (str(e), e.span, e.source, e.hint))


class TestRender(unittest.TestCase):
    """The six golden renderings of the error design."""

    def test_file_position(self):
        src = Source("examples/demo.kg", 'b::2\n\na::1+"x"\n')
        e = KlongTypeError("cannot apply + (Plus) to integer and string", span=(11, 12), source=src,
                           hint='convert with $ first, e.g. ($1),"x"')
        self.assertEqual(e.render(), "\n".join([
            'examples/demo.kg:3:6: type error: cannot apply + (Plus) to integer and string',
            '  a::1+"x"',
            '       ^',
            '  hint: convert with $ first, e.g. ($1),"x"']))

    def test_multi_line_source_and_span_clipped_to_its_line(self):
        src = Source("m.kg", "f::{x+\n  y}\ng::1")
        e = KlongSyntaxError("bad", span=(3, 11), source=src)      # the span runs on to line 2
        self.assertEqual(e.render(), "m.kg:1:4: syntax error: bad\n  f::{x+\n     ^^^")
        e = KlongNameError("undefined variable: g", span=(12, 13), source=src)
        self.assertEqual(e.render(), "m.kg:3:1: name error: undefined variable: g\n  g::1\n  ^")

    def test_tab_is_one_column(self):
        src = Source("t.kg", "\ta::\tfoo(1)")
        e = KlongNameError("undefined function: foo", span=(5, 11), source=src)
        self.assertEqual(e.render(), "t.kg:1:6: name error: undefined function: foo\n   a:: foo(1)\n       ^^^^^^")

    def test_repl_source_has_no_name(self):
        e = KlongTypeError("m", span=(5, 6), source=Source("<repl:12>", 'a::1+"x"'))
        self.assertEqual(e.render(), '1:6: type error: m\n  a::1+"x"\n       ^')

    def test_hint_without_span(self):
        e = KlongNameError("undefined variable: k", hint="declare it")
        self.assertEqual(e.render(), "name error: undefined variable: k\n  hint: declare it")

    def test_no_span(self):
        self.assertEqual(KlongSystemError("file does not exist: x").render(), "system error: file does not exist: x")
        # a span without a source cannot be shown either
        self.assertEqual(KlongSystemError("m", span=(1, 2)).render(), "system error: m")

    def test_long_line_is_clipped_around_the_caret(self):
        text = "a::" + "1+" * 200 + "foo(1)" + "+1" * 200
        at = text.index("foo")
        e = KlongNameError("undefined function: foo", span=(at, at + 6), source=Source("l.kg", text))
        head, line, carets = e.render().split("\n")
        self.assertEqual(head, f"l.kg:1:{at + 1}: name error: undefined function: foo")
        self.assertEqual(len(line), 2 + 120)
        self.assertEqual(line[carets.index("^"):carets.index("^") + 6], "foo(1)")
        self.assertEqual(carets.count("^"), 6)

    def test_span_at_end_of_input_gets_one_caret(self):
        e = raised(KlongInterpreter(), "(1+2")
        self.assertEqual(e.render(), "<string>:1:5: syntax error: expected ')' but found end of input\n  (1+2\n      ^")

    def test_color_only_on_the_first_line(self):
        e = KlongTypeError("m", span=(0, 1), source=Source("c.kg", "a"), hint="h")
        lines = e.render(color=True).split("\n")
        self.assertEqual(lines[0], "\x1b[31mc.kg:1:1: type error: m\x1b[0m")
        self.assertEqual(lines[1:], ["  a", "  ^", "  hint: h"])


class TestSyntaxErrors(unittest.TestCase):
    def test_unexpected_char_window_contains_the_offset(self):
        # the old message sliced t[i-10:i+10] with a negative start: an empty window
        text = "(1}" + "x" * 35
        e = raised(KlongInterpreter(), text, UnexpectedChar)
        self.assertEqual(str(e), "expected ')' but found '}'")
        self.assertEqual(e.span, (2, 3))
        self.assertEqual(e.source.text, text)
        line, carets = e.render().split("\n")[1:]
        self.assertEqual(line[carets.index("^")], "}")

    def test_says_what_was_expected_and_what_was_found(self):
        k = KlongInterpreter()
        for text, message, span in [
                ("(1+2", "expected ')' but found end of input", (4, 4)),
                (":[1;2;3", "expected ']' but found end of input", (7, 7)),
                (":[1;2 3]", "expected ';' but found '3'", (6, 7)),
                (":[1;2]", "expected ';' but found ']'", (5, 6)),
                ("{x+1", "expected '}' but found end of input", (4, 4)),
                ("f(1 )", "expected ')' but found whitespace", (3, 4)),
                ("{x\n", "expected '}' but found end of input", (3, 3)),
                ("0c", "expected a character after 0c but found end of input", (0, 2)),
                ("1+1.2.3", "malformed number '1.2.3'", (2, 7)),
                ("[1 -1.2.3]", "malformed number '-1.2.3'", (3, 9)),
        ]:
            with self.subTest(text=text):
                e = raised(k, text, KlongSyntaxError)
                self.assertEqual(str(e), message)
                self.assertEqual(e.span, span)
                self.assertEqual(e.source.text, text)

    def test_legacy_text_is_still_there(self):
        e = raised(KlongInterpreter(), "(1+2", UnexpectedChar)
        self.assertEqual(e.legacy_text, "t: (1+2 pos: 4 char: )")
        self.assertEqual((e.t, e.i, e.expected), ("(1+2", 4, ")"))
        e = raised(KlongInterpreter(), "1+0c", UnexpectedEOF)
        self.assertEqual(e.legacy_text, "t: 1+0c pos: 4")
        e = raised(KlongInterpreter(), "3e", KlongSyntaxError)
        self.assertEqual(e.legacy_text, "could not convert string to float: '3e'")

    def test_builtin_classes_the_parser_used_to_raise_still_catch(self):
        k = KlongInterpreter()
        with self.assertRaises(ValueError):
            k("1.2.3")
        with self.assertRaises(RuntimeError):
            k('.comment("END") 1')
        e = raised(k, '.comment("END") 1', KlongSyntaxError)
        self.assertEqual(e.span, (0, 15))

    def test_legacy_parser_raises_the_same_classes_and_messages(self):
        k = KlongInterpreter(parser="legacy")
        for text, message in [("(1+2", "expected ')' but found end of input"),
                              ("1.2.3", "malformed number '1.2.3'"),
                              ("1+", "missing right operand for +"),
                              ("-", "missing operand for -")]:
            with self.subTest(text=text):
                self.assertEqual(str(raised(k, text, KlongSyntaxError)), message)

    def test_syntax_error_in_a_named_source(self):
        e = raised(KlongInterpreter(), Source("lib/a.kg", "a::1\nb::(2"), KlongSyntaxError)
        self.assertEqual(e.render().split("\n")[0], "lib/a.kg:2:6: syntax error: expected ')' but found end of input")


class TestMissingOperands(unittest.TestCase):
    def setUp(self):
        # these inputs are malformed on purpose: the parser warns about them at strict level 0
        cw = warnings.catch_warnings()
        cw.__enter__()
        self.addCleanup(cw.__exit__, None, None, None)
        warnings.simplefilter("ignore", KlongSyntaxWarning)

    def test_dyad_without_right_operand(self):
        e = raised(KlongInterpreter(), "1+")
        self.assertIs(type(e), KlongSyntaxError)
        self.assertNotIsInstance(e, TypeError)
        self.assertEqual(str(e), "missing right operand for +")
        self.assertEqual(e.span, (1, 2))                     # the span of '+'
        self.assertIsInstance(e.__cause__, TypeError)        # what the operator said
        e = raised(KlongInterpreter(), "a::[1 2 3];a*")
        self.assertEqual((str(e), e.span), ("missing right operand for *", (12, 13)))

    def test_monad_without_operand(self):
        for text in ("-", "#"):
            with self.subTest(text=text):
                e = raised(KlongInterpreter(), text)
                self.assertIs(type(e), KlongSyntaxError)
                self.assertEqual(str(e), f"missing operand for {text}")
                self.assertEqual(e.span, (0, 1))

    def test_in_evaluated_text(self):
        e = raised(KlongInterpreter(), 'a::2;.E("a-")')
        self.assertEqual((str(e), e.span, e.source.text), ("missing right operand for -", (1, 2), "a-"))

    def test_the_parser_marks_the_node(self):
        k = KlongInterpreter()
        self.assertTrue(k.prog("1+")[1][0].missing_operand)
        self.assertTrue(k.prog("-")[1][0].missing_operand)
        self.assertFalse(k.prog("1+2")[1][0].missing_operand)
        self.assertFalse(KGFn(KGOp("+", 2), [1, 2], 2).missing_operand)

    def test_operators_that_tolerate_it_still_return_silently(self):
        # x@ and f:: return silently today; the parser half of the rule (a warning) is in the parser
        k = KlongInterpreter()
        k("x::[1 2 3]")
        self.assertEqual(list(k("x@")), [1, 2, 3])
        k("f::")


class TestRuntimePositions(unittest.TestCase):
    def test_name_errors(self):
        k = KlongInterpreter()
        e = raised(k, "1+R(1)", KlongNameError)
        self.assertEqual(str(e), "undefined function: R")
        self.assertEqual(e.span, (2, 6))
        self.assertEqual(e.source.text, "1+R(1)")

    def test_strict_assignment_keeps_its_help_as_a_hint(self):
        k = KlongInterpreter()
        k(".strict()")
        e = raised(k, "{q::1}()", KlongNameError)
        self.assertEqual(str(e), "undefined variable: q")
        self.assertIn("{[q]; ...}", e.hint)
        self.assertEqual(e.render().split("\n")[0], "<string>:1:3: name error: undefined variable: q")

    def test_innermost_node_wins_two_calls_deep(self):
        k = KlongInterpreter()
        k('f::{x+"a"}')
        k("g::{f(x)}")
        text = "1+g(2)"
        e = raised(k, text, KlongTypeError)
        # not the span of g(2) or f(x): the '+' inside f, in the source that defined f
        self.assertEqual(e.span, (5, 6))
        self.assertEqual(e.source.text, 'f::{x+"a"}')

    def test_function_defined_in_one_source_called_from_another(self):
        k = KlongInterpreter()
        k(Source("lib/a.kg", "\n\nidx::{x@y}"))
        e = raised(k, Source("main.kg", "idx([1 2];5)"), KlongRangeError)
        self.assertEqual(e.source.name, "lib/a.kg")
        self.assertEqual(e.render(), "lib/a.kg:3:8: range error: index 5 out of range for list of 2\n  idx::{x@y}\n         ^")

    def test_argument_error_uses_arg_spans(self):
        # a bare name is a leaf and has no span of its own: the node that holds it supplies it
        k = KlongInterpreter()
        k(".strict()")
        k("f::{1+undefinedname}")
        e = raised(k, "f()", KlongNameError)
        self.assertEqual(str(e), "undefined: undefinedname")     # a read (was "undefined variable:")
        self.assertEqual(e.span, (6, 19))
        self.assertEqual(e.source.text, "f::{1+undefinedname}")
        k("h::{x+y}")
        k("g::{h(1;alsoundefined)}")
        e = raised(k, "g()", KlongNameError)
        self.assertEqual(e.span, (8, 21))

    @requires_python_engine("replaces __getitem__ on the class of the Python engine's context")
    def test_bare_leaf_statement_gets_its_statement_span(self):
        k = KlongInterpreter()
        real = k._context.__class__.__getitem__

        class Stub(k._context.__class__):
            def __getitem__(self, key):
                if key == KGSym("boom"):
                    raise KlongNameError("undefined variable: boom")
                return real(self, key)
        k._context.__class__ = Stub
        text = "a::1;  boom  ;b::2"
        e = raised(k, text, KlongNameError)
        self.assertEqual(e.span, (7, 11))
        self.assertEqual(e.source.text, text)
        self.assertEqual(k("a"), 1)                          # the statements before it ran
        e = raised(k, "boom", KlongNameError)                # a program of one statement
        self.assertEqual(e.span, (0, 4))
        with self.assertRaises(KlongNameError) as cm:
            k.exec("1;boom")
        self.assertEqual(cm.exception.span, (2, 6))

    def test_system_function_error_gets_the_call_span(self):
        k = KlongInterpreter()
        e = raised(k, 'a::.fmt("{";1)', KlongRangeError)
        self.assertEqual(e.span, (3, 14))

    def test_hand_built_nodes_have_no_position(self):
        k = KlongInterpreter()
        with self.assertRaises(KlongTypeError) as cm:
            k.call(KGCall(KGOp("+", 2), [1, "a"], 2))
        self.assertIsNone(cm.exception.span)
        self.assertEqual(cm.exception.render(), "type error: cannot apply + (Plus) to integer and string")

    @requires_python_engine("the legacy parser of the Python engine")
    def test_legacy_parser_has_no_runtime_position(self):
        e = raised(KlongInterpreter(parser="legacy"), '1+"a"', KlongTypeError)
        self.assertIsNone(e.span)

    def test_error_from_the_parse_cache_has_a_position_again(self):
        k = KlongInterpreter()
        for _ in range(2):
            e = raised(k, '1+"a"', KlongTypeError)
            self.assertEqual(e.span, (1, 2))
            self.assertEqual(e.source.text, '1+"a"')


class TestSourceStack(unittest.TestCase):
    def test_call_pushes_and_pops(self):
        k = KlongInterpreter()
        seen = []
        k["probe"] = lambda: seen.append(k.current_source()) or 1
        self.assertIsNone(k.current_source())
        k("probe()")
        self.assertEqual(seen[-1].text, "probe()")
        self.assertEqual(seen[-1].name, "<string>")
        k(Source("x.kg", "probe()"))
        self.assertEqual(seen[-1].name, "x.kg")
        k.exec("probe()")
        self.assertEqual(seen[-1].text, "probe()")
        self.assertIsNone(k.current_source())
        with self.assertRaises(KlongError):
            k('probe();1+"a"')
        self.assertIsNone(k.current_source())               # popped on the way out of an error too
        self.assertEqual(k._sources, [])

    def test_evaluate_pushes_its_text(self):
        e = raised(KlongInterpreter(), 'a::1;.E("1+""x""")', KlongTypeError)
        self.assertEqual((e.source.text, e.span), ('1+"x"', (1, 2)))

    def test_load_names_the_file(self):
        k = KlongInterpreter()
        seen = []
        k["probe"] = lambda: seen.append([s.name for s in k._sources]) or 1
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "prog.kg")
            with open(path, "w") as f:
                f.write('probe()\nbad::{x+"a"}\n')
            k(f'.l("{path}")')
            self.assertEqual(seen[-1], ["<string>", path])
            e = raised(k, "bad(1)", KlongTypeError)
            self.assertEqual(e.render().split("\n")[0],
                             f"{path}:2:8: type error: cannot apply + (Plus) to integer and string")

    def test_error_in_a_loaded_file_keeps_its_position(self):
        k = KlongInterpreter()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "prog.kg")
            with open(path, "w") as f:
                f.write('a::1\nb::a+"x"\n')
            e = raised(k, f'.l("{path}")', KlongTypeError)
            self.assertEqual((e.source.name, e.source.line_col(e.span[0])), (path, (2, 5)))


class TestConvertedRaiseSites(unittest.TestCase):
    """Every converted raise is still caught by `except KlongException`, and has the class of the design."""

    def test_classes(self):
        k = KlongInterpreter()
        k("d:::{[1 2]}")
        cases = [
            ("R(1)", KlongNameError),
            ("!d", KlongTypeError),
            ('!"a"', KlongTypeError),
            ("[1 2 3]:=0,7", KlongRangeError),
            ('[1 2 3]:=0,"a"', KlongTypeError),
            ("(-1){x}:*1", KlongRangeError),
            ('"a"{x}:*1', KlongTypeError),
            ("[]:#[1 2 3]", KlongRangeError),
            ("[0]:#[1 2 3]", KlongRangeError),
            ("[-2 3]:^[1 2 3]", KlongRangeError),
            ('.fmt(1;2)', KlongTypeError),
            ('.fmt("{} {0}";[1 2])', KlongRangeError),
            ('.fmt("{:8.2}";,"a")', KlongTypeError),
            ('.pya(1;2)', KlongTypeError),
            ('.pya("x";"nosuchattribute")', KlongNameError),
            ('.timer("t";-1;{1})', KlongRangeError),
            ('.timer("t";"x";{1})', KlongTypeError),
            ('.timer("t";1;5)', KlongTypeError),
            ('.l("no/such/file/anywhere")', KlongSystemError),
            ('.df("no/such/file/anywhere")', KlongSystemError),
        ]
        for text, cls in cases:
            with self.subTest(text=text):
                with self.assertRaises(KlongException) as cm:
                    k(text)
                self.assertIsInstance(cm.exception, cls)

    def test_os_errors_of_system_functions_keep_their_builtin_class(self):
        k = KlongInterpreter()
        with self.assertRaises(FileNotFoundError) as cm:
            k('.l("no/such/file/anywhere")')
        self.assertIsInstance(cm.exception, KlongSystemError)
        self.assertIsInstance(cm.exception.__cause__, FileNotFoundError)

    def test_wrapper_arity(self):
        k = KlongInterpreter()
        k("f::{x+y}")
        with self.assertRaises(KlongArityError):
            k["f"](1)
        with self.assertRaises(RuntimeError):               # what it was before
            k["f"](1)

    def test_user_flag_defaults_to_false(self):
        self.assertFalse(KGLambda(lambda x: x).user)
        self.assertTrue(KGLambda(lambda x: x, user=True).user)


if __name__ == "__main__":
    unittest.main()
