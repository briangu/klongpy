"""
Library exceptions (numpy; torch by the same rules) are turned
into Klong errors at operator dispatch.
"""
import os
import signal
import tempfile
import textwrap
import unittest

import numpy as np

from klongpy import KlongInterpreter
from klongpy.backends.base import UnsupportedDtypeError
from klongpy.errors import (
    KlongError, KlongInternalError, KlongMemoryError, KlongPythonError, KlongRangeError,
    KlongShapeError, KlongSyntaxError, KlongSystemError, KlongTypeError, adverb_name, classify,
    describe, operator_error, operator_name,
)
from klongpy.types import KGCall, KGChar, KGLambda, KGOp, KGSym, KLONG_UNDEFINED


def raised(klong, text, cls=KlongError):
    try:
        klong(text)
    except cls as e:
        return e
    raise AssertionError(f"{text!r} did not raise {cls.__name__}")


class TestOperatorErrors(unittest.TestCase):
    """The table: a KlongError subclass with __cause__ set, not a numpy type."""

    def test_table(self):
        k = KlongInterpreter()
        for text, cls, builtin, message in [
                ('1+"a"', KlongTypeError, TypeError, "cannot apply + (Plus) to integer and string"),
                ('"abc"-1', KlongTypeError, TypeError, "cannot apply - (Minus) to string and integer"),
                ('[1 2 3]+[1 2]', KlongShapeError, ValueError, "cannot apply + (Plus) to list of 3 and list of 2"),
                ('a+1', KlongTypeError, TypeError, "cannot apply + (Plus) to symbol and integer"),
                ('-"a"', KlongTypeError, TypeError, "cannot apply - (Negate) to string"),
                ('[[1 2] [3 4]]*[1 2 3]', KlongShapeError, ValueError,
                 "cannot apply * (Times) to 2x2 matrix and list of 3"),
        ]:
            with self.subTest(text=text):
                e = raised(k, text)
                self.assertIsInstance(e, cls)
                self.assertIsInstance(e, builtin)                 # code catching the builtin keeps working
                self.assertEqual(type(e).__name__, cls.__name__)
                self.assertEqual(str(e), message)
                self.assertIsNotNone(e.__cause__)
                self.assertNotIsInstance(e.__cause__, KlongError)
                self.assertFalse(type(e).__module__.startswith("numpy"))

    def test_position_is_the_operator(self):
        e = raised(KlongInterpreter(), 'a::1+"x"', KlongTypeError)
        self.assertEqual(e.span, (4, 5))
        self.assertEqual(e.render(), "\n".join([
            '<string>:1:5: type error: cannot apply + (Plus) to integer and string',
            '  a::1+"x"',
            '      ^']))

    def test_index_out_of_range_is_checked_by_the_operator(self):
        k = KlongInterpreter()
        e = raised(k, "[1 2]@5", KlongRangeError)
        self.assertEqual(str(e), "index 5 out of range for list of 2")
        self.assertIsInstance(e, IndexError)
        self.assertEqual(e.span, (5, 6))
        self.assertEqual(str(raised(k, '"abc"@[0 9]', KlongRangeError)), "index 9 out of range for string of 3")
        self.assertEqual(k("[1 2]@[;1]").tolist(), [2])           # the correction: not an error

    def test_undefined_in_arithmetic_names_undefined(self):
        e = raised(KlongInterpreter(), "[1.0 :undefined]+1", KlongTypeError)
        self.assertIsInstance(e, TypeError)
        self.assertIn(":undefined", str(e))
        self.assertEqual(str(e), "cannot apply + (Plus) to list of 2 and integer: the list holds :undefined")
        self.assertIsNotNone(e.__cause__)

    def test_division_by_zero_is_still_undefined(self):
        k = KlongInterpreter()
        self.assertEqual(describe(k("1%0")), ":undefined")

    def test_adverb_chains(self):
        k = KlongInterpreter()
        e = raised(k, '+/[1 "a"]', KlongTypeError)
        self.assertEqual(str(e), "cannot apply +/ (Plus-Over) to list of 2: the list holds a string")
        self.assertEqual(e.span, (0, 2))
        self.assertIsInstance(e.__cause__, TypeError)
        e = raised(k, '{x+"a"}\'[1 2]', KlongTypeError)
        # the verb is a function: the error is the one of the operator inside it
        self.assertEqual((str(e), e.span), ("cannot apply + (Plus) to integer and string", (2, 3)))

    def test_apply_of_an_operator_value(self):
        k = KlongInterpreter()
        with self.assertRaises(KlongTypeError) as cm:
            k.apply(KGOp("+", 2), [1, "a"])
        self.assertEqual(str(cm.exception), "cannot apply + (Plus) to integer and string")

    def test_unknown_operator(self):
        e = raised(KlongInterpreter(), "1:?2", KlongSyntaxError)
        self.assertEqual(str(e), "unknown operator :?")
        self.assertIsInstance(e, KeyError)

    def test_klong_errors_pass_untouched(self):
        k = KlongInterpreter()
        e = raised(k, "[1 2 3]:=0,7", KlongRangeError)
        self.assertIn("out of range", str(e))
        self.assertNotIn("cannot apply", str(e))
        self.assertIsNone(e.__cause__)

    def test_compiled_path_raises_the_same_error(self):
        k = KlongInterpreter()
        k("a::[1 2 3];b::[1 2];c::[1 1 1]")
        for _ in range(2):                                          # second time from the caches
            e = raised(k, "a+b", KlongShapeError)
            self.assertEqual((str(e), e.span), ("cannot apply + (Plus) to list of 3 and list of 2", (1, 2)))
            self.assertIsNotNone(e.__cause__)
        # the operator that failed is the inner one
        e = raised(k, "c+a*b", KlongShapeError)
        self.assertEqual((str(e), e.span), ("cannot apply * (Times) to list of 3 and list of 2", (3, 4)))
        k("f::{x+b}")
        e = raised(k, "f(a)", KlongShapeError)
        self.assertEqual(e.source.text, "f::{x+b}")


class TestClassify(unittest.TestCase):
    def test_table(self):
        class TorchLike(RuntimeError):
            pass

        class Odd(Exception):
            pass
        for exc, cls in [
                (IndexError("index 5 is out of bounds"), KlongRangeError),
                (TypeError("unsupported operand"), KlongTypeError),
                (UnsupportedDtypeError("no strings"), KlongTypeError),
                (ValueError("operands could not be broadcast together with shapes (3,) (2,)"), KlongShapeError),
                (ValueError("setting an array element with a sequence. The requested array has an inhomogeneous shape"),
                 KlongShapeError),
                (ValueError("atomic dyad shape mismatch: (3,) and (2,)"), KlongShapeError),
                # torch raises a plain RuntimeError for sizes that do not match
                (RuntimeError("The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0"),
                 KlongShapeError),
                (RuntimeError("shape '[2, 2]' is invalid for input of size 3"), KlongShapeError),
                (TorchLike("The size of tensor a (3) must match"), KlongShapeError),
                (ValueError("math domain error"), KlongRangeError),
                (ZeroDivisionError("division by zero"), KlongRangeError),
                (OverflowError("too big"), KlongRangeError),
                (MemoryError(), KlongMemoryError),
                (RecursionError("maximum recursion depth exceeded"), KlongRangeError),
                (RuntimeError("something else"), KlongInternalError),
                (AttributeError("x"), KlongInternalError),
                (Odd("x"), KlongInternalError),
        ]:
            with self.subTest(exc=repr(exc)[:50]):
                self.assertIs(classify(exc), cls)

    def test_numpy_ufunc_type_error(self):
        try:
            np.add(np.array(["a"]), 1)
        except Exception as exc:
            self.assertIs(classify(exc), KlongTypeError)
        else:
            self.fail("numpy accepted it")

    def test_messages(self):
        e = operator_error(RecursionError("maximum recursion depth exceeded"), "+", "Plus", (1, 2))
        self.assertEqual(str(e), "expression nested too deeply")
        self.assertIsInstance(e, RecursionError)
        e = operator_error(MemoryError(), "+", "Plus", (1, 2))
        self.assertIsInstance(e, KlongMemoryError)
        self.assertIsInstance(e, MemoryError)
        e = operator_error(AttributeError("no x"), "+", "Plus", (1, 2))
        self.assertIs(type(e).__mro__[1], KlongInternalError)
        self.assertEqual(str(e), "cannot apply + (Plus) to integer and integer: AttributeError: no x")

    def test_recursion_is_a_range_error(self):
        k = KlongInterpreter()
        k("f::{.f(x+1)}")
        e = raised(k, "f(0)", KlongRangeError)
        self.assertEqual(str(e), "expression nested too deeply")
        self.assertIsInstance(e, RecursionError)
        self.assertIsInstance(e.__cause__, RecursionError)
        self.assertEqual(k("1+1"), 2)                               # the interpreter is usable afterwards


class TestDescribe(unittest.TestCase):
    def test_values(self):
        k = KlongInterpreter()
        for value, text in [
                (1, "integer"), (np.int64(1), "integer"), (True, "integer"),
                (1.5, "float"), (np.float64(1.5), "float"), (np.array(2.5), "float"), (np.array(2), "integer"),
                ("abc", "string"), ("", "string"), (KGChar("a"), "char"), (KGSym("a"), "symbol"),
                ({1: 2}, "dict"),
                (k("{x}"), "function"), (KGLambda(lambda x: x), "function"), (KGOp("+", 2), "function"),
                (len, "function"),
                (np.array([1, 2, 3]), "list of 3"), ([1, 2], "list of 2"), (np.array([]), "empty list"),
                (np.array([1, "a", None], dtype=object), "list of 3"),
                (np.zeros((2, 3)), "2x3 matrix"), (np.zeros((2, 3, 4)), "2x3x4 array"),
                (k("[[1 2] [3]]"), "list of 2"),
                (KLONG_UNDEFINED, ":undefined"), (float("inf"), ":undefined"), (np.inf, ":undefined"),
                (None, "nothing"),
        ]:
            with self.subTest(value=repr(value)[:40]):
                self.assertEqual(describe(value), text)

    def test_klong_values(self):
        k = KlongInterpreter()
        self.assertEqual(describe(k("1%0")), ":undefined")
        self.assertEqual(describe(k(":{[1 2]}")), "dict")
        self.assertEqual(describe(k("0cx")), "char")
        self.assertEqual(describe(k(":foo")), "symbol")
        self.assertEqual(describe(k("[[1 2 3] [4 5 6]]")), "2x3 matrix")

    def test_torch_like_tensor(self):
        # torch is not imported anywhere outside its backend: tensors are described by duck typing
        class Dtype:
            def __init__(self, name):
                self.name = name

            def __str__(self):
                return self.name

        class Tensor:
            def __init__(self, shape, dtype):
                self.shape, self.dtype, self.ndim = shape, Dtype(dtype), len(shape)
        self.assertEqual(describe(Tensor((3,), "torch.float32")), "list of 3")
        self.assertEqual(describe(Tensor((2, 3), "torch.int64")), "2x3 matrix")
        self.assertEqual(describe(Tensor((), "torch.int64")), "integer")


class TestOperatorNames(unittest.TestCase):
    def test_names_come_from_the_docstring_headers(self):
        from klongpy.operator_help import operator_docs
        dyads, monads = operator_docs()["dyads"], operator_docs()["monads"]
        self.assertEqual(operator_name("+", 2), "Plus")
        self.assertIn("a+b", dyads["eval_dyad_add"])
        self.assertIn("[Plus]", dyads["eval_dyad_add"])
        self.assertEqual(operator_name("-", 1), "Negate")
        self.assertIn("[Negate]", monads["eval_monad_negate"])
        self.assertEqual(operator_name("-", 2), "Minus")
        self.assertEqual(operator_name(":#", 2), "Split")
        self.assertEqual(operator_name(":#", 1), "Char")
        self.assertEqual(operator_name("@", 2), "At/Index")
        self.assertIsNone(operator_name("nope", 2))

    def test_adverbs(self):
        self.assertEqual(adverb_name("/", 1), "Over")
        self.assertEqual(adverb_name("/", 2), "Over-Neutral")
        self.assertEqual(adverb_name("'", 1), "Each")
        self.assertEqual(adverb_name(":~", 1), "Converge")
        self.assertEqual(adverb_name(":~", 2), "While")
        self.assertEqual(adverb_name(":*", 2), "Iterate")


class TestPythonAndSystemFunctions(unittest.TestCase):
    def test_user_python_through_py(self):
        k = KlongInterpreter()
        k('.py("math")')
        e = raised(k, "sqrt(-1)", KlongPythonError)
        self.assertIsInstance(e.__cause__, ValueError)
        self.assertEqual(str(e), f"ValueError: {e.__cause__}")
        self.assertIsInstance(e, ValueError)                        # still catchable as what it was
        self.assertEqual(e.span, (0, 8))

    def test_user_python_through_pyf_under_an_adverb(self):
        k = KlongInterpreter()
        k('.pyf("math";"sqrt")')
        e = raised(k, "sqrt'[4 -1]", KlongPythonError)
        self.assertIsInstance(e.__cause__, ValueError)
        self.assertEqual(str(e), f"ValueError: {e.__cause__}")

    def test_user_python_through_pyc(self):
        k = KlongInterpreter()
        k('.pyf("math";"sqrt")')
        e = raised(k, '.pyc("sqrt";[-1];:{})', KlongPythonError)
        self.assertIsInstance(e.__cause__, ValueError)

    def test_user_python_file_module(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "kgerr_mod.py")
            with open(path, "w") as f:
                f.write(textwrap.dedent("""
                    class Custom(Exception):
                        pass

                    def fail(x):
                        raise Custom("no " + str(x))

                    def many(a, b, c, d):
                        raise KeyError("k")
                """))
            k = KlongInterpreter()
            k(f'.py("{path}")')
            e = raised(k, "fail(1)", KlongPythonError)
            self.assertEqual(str(e), "Custom: no 1")
            self.assertEqual(type(e.__cause__).__name__, "Custom")
            self.assertIsInstance(e, type(e.__cause__))
            # more than three parameters: imported as a function of (list, dict)
            e = raised(k, "many([1 2 3 4];:{})", KlongPythonError)
            self.assertIsInstance(e.__cause__, KeyError)

    def test_imported_lambdas_are_flagged(self):
        k = KlongInterpreter()
        k('.pyf("math";"sqrt")')
        self.assertTrue(k["sqrt"].user)

    def test_system_function_is_not_a_python_error(self):
        k = KlongInterpreter()
        with self.assertRaises(Exception) as cm:
            k(".fc(1)")                                             # a bug-like failure inside a system function
        self.assertNotIsInstance(cm.exception, KlongPythonError)
        e = raised(k, '.l("no/such/file/anywhere")', KlongSystemError)
        self.assertNotIsInstance(e, KlongPythonError)
        self.assertIsInstance(e.__cause__, FileNotFoundError)

    def test_system_function_error_is_not_relabelled_by_an_operator(self):
        # the failure of a system function called by the verb of an adverb is not "cannot apply f'"
        k = KlongInterpreter()
        e = raised(k, '{.l("no/such/file/anywhere")}\'[1 2]', KlongSystemError)
        self.assertIn("file does not exist", str(e))

    def test_host_callable_is_passed_on_as_it_is(self):
        # a Python callable put in by the host program is not an import: its exception is the host's own
        k = KlongInterpreter()

        def bad(x):
            raise LookupError("mine")
        k["bad"] = bad
        with self.assertRaises(LookupError) as cm:
            k("bad'[1 2]")
        self.assertNotIsInstance(cm.exception, KlongError)

    def test_klong_error_from_python_called_back_into_klong(self):
        k = KlongInterpreter()
        k["again"] = lambda x: k('1+"a"')
        e = raised(k, "again(1)", KlongTypeError)
        self.assertEqual(str(e), "cannot apply + (Plus) to integer and string")


if __name__ == "__main__":
    unittest.main()
