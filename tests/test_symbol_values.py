"""
A symbol held as data must never be looked up as a variable.

Two parts:

* Calls built at run time (adverbs, `@` apply, projections, KGFnWrapper, IPC,
  autograd helpers) bind already-evaluated values without evaluating them a
  second time. This holds at every strict level.
* A quoted symbol typed directly (`:foo`) is its own tree node, KGQSym. At
  strict levels 0 and 1 it keeps the legacy evaluation (the variable's value if
  `foo` is defined, the symbol otherwise).

In these tests `foo` (and `s`) are defined as variables, and the symbols are
held in list literals, which are data and are never evaluated.
"""
import asyncio
import pickle
import unittest
import uuid

import numpy as np

from klongpy import KlongInterpreter
from klongpy.errors import KlongError
from klongpy.core import KGCall, KGFn, KGFnWrapper, KGLambda, KGSym
from klongpy.types import KGQSym
from klongpy.sys_fn_ipc import (KGAsyncCall, KGRemoteFnCall, decode_message,
                                encode_message, execute_server_command)
from klongpy.writer import kg_write
from tests.backend_compat import to_numpy

FOO = KGSym("foo")
BAR = KGSym("bar")
BAZ = KGSym("baz")


def _plain(x):
    """Convert a Klong result to plain Python lists for comparison."""
    x = to_numpy(x)  # a tensor on the torch backend
    if isinstance(x, np.ndarray):
        return [_plain(q) for q in x]
    if isinstance(x, list):
        return [_plain(q) for q in x]
    return x


class SymbolValueCase(unittest.TestCase):

    def setUp(self):
        self.klong = KlongInterpreter()
        self.klong("foo::5")
        self.klong('s::"abc"')

    def check(self, expr, expected):
        r = _plain(self.klong(expr))
        self.assertEqual(r, expected, msg=expr)
        self._assert_symbols(r, expected, expr)

    def _assert_symbols(self, r, expected, expr):
        # 5 == KGSym is False already, but be explicit about types too.
        if isinstance(expected, list):
            for a, b in zip(r, expected):
                self._assert_symbols(a, b, expr)
        elif isinstance(expected, KGSym):
            self.assertIsInstance(r, KGSym, msg=expr)


class TestSymbolTable(SymbolValueCase):

    def test_each(self):
        self.check("{x}'[:foo :bar]", [FOO, BAR])

    def test_over_lambda(self):
        self.check("{x,y}/[:foo :bar]", [FOO, BAR])

    def test_over_operator(self):
        self.check(",/[:foo :bar]", [FOO, BAR])

    def test_apply_at(self):
        self.check("f::{x};f@*[:foo :bar]", FOO)

    def test_each_string_variable(self):
        self.check("{x}'[:s]", [KGSym("s")])

    def test_quoted_symbol_is_legacy_at_default_strict_level(self):
        # Strict level 2 (always the symbol) is tested in test_strict_level2.py.
        self.check(":foo", 5)
        self.check("{x}(:foo)", 5)
        self.check(":bar", BAR)
        self.check("{x}(:bar)", BAR)


class TestAdverbs(SymbolValueCase):

    def test_each_named_function(self):
        self.check("f::{x};f'[:foo :bar]", [FOO, BAR])

    def test_each_dict(self):
        self.check("{x@1}':{[1 :foo]}", [FOO])

    def test_each_operator(self):
        self.check(",'[:foo :bar]", [[FOO], [BAR]])

    def test_each2(self):
        self.check("[:foo :bar]{x,y}'[:bar :foo]", [[FOO, BAR], [BAR, FOO]])
        self.check("[:foo :bar],'[:bar :foo]", [[FOO, BAR], [BAR, FOO]])

    def test_each2_atoms(self):
        self.check("(*[:foo]){x,y}'*[:foo]", [FOO, FOO])

    def test_over(self):
        self.check("{x,y}/[:foo :bar :foo]", [FOO, BAR, FOO])

    def test_over_single_element(self):
        self.check("{x,y}/[:foo]", FOO)

    def test_over_with_seed(self):
        self.check("(*[:foo]){x,y}/[:bar :foo]", [FOO, BAR, FOO])
        self.check("(*[:foo]),/[:bar :foo]", [FOO, BAR, FOO])

    def test_scan(self):
        self.check("{y}\\[:foo :bar :foo]", [FOO, BAR, FOO])

    def test_scan_with_seed(self):
        self.check("(*[:foo]){y}\\[:bar :foo]", [FOO, BAR, FOO])

    def test_each_pair(self):
        self.check("{x,y}:'[:foo :bar :foo]", [[FOO, BAR], [BAR, FOO]])
        self.check(",:'[:foo :bar :foo]", [[FOO, BAR], [BAR, FOO]])

    def test_each_left(self):
        self.check("(*[:foo]){x,y}:\\[:bar :foo]", [[FOO, BAR], [FOO, FOO]])
        self.check("(*[:foo]),:\\[:bar :foo]", [[FOO, BAR], [FOO, FOO]])

    def test_each_right(self):
        self.check("(*[:foo]){x,y}:/[:bar :foo]", [[BAR, FOO], [FOO, FOO]])
        self.check("(*[:foo]),:/[:bar :foo]", [[BAR, FOO], [FOO, FOO]])

    def test_converge(self):
        self.check("{x}:~*[:foo]", FOO)

    def test_scan_converging(self):
        self.check("{x}\\~*[:foo]", [FOO])

    def test_while(self):
        # the predicate receives the symbol, not 5: 5~:foo would stop at once
        self.check("{x~*[:foo]}{[:bar :baz]@0}:~*[:foo]", BAR)

    def test_scan_while(self):
        self.check("{x~*[:foo]}{[:bar :baz]@0}\\~*[:foo]", [FOO])

    def test_iterate(self):
        self.check("3{x}:**[:foo]", FOO)

    def test_scan_iterating(self):
        self.check("2{x}\\**[:foo]", [FOO, FOO, FOO])

    def test_each_index(self):
        self.check("{x@1}@'[:foo :bar]", [FOO, BAR])

    def test_chained_adverbs(self):
        self.check("{x}''[[:foo :bar] [:bar :foo]]", [[FOO, BAR], [BAR, FOO]])
        self.check(",/'[[:foo :bar] [:bar :foo]]", [[FOO, BAR], [BAR, FOO]])


class TestApplyAndProjection(SymbolValueCase):

    def test_at_with_function_literal(self):
        self.check("{x}@*[:foo :bar]", FOO)

    def test_at_with_argument_list(self):
        self.check("{x,y}@[:foo :bar]", [FOO, BAR])

    def test_at_with_python_function(self):
        self.klong['ident'] = lambda x: x
        self.check("ident@*[:foo :bar]", FOO)

    def test_projection_through_each(self):
        self.check("f::{x,y};g::f(;1);g'[:foo :bar]", [[FOO, 1], [BAR, 1]])
        self.check("f::{x,y};g::f(1;);g'[:foo :bar]", [[1, FOO], [1, BAR]])

    def test_projection_through_at(self):
        self.check("f::{x,y};g::f(1;);g@*[:foo]", [1, FOO])

    def test_projection_of_projection(self):
        self.check("f::{x,y,z};g::f(1;;);h::g(2;);h'[:foo :bar]", [[1, 2, FOO], [1, 2, BAR]])

    def test_projection_arguments_are_still_expressions(self):
        # Legacy behaviour kept: a projection stores its arguments as
        # expressions and evaluates them when the call completes.
        self.check("a::1;f::{x,y};g::f(a;);a::2;g'[:foo]", [[2, FOO]])

    def test_function_values_stay_functions(self):
        self.check("f::{x+1};g::{x(1)};g(f)", 2)
        r = self.klong("g::{x};h::{x+1};g@h")
        self.assertIsInstance(r, KGFn)
        self.assertEqual(self.klong("g::{x};h::{x+1};k::g@h;k(1)"), 2)

    def test_symbol_naming_a_function_can_be_called(self):
        # A list literal holds symbols, not functions. Calling through an
        # argument that holds such a symbol calls the function it names.
        self.klong("p::{x+1};q::{x*2}")
        self.check("{x(3)}'[p q]", [4, 6])
        self.check("{x@3}'[p q]", [4, 6])
        self.check("{x'[1 2]}'[p q]", [[2, 3], [2, 4]])
        self.check("{y(x)}/[3 p q]", 8)
        self.check("{x}'[p q]", [KGSym("p"), KGSym("q")])

    def test_calling_argument_holding_other_symbol_is_an_error(self):
        # not a function name: a call of a symbol that names nothing is an error
        with self.assertRaises(KlongError):
            self.klong("{x(3)}'[:foo :bar]")

    def test_nilad_value_is_not_invoked_by_apply(self):
        self.klong("n::{7};id::{x}")
        r = self.klong.apply(self.klong._context[KGSym("id")], [self.klong._context[KGSym("n")]])
        self.assertIsInstance(r, KGFn)

    def test_dict_values_passed_to_function(self):
        self.check("d:::{[1 :foo] [2 :bar]};{x?1}(d)", FOO)
        self.check("d:::{[1 :foo] [2 :bar]};{d?x}'[1 2]", [FOO, BAR])
        self.check("d:::{[1 :foo] [2 :bar]};{x}'{x@1}'d", [FOO, BAR])
        self.check("d:::{[:foo 1]};{(*x)}'d", [FOO])

    def test_grad_helper_does_not_evaluate_value(self):
        from klongpy.autograd import _invoke_fn
        f = self.klong._context[KGSym("f")] if self.klong("f::{x};1") else None
        self.assertIs(_invoke_fn(self.klong, f, [FOO]), FOO)
        self.assertIs(_invoke_fn(self.klong, KGSym("f"), [FOO]), FOO)


class TestApplyApi(SymbolValueCase):

    def test_apply_function_value(self):
        f = self.klong._context[KGSym("f")] if self.klong("f::{x,y};1") else None
        self.assertEqual(_plain(self.klong.apply(f, [FOO, BAR])), [FOO, BAR])

    def test_apply_symbol_naming_function(self):
        self.klong("f::{x}")
        self.assertIs(self.klong.apply(KGSym("f"), [FOO]), FOO)

    def test_apply_operator(self):
        from klongpy.core import KGOp
        self.assertEqual(_plain(self.klong.apply(KGOp(",", 2), [FOO, BAR])), [FOO, BAR])
        self.assertEqual(_plain(self.klong.apply(KGOp(",", 1), [FOO])), [FOO])

    def test_apply_lambda(self):
        self.assertIs(self.klong.apply(KGLambda(lambda x: x), [FOO]), FOO)

    def test_apply_system_function(self):
        self.assertEqual(self.klong.apply(KGSym(".rs"), ["1+1"]), 1)

    def test_apply_nilad(self):
        self.klong("n::{7}")
        self.assertEqual(self.klong.apply(KGSym("n"), []), 7)

    def test_apply_dot_f_recursion(self):
        self.klong("fact::{:[x<2;1;x*.f(x-1)]}")
        self.assertEqual(self.klong.apply(KGSym("fact"), [5]), 120)

    def test_apply_locals(self):
        self.klong("f::{[a];a::x;a}")
        self.assertIs(self.klong.apply(KGSym("f"), [FOO]), FOO)

    def test_apply_too_few_values_returns_function(self):
        self.klong("f::{x,y}")
        self.assertIsInstance(self.klong.apply(KGSym("f"), [1]), KGFn)

    def test_apply_does_not_run_list_values_as_programs(self):
        self.klong("f::{x}")
        v = [KGSym("foo")]
        self.assertIs(self.klong.apply(KGSym("f"), [v]), v)


class TestFnWrapper(SymbolValueCase):

    def test_wrapper_from_getitem(self):
        self.klong("f::{x}")
        self.assertIs(self.klong['f'](FOO), FOO)

    def test_wrapper_two_arguments(self):
        self.klong("f::{x,y}")
        self.assertEqual(_plain(self.klong['f'](FOO, BAR)), [FOO, BAR])

    def test_wrapper_without_symbol(self):
        fn = self.klong("{x}")
        w = KGFnWrapper(self.klong, fn)
        self.assertIs(w(FOO), FOO)

    def test_wrapper_string_named_like_variable(self):
        self.klong("f::{x}")
        self.assertEqual(self.klong['f']("abc"), "abc")

    def test_wrapper_redefinition_still_resolves(self):
        self.klong("f::{x}")
        w = self.klong['f']
        self.klong("f::{x,x}")
        self.assertEqual(_plain(w(FOO)), [FOO, FOO])

    def test_wrapper_list_argument(self):
        self.klong("f::{x@0}")
        self.assertEqual(self.klong['f']([1, 2, 3]), 1)


class TestQuotedSymbolNode(SymbolValueCase):

    def _parse(self, s):
        return self.klong.prog(s)[1][0]

    def test_reader_produces_qsym(self):
        self.assertIs(type(self._parse(":foo")), KGQSym)
        self.assertIs(type(self._parse("foo")), KGSym)

    def test_qsym_equal_and_hash_equal(self):
        q = KGQSym("foo")
        self.assertEqual(q, FOO)
        self.assertEqual(FOO, q)
        self.assertEqual(hash(q), hash(FOO))
        self.assertEqual({FOO: 1}[q], 1)
        self.assertEqual({q: 1}[FOO], 1)

    def test_list_and_dict_literals_hold_plain_symbols(self):
        for q in self.klong("[:foo [:bar]]")[0], self.klong("[:foo [:bar]]")[1][0]:
            self.assertIs(type(q), KGSym)
        d = self.klong(":{[:foo :bar]}")
        for k, v in d.items():
            self.assertIs(type(k), KGSym)
            self.assertIs(type(v), KGSym)

    def test_legacy_eval_bound(self):
        self.assertEqual(self.klong(":foo"), 5)
        self.assertEqual(self.klong(":s"), "abc")

    def test_legacy_eval_unbound_gives_plain_symbol(self):
        r = self.klong(":bar")
        self.assertEqual(r, BAR)
        self.assertIs(type(r), KGSym)
        r = self.klong(":bar,:baz")
        self.assertEqual(_plain(r), [BAR, BAZ])
        for q in r:
            self.assertIs(type(q), KGSym)

    def test_legacy_quoted_reserved_name(self):
        self.assertEqual(self.klong("{:x}(3)"), 3)

    def test_legacy_quoted_name_call(self):
        self.assertEqual(self.klong("f::{x+1};:f(3)"), 4)

    def test_define_with_quoted_name(self):
        self.klong(":q::7")
        self.assertEqual(self.klong("q"), 7)

    def test_writer(self):
        self.assertEqual(kg_write(KGQSym("foo"), self.klong._backend), ":foo")
        self.assertEqual(kg_write(KGQSym("foo"), self.klong._backend, display=True), "foo")
        self.assertEqual(repr(KGQSym("foo")), ":foo")

    def test_pickle_as_plain_symbol(self):
        r = pickle.loads(pickle.dumps(KGQSym("foo")))
        self.assertIs(type(r), KGSym)
        self.assertEqual(r, FOO)
        r = pickle.loads(pickle.dumps([KGQSym("foo")]))
        self.assertIs(type(r[0]), KGSym)

    def test_module_qualified(self):
        klong = KlongInterpreter()
        klong('.module(:m);f::{x};g::{f\'[:a :b]};.module(0)')
        r = klong("g()")
        self.assertEqual(len(r), 2)
        for q in r:
            self.assertIsInstance(q, KGSym)


class TestIPC(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.klong = KlongInterpreter()
        self.klong("foo::5;update::99")

    async def _serve(self, command):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await execute_server_command(loop, fut, self.klong, command, None)
        return await fut

    async def test_remote_call_round_trip(self):
        self.klong("handler::{x,y}")
        msg_id = uuid.uuid4()
        wire = encode_message(msg_id, KGRemoteFnCall(KGSym("handler"), [KGSym("update"), KGQSym("foo")]))
        _, command = decode_message(wire[:16], wire[20:])
        for p in command.params:
            self.assertIs(type(p), KGSym)
        r = await self._serve(command)
        self.assertEqual(_plain(r), [KGSym("update"), FOO])

    async def test_remote_call_message_tags_in_list(self):
        self.klong("handler::{x}")
        wire = encode_message(uuid.uuid4(), KGRemoteFnCall(KGSym("handler"), [np.asarray([KGSym("update"), 1], dtype=object)]))
        _, command = decode_message(wire[:16], wire[20:])
        r = await self._serve(command)
        self.assertEqual(_plain(r), [KGSym("update"), 1])

    async def test_async_call_passes_symbols(self):
        self.klong("result::0;fn::{x};cb::{result::x}")
        fn = self.klong._context[KGSym("fn")]
        cb = self.klong._context[KGSym("cb")]
        loop = asyncio.get_running_loop()
        call = KGAsyncCall(loop, fn, cb, self.klong)
        await call.acall(self.klong, [FOO])
        self.assertEqual(self.klong("result"), FOO)

    async def test_async_call_unwrapped_function(self):
        # acall's last branch: a KGFn that was not wrapped
        self.klong("result::0;fn::{x};cb::{result::x}")
        loop = asyncio.get_running_loop()
        call = KGAsyncCall(loop, self.klong._context[KGSym("fn")], self.klong._context[KGSym("cb")], self.klong)
        call.fn = self.klong._context[KGSym("fn")]
        await call.acall(self.klong, [FOO])
        self.assertEqual(self.klong("result"), FOO)


if __name__ == "__main__":
    unittest.main()
