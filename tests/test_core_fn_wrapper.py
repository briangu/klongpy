import unittest

from klongpy import KlongInterpreter
from klongpy.core import KGCall, KGFnWrapper, KGSym


class TestKGFnWrapper(unittest.TestCase):
    def test_find_symbol_skips_reserved(self):
        klong = KlongInterpreter()
        klong("f::{x+1}")
        fn = klong._context[KGSym("f")]
        del klong["f"]
        klong._context[KGSym("x")] = fn

        wrapper = KGFnWrapper(klong, fn)

        self.assertIsNone(wrapper._sym)

    def test_find_symbol_non_kgfn(self):
        klong = KlongInterpreter()
        wrapper = KGFnWrapper(klong, lambda x: x)

        self.assertIsNone(wrapper._sym)

    def test_find_symbol_kgcall(self):
        klong = KlongInterpreter()
        call = KGCall(KGSym("f"), [], 1)

        wrapper = KGFnWrapper(klong, call)

        self.assertIsNone(wrapper._sym)

    def test_ragged_list_argument(self):
        # a message decoded from JSON, [[1,2],[3]], must reach the function
        klong = KlongInterpreter()
        klong("f::{#x}")
        klong("g(a;b)::{#b}")
        self.assertEqual(klong["f"]([[1, 2], [3]]), 2)
        self.assertEqual(klong["f"]([1, 2, 3]), 3)
        self.assertEqual(klong["g"](0, [[1, 2], [3], [4, 5, 6]]), 3)
        klong("h::{x,y}")
        klong("p::h(1;)")
        self.assertEqual(len(klong["p"]([[1], [2, 3]])), 3)

    def test_ws_message_handler_gets_a_ragged_reply(self):
        import asyncio
        try:
            import websockets  # noqa: F401
        except ImportError:
            self.skipTest("requires websockets")
        from klongpy.ws.sys_fn_ws import execute_server_command
        klong = KlongInterpreter()
        klong(".ws.m::{[h m];h::x;m::y;#m}")
        out = []

        async def run():
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            await execute_server_command(loop, fut, klong, KGSym(".ws.m"), [[1, 2], [3]], object())
            await asyncio.sleep(0)
            out.append(fut.result())

        asyncio.run(run())
        self.assertEqual(out, [2])

    def test_dynamic_resolution_arity_mismatch(self):
        klong = KlongInterpreter()
        klong("f::{x+1}")
        wrapper = klong["f"]
        klong("f::{1}")

        with self.assertRaises(RuntimeError):
            wrapper(2)

    def test_fallback_when_symbol_deleted(self):
        klong = KlongInterpreter()
        klong("f::{x+1}")
        wrapper = klong["f"]
        del klong["f"]

        self.assertEqual(wrapper(2), 3)


class TestPythonFunctionAssignedTwice(unittest.TestCase):
    """A Python function assigned over an existing name is wrapped as for a new name."""

    def test_python_assignment_twice(self):
        klong = KlongInterpreter()
        klong["g"] = lambda x: x + 1
        klong["g"] = lambda x: x + 2
        self.assertEqual(klong("g(1)"), 3)

    def test_result_of_a_system_function_assigned_twice(self):
        # .pyc, .compile and friends return Python callables
        klong = KlongInterpreter()
        klong('.pyf("operator";"neg")')
        klong("h::neg")
        klong("h::neg")
        self.assertEqual(klong("h(3)"), -3)
