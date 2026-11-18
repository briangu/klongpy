"""Every kind of function crosses the kdb+ wire as text and arrives with the same parameters and arity."""
import unittest

from ipc_helpers import free_port  # noqa: F401  (keeps the helper path on sys.path)
from test_ipc_endpoint import IPCCase

from klongpy import KlongInterpreter
from klongpy.ipc import decode, encode
from klongpy.ipc.endpoint import function_source_text, lambda_value

# name of the function on the sender, definition, arity, a call (argument text) and its result
CASES = [
    ("c0", "c0::{42}", 0, "()", 42),
    ("c1", "c1::{x+1}", 1, "(10)", 11),
    ("c2", "c2::{x*y}", 2, "(3;4)", 12),
    ("c3", "c3::{x+y+z}", 3, "(1;2;3)", 6),
    ("n0", "n0()::{7}", 0, "()", 7),
    ("n1", "n1(a)::{a+1}", 1, "(1)", 2),
    ("n2", "n2(a;b)::{a-b}", 2, "(5;2)", 3),
    ("n3", "n3(a;b;c)::{a+b+c}", 3, "(1;2;3)", 6),
    ("n4", "n4(a;b;c;d)::{(a*b)+c*d}", 4, "(1;2;3;4)", 14),
    ("n5", "n5(a;b;c;d;e)::{a+b+c+d+e}", 5, "(1;2;3;4;5)", 15),
    ("n6", "n6(a;b;c;d;e;f)::{a+b+c+d+e+f}", 6, "(1;2;3;4;5;6)", 21),
    ("hd", "hd::{[a b]::x;a-b}", 1, "([9 4])", 5),
    ("hn", "hn(p)::{[a b]::p;a-b}", 1, "([9 4])", 5),
    ("lc", "lc::{[t];t::x*2;t+1}", 1, "(5)", 11),
    ("ln", "ln(a)::{[t];t::a*2;t+1}", 1, "(5)", 11),
    ("rc", "rc::{:[x<1;0;x+.f(x-1)]}", 1, "(4)", 10),
    ("rn", "rn(a)::{:[a<1;0;a+.f(a-1)]}", 1, "(4)", 10),
]


class TestFunctionText(unittest.TestCase):
    def test_text_and_rebuild(self):
        k = KlongInterpreter()
        for name, src, arity, _, _ in CASES:
            k(src)
            fn = k(name)
            text = function_source_text(fn)
            self.assertIsNotNone(text, name)
            self.assertEqual(text.startswith("fn("), "(" in src.split("::")[0], name)
            got = lambda_value(KlongInterpreter(), decode(encode(fn)))
            self.assertEqual(got.arity, arity, name)
            self.assertEqual(getattr(got.a, "params", None), getattr(fn.a, "params", None), name)

    def test_projection_stays_opaque(self):
        k = KlongInterpreter()
        k("g::{x+y};p::g(1;)")
        self.assertIsNone(function_source_text(k("p")))
        self.assertEqual(decode(encode(k("p"))).text, ":monad")


class TestApplyToAScalarResult(unittest.TestCase):
    def test_function_applied_to_the_scalar_another_application_gave(self):
        # on the torch backend that scalar is a 0-d tensor, which @ must take as one argument
        k = KlongInterpreter()
        k("twice(g;v)::{g@g@v}")
        self.assertEqual(int(k("twice({x+7};1)")), 15)


class TestFunctionsOverIPC(IPCCase):
    def test_argument_to_a_call(self):
        server, client = self.pair()
        self.kg(server, "ap::{x@y}")      # x applied to y
        self.kg(server, "twice(g;v)::{g@g@v}")
        self.kg(client, "k::100;inc(n)::{n+k}")
        self.kg(server, "k::7")
        self.assertEqual(self.kg(client, "c(:twice,inc,1)"), 15)
        self.assertEqual(self.kg(client, "c(:twice,{x+k},1)"), 15)

    def test_every_function_runs_on_the_receiver(self):
        server, client = self.pair()
        for name, src, arity, args, want in CASES:
            self.kg(client, src)
            self.kg(server, "run::{[f];f::x;f%s}" % args)
            self.assertEqual(self.kg(client, "c(:run,,%s)" % name), want, name)

    def test_receiver_keeps_names_and_arity(self):
        server, client = self.pair()
        self.kg(client, "d::.clid(c)")
        self.kg(client, "n3(a;b;c)::{a+b+c}")
        self.kg(client, "d,:got,,n3")
        got = self.kg(server, "got")
        self.assertEqual(got.arity, 3)
        self.assertEqual([str(p) for p in got.a.params], ["a", "b", "c"])
        self.assertEqual(self.kg(server, "got(1;2;3)"), 6)

    def test_returned_value(self):
        server, client = self.pair()
        for name, src, arity, args, want in CASES:
            if arity == 0:
                continue                                        # naming a nilad runs it: there is no value to return
            self.kg(server, src)
            f = self.kg(client, 'c("%s")' % name)                # the function came back as a value
            self.assertEqual(f.arity, arity, name)
            self.kg(client, 'w::c("%s")' % name)
            self.assertEqual(self.kg(client, "w%s" % args), want, name)
        self.kg(server, "gen(n)::{[g];g(a;b)::{a-b};g}")
        self.assertEqual(self.kg(client, "c(:gen,,0)").arity, 2)

    def test_projection_is_opaque(self):
        server, client = self.pair()
        self.kg(client, "d::.clid(c);g::{x+y};p::g(1;);d,:pj,,p")
        self.assertEqual(str(self.kg(server, "pj")), ":monad")


if __name__ == "__main__":
    unittest.main()
