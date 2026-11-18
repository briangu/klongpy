"""The else-if chain `:[a;b:|c;d;e]` must parse to the same tree as the nested `:[a;b;:[c;d;e]]`.

The chains in tests/fixtures/cond_chain.kg are the ones the earlier broker examples used."""
import os
import unittest

from klongpy import KlongInterpreter
from klongpy.types import KGAdverb, KGFn, KGOp

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "cond_chain.kg")

PAIRS = [
    (':[msg="connected";sendAuth(x);:[msg="authenticated";handleLogin();handleAuthFailure()]]',
     ':[msg="connected";sendAuth(x):|msg="authenticated";handleLogin();handleAuthFailure()]'),
    (':[subscription?all;1;:[subscription?k;1;0]]',
     ':[subscription?all;1:|subscription?k;1;0]'),
    (':[p<lower;"buy";:[p>upper;"sell";"hold"]]',
     ':[p<lower;"buy":|p>upper;"sell";"hold"]'),
    (':[msg="connected";sendAuth(x);:[msg="authenticated";handleLogin(x);handleAuthFailure()]]',
     ':[msg="connected";sendAuth(x):|msg="authenticated";handleLogin(x);handleAuthFailure()]'),
    (':[T="success";handleControl(x;y);:[T="error";handleError(x;y);handleData(x;y)]]',
     ':[T="success";handleControl(x;y):|T="error";handleError(x;y);handleData(x;y)]'),
    (':[:_x;startdate;:[(#x)>0;0$(((x?"t")@-1):%1);startdate]]',
     ':[:_x;startdate:|(#x)>0;0$(((x?"t")@-1):%1);startdate]'),
    (':[q="connected";sendAuth(x);:[q="auth_success";handleLogin();handleAuthFailure()]]',
     ':[q="connected";sendAuth(x):|q="auth_success";handleLogin();handleAuthFailure()]'),
]


def dump(x):
    """Structural dump of a parse tree."""
    if isinstance(x, KGFn):
        return ("KGFn", dump(x.a), dump(x.args), x.arity, tuple(sorted(map(str, x.global_params))))
    if isinstance(x, (KGOp, KGAdverb)):
        return (type(x).__name__, x.a, x.arity)
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, tuple(dump(i) for i in x))
    return (type(x).__name__, repr(x))


def parse(text):
    return dump(KlongInterpreter().prog(text)[1])


class TestExampleCondChain(unittest.TestCase):
    def test_pairs_parse_to_same_tree(self):
        for old, new in PAIRS:
            with self.subTest(rewrite=new):
                self.assertEqual(parse(old), parse(new))
                self.assertNotEqual(old, new)

    def test_fixture_uses_rewrite_and_parses(self):
        with open(FIXTURE) as f:
            text = f.read()
        for old, new in PAIRS:
            with self.subTest(rewrite=new):
                self.assertIn(new, text)
                self.assertNotIn(old, text)
        KlongInterpreter().prog(text)

    def test_semantics_of_chain(self):
        k = KlongInterpreter()
        self.assertEqual(k(":[1;1:|0;2;3]"), 1)
        self.assertEqual(k(":[0;1:|1;2;3]"), 2)
        self.assertEqual(k(":[0;1:|0;2;3]"), 3)


if __name__ == "__main__":
    unittest.main()
