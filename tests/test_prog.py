import unittest
from klongpy import KlongInterpreter
from klongpy.core import KGSym, read_sys_comment
from utils import *

class TestProg(unittest.TestCase):

    def test_read_string_vs_op(self):
        """
        This test makes sure that we can parse a string that's the same character string as an operation.
        """
        klong = KlongInterpreter()
        r = klong("""
        bar::{x}
lin2::{bar("+")}
lin2()
        """)
        self.assertEqual(r,'+')

    def test_read_string_arg(self):
        """
        This test makes sure that we can parse a string that's the same character string as an operation.
        """
        klong = KlongInterpreter()
        r = klong("""
lin2::{.d("+");{.d((2+#x):^"-");.d("+")}'x;.p("");x}
lin2(1)
            """)
        self.assertEqual(r, 1)

    def test_prog_arr(self):
        klong = KlongInterpreter()
        r = klong.prog("[[]]")[1][0]
        r = [x.tolist() for x in r]
        self.assertEqual(str(r), "[[]]")

    @unittest.skip
    def test_prog_define(self):
        klong = KlongInterpreter()
        r = klong.prog('t::{:[~y~z;fail(x;y;z);[]]}')[1]
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].a.a, "::")
        self.assertEqual(r[0].args[0], KGSym("t"))
        self.assertEqual(len(r[0].args[1]), 1)
        self.assertEqual(len(r[0].args[1][0]), 3)

    @unittest.skip
    def test_prog_fn(self):
        klong = KlongInterpreter()
        r = klong.prog('t("fv()"     ; fv()     ; 1)')[1]
        self.assertEqual(len(r), 1)
        self.assertEqual(len(r[0]), 2)
        self.assertEqual(len(r[0][1]), 3)

    @unittest.skip
    def test_prog_multi_prog(self):
        klong = KlongInterpreter()
        r = klong.prog('fv::{[a];a::1;a} ; t("fv()"     ; fv()     ; 1)')[1]
        self.assertEqual(len(r), 2)
        self.assertEqual(len(r[0]), 3)
        self.assertEqual(len(r[0][2]), 1)
        self.assertEqual(len(r[0][2][0]), 3)
        self.assertEqual(len(r[0][2][0][0]), 1)
        self.assertEqual(len(r[0][2][0][1]), 3)
        self.assertEqual(len(r[1]), 2)
        self.assertEqual(len(r[1][1]), 3)
        r = klong('fv::{[a];a::1;a} ; fv()')
        self.assertEqual(r[1], 1)

    @unittest.skip
    def test_prog(self):
        klong = KlongInterpreter()
        s = '"{x-y}:(1;)@2"        ; {x-y}:(1;)@2        ; -1'
        i,p = klong.prog(s)
        self.assertEqual(i, len(s))
        self.assertEqual(p[0], '{x-y}:(1;)@2')
        self.assertEqual(str(p[1]), "[[[[:x, '-', :y]], [1, None]], '@', 2]")
        self.assertEqual(str(p[2]), "['-', 1]")
        a = klong.call(p[1])
        b = klong.call(p[2])
        self.assertEqual(a, b)

    def test_sys_comment_once(self):
        t = """
.comment("*****")

.fastpow     when set, x^y compiles to .pow(x;y)

*****X
        """
        x = t.index("X")
        cmd = '.comment("*****")'
        i = read_sys_comment(t, t.index(cmd)+len(cmd), "*****")
        assert i == x

    def test_sys_comment_replicated(self):
        t = """
.comment("*****")

.fastpow     when set, x^y compiles to .pow(x;y)

************************************************************************X
        """
        x = t.index("X")
        cmd = '.comment("*****")'
        i = read_sys_comment(t, t.index(cmd)+len(cmd), "*****")
        assert i == x
