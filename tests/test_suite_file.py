import unittest
from klongpy import KlongInterpreter
from utils import create_test_klong


class FailedUnitTest(Exception):
    pass


def die(x,y):
    raise FailedUnitTest(f"expected {x} got {y}")


class TestSuiteFile(unittest.TestCase):

    def test_fail(self):
        klong = create_test_klong()
        klong('t("success";1;1)')
        with self.assertRaises(RuntimeError):
            klong('t("fail";0;1)')

    def test_call_lambda(self):
        klong = KlongInterpreter()
        klong['die'] = die
        with self.assertRaises(FailedUnitTest):
            klong('foo::{die(x;y)};foo("a";"b")')

    def test_simple_script(self):
        t = """
wl::{.w(x);.p("");x}
wl("hello")
        """
        klong = KlongInterpreter()
        r = klong(t)
        self.assertEqual(r, 'hello')

    def test_suite_head(self):
        t = """
:"Klong test suite; nmh 2015--2020; public domain"

.comment("end-of-comment")
abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
0123456789
~!@#$%^&*()_+-=`[]{}:;"'<,>.
end-of-comment

err::0
wl::{.w(x);.p("")}
fail::{err::1;.d("failed: ");.p(x);.d("expected: ");wl(z);.d("got: ");wl(y);die(x;y)}
t::{:[~y~z;fail(x;y;z);[]]}

rnd::{:[x<0;-1;1]*_0.5+#x}
rndn::{rnd(x*10^y)%10^y}

:" Atom "
t("@:foo"       ; @:foo     ; 1)
t("@0"          ; @0        ; 1)
t("@123"        ; @123      ; 1)
t("@-1"         ; @-1       ; 1)
t("@1.23"       ; @1.23     ; 1)
t("@1e5"        ; @1e5      ; 1)
        """
        klong = KlongInterpreter()
        klong['die'] = die
        klong(t)

    def test_dict_in_situ_mutation(self):
        klong = create_test_klong()
        t = """
:" Dictionaries: in situ mutation (pre-assign s)"
s::!100
D:::{};{D,x,x}'!5
t("D:::{};{D,x,x}'!5;D" ; s@<s::*'D ; [0 1 2 3 4])
        """
        klong(t)

    def test_dyad_contexts(self):
        """
        Verify that we properly evaluate (KGCall) the dyadic function
        """
        klong = create_test_klong()
        klong['die'] = die
        t = """
:" Dyad Contexts "
f::{x,y}
g::{x,y,z}
t("1,2"                   ; 1,2                   ; [1 2])
t("0,:\[1 2 3]"           ; 0,:\[1 2 3]           ; [[0 1] [0 2] [0 3]])
t("0{x,y}[1 2 3]"         ; 0{x,y}[1 2 3]         ; [0 1 2 3])
t("0{x,y}:/[1 2 3]"       ; 0{x,y}:/[1 2 3]       ; [[1 0] [2 0] [3 0]])
t("0f[1 2 3]"             ; 0f[1 2 3]             ; [0 1 2 3])
t("9f:/[1 2 3]"           ; 9f:/[1 2 3]           ; [[1 9] [2 9] [3 9]])
t("[1 2 3]g(;0;)[4 5 6]"  ; [1 2 3]g(;0;)[4 5 6]  ; [1 2 3 0 4 5 6])
t("[1 2 3]g(;0;)'[4 5 6]" ; [1 2 3]g(;0;)'[4 5 6] ; [[1 0 4] [2 0 5] [3 0 6]])
        """
        klong(t)

    def test_module(self):
        t = """
:" Modules "
.module(:test)
a::1
g::{a}
f::{g()}
s::{a::x}
.module(0)

       t("a"    ; a    ; 1)
       t("g()"  ; g()  ; 1)
       t("s(2)" ; s(2) ; 2)
       t("g()"  ; g()  ; 2)
a::0 ; t("g()"  ; g()  ; 2)
g::0 ; t("f()"  ; f()  ; 2)
        """
        klong = create_test_klong()
        klong(t)

    def test_file_by_lines(self):
        """
        Test the suite file line by line using our own t()
        """
        klong = create_test_klong()
        with open("tests/klong_suite.kg", "r") as f:
            skip_header = True
            i = 0
            for r in f.readlines():
                if skip_header:
                    if r.startswith("rnd::"):
                        skip_header = False
                    else:
                        continue
                r = r.strip()
                if len(r) == 0:
                    continue
                i += 1
                klong.exec(r)
            print(f"executed {i} lines")

    def test_file_custom_test(self):
        """
        Test the suite file in one go using our own t()
        """
        klong = create_test_klong()
        with open("tests/klong_suite.kg", "r") as f:
            r = f.read()
            i = r.index('rnd::')
            r = r[i:]
            klong(r)

    def test_file(self):
        """
        Test the entire suite file.
        """
        klong = KlongInterpreter()
        with open("tests/klong_suite.kg", "r") as f:
            klong(f.read())
            r = klong('err')
            self.assertEqual(r, 0)
            r = klong['err']
            self.assertEqual(r, 0)


if __name__ == '__main__':
  unittest.main(failfast=True, exit=False)
