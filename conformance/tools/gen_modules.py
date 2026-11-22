#!/usr/bin/env python3
"""Generates conformance/cases/87-modules.cases from the readable specification below.

Usage (from the repository root):  python3 conformance/tools/gen_modules.py > conformance/cases/87-modules.cases

A case lists its main program (the `main` text) and, when it needs them, module files.
The runner has no fixtures, so the generated program writes the files itself (.oc .tc .p
.cc), runs the main text, deletes the files (.df) and ends with the value of the last
statement. Macros in main and module text (a file `name` of the case `c`):
    {P}name   absolute path, no extension:            /tmp/kgm87-c-name
    {R}name   ./kgm87-c-name       (module files only: relative to the importing file)
    {U}name   ../tmp/kgm87-c-name  in a module file (relative, through ..);
              /tmp/../tmp/kgm87-c-name in the main program (the program has no file: it is
              read from standard input, so only absolute paths are used there)
    {B}name   kgm87-c-name         (a bare name: searched beside the importing file, then KLONGPATH, then the library)
"""
import re, sys

CASES = []
def C(name, tags, main, files=None, v=None, err=None, out=None, why=None, nodel=False, al=False):
    CASES.append(dict(name=name, tags=tags, main=main, files=files or {}, v=v, err=err, out=out, why=why, al=al))

Q = "strings dicts qualified"            # qualified names on ordinary dictionaries
M = "strings dicts sysio qualified modules"   # needs the export variable, .l of module files and files

# ------------------------------------------------------------------ A. a.b on ordinary dictionaries
C("read", Q, 'd:::{;"a";1}\nd.a', v="1", why="d.a is d@\"a\"")
C("read-nested", Q, 'd:::{;"a";:{;"b";2}}\nd.a.b', v="2")
C("read-three-deep", Q, 'd:::{;"a";:{;"b";:{;"c";3}}}\nd.a.b.c', v="3")
C("read-in-expression", Q, 'd:::{;"a";4}\n(d.a*2)+d.a', v="12")
C("read-then-index", Q, 'd:::{;"l";[10 20 30]}\nd.l@1', v="20", why="d.l@1 is (d.l)@1: the name is d.l")
C("read-under-monads", Q, 'd:::{;"l";[10 20 30]}\n(#d.l),*d.l', v="[3 10]")
C("read-letters-and-digits", Q, 'd:::{;"a1";3}\nd.a1', v="3")
C("read-float-entry", Q + " float", 'd:::{;"a";1.5}\nd.a+2.5', v="4.0")
C("read-key-case-sensitive", Q, 'd:::{;"A";1}\nd.a', err="name")
C("read-missing-key", Q, 'd:::{;"a";1}\nd.b', err="name", why="decision 7 of the design: a missing key is a name error naming the dictionary and the key")
C("read-missing-key-nested", Q, 'd:::{;"a";:{;"b";1}}\nd.a.c', err="name")
C("at-still-gives-undefined", Q, 'd:::{;"a";1}\n:_(d@"b")', v="1", why="@ keeps the classic answer: only a.b is loud")
C("present-key-with-undefined-value", Q, 'd:::{;"a";1%0}\n:_(d.a)', v="1", why="the key is present, so a.b answers its value, which is :undefined")
C("symbol-key-is-not-the-string-key", Q, 'd:::{;:a;1}\nd.a', err="name", why="a.b is d@\"b\" with a string key")
C("key-with-dot-by-at", Q, 'd:::{;"a.b";1}\nd@"a.b"', v="1")
C("key-with-dot-not-reachable-by-dots", Q, 'd:::{;"a.b";1}\nd.a', err="name")
C("undefined-base-is-a-name-error", Q, 'zz.a', err="name", why="not the symbol :zz.a that an undefined name gives at strict level 0: a.b is never a symbol")
C("undefined-base-strict1", Q + " strict1", 'zz.a', err="name")
C("undefined-base-strict2", Q + " strict2", 'zz.a', err="name")
C("declared-unassigned-local-is-undefined", Q, 'f::{[d];d.a}\nf()', err="name")
C("base-is-an-integer", Q, 'n::5\nn.a', err="type")
C("base-is-a-list", Q, 'l::[1 2]\nl.a', err="type")
C("base-is-a-string", Q, 's::"abc"\ns.a', err="type")
C("base-is-a-symbol", Q, 's::(:abc)\ns.a', err="type")
C("base-is-a-function", Q, 'f::{x}\nf.a', err="type")
C("base-is-an-integer-strict1", Q + " strict1", 'n::5\nn.a', err="type")
C("missing-key-strict1", Q + " strict1", 'd:::{;"a";1}\nd.b', err="name")
C("missing-key-strict2", Q + " strict2", 'd:::{;"a";1}\nd.b', err="name")

C("write", Q, 'd:::{;"a";1}\nd.a::5\nd.a', v="5")
C("write-value", Q, 'd:::{;"a";1}\nd.a::7', v="7", why="the value of a.b::v is v, as for a::v")
C("write-seen-by-at", Q, 'd:::{;"a";1}\nd.a::5\nd@"a"', v="5", why="a.b::v sets the entry; it does not create a variable named a.b")
C("write-uses-read", Q, 'd:::{;"a";1}\nd.a::d.a+1\nd.a', v="2")
C("write-new-key", Q, 'd:::{;"a";1}\nd.b::2\nd@"b"', v="2", why="an ordinary dictionary takes a new entry (cfg.port::8080)")
C("write-nested", Q, 'd:::{;"a";:{;"b";1}}\nd.a.b::9\nd.a.b', v="9")
C("write-three-deep", Q, 'd:::{;"a";:{;"b";:{;"c";1}}}\nd.a.b.c::8\nd.a.b.c', v="8")
C("write-missing-middle-is-a-name-error", Q, 'd:::{;"a";1}\nd.x.y::1', err="name", why="no autovivification: principle 5")
C("write-undefined-base", Q, 'zz.a::1', err="name", why="at level 0 this used to create a variable named zz.a")
C("write-undefined-base-strict1", Q + " strict1", 'zz.a::1', err="name")
C("write-base-is-an-integer", Q, 'n::5\nn.a::1', err="type")
C("write-base-is-a-list", Q, 'l::[1 2 3]\nl.a::1', err="type")
C("write-through-alias", Q, 'd:::{;"a";1}\ne::d\ne.a::6\nd.a', v="6", why="dictionaries are shared, as with , today; principle 4 is not served, the language decision stands")
C("write-through-nested-alias", Q, 'd:::{;"a";:{;"b";1}}\ns::d.a\ns.b::5\nd.a.b', v="5", why="d.a is the nested dictionary itself, not a copy")
C("write-is-not-a-variable", Q, 'd:::{;"a";1}\nd.a::5\n:_(d.a)', v="0")
C("write-in-function-to-global-dict", Q, 'd:::{;"a";1}\nf::{d.a::9}\nf()\nd.a', v="9")
C("write-in-function-strict1", Q + " strict1", 'd:::{;"a";1}\nf::{d.b::2}\nf()\nd.b', v="2", why="an entry is not a variable: level 1 allows it")
C("write-through-parameter", Q, 'f(p)::{p.a::9;p.a}\nd:::{;"a";1}\nf(d)', v="9")
C("write-through-parameter-visible-to-caller", Q, 'f(p)::{p.a::9}\nd:::{;"a";1}\nf(d)\nd.a', v="9")
C("write-through-local", Q, 'f::{[d];d:::{;"a";1};d.a::4;d.a}\nf()', v="4")
C("write-destructuring-targets", Q, 'd:::{;"a";0;"b";0}\n[d.a d.b]::[1 2]\nd.a+d.b*10', v="21", why="a qualified name is a target wherever a name is")
C("write-destructuring-targets-strict2", Q + " strict2", 'd:::{;"a";0}\n[d.a q]::[1 2]\nd.a+q', v="3")

C("call-one", Q, 'd:::{;"f";{x+1}}\nd.f(2)', v="3")
C("call-two", Q, 'd:::{;"g";{x-y}}\nd.g(10;3)', v="7")
C("call-zero", Q, 'h::{7}\nd:::{;"h";h}\nd.h()', v="7", why="a niladic function literal written inside :{;...} is called there, so the entry is a name that holds one")
C("call-named-parameters", Q, 'f(a;b)::{a-b}\nd:::{;"f";f}\nd.f(5;2)', v="3")
C("call-nested", Q, 'd:::{;"m";:{;"f";{x*2}}}\nd.m.f(4)', v="8")
C("call-argument-is-a-call", Q, 'd:::{;"f";{x+1}}\nd.f(d.f(1))', v="3")
C("call-non-function", Q, 'd:::{;"a";1}\nd.a(2)', err="type")
C("call-missing-key", Q, 'd:::{;"a";1}\nd.nope(2)', err="name")
C("call-wrong-arity", Q, 'f(a;b)::{a-b}\nd:::{;"f";f}\nd.f(1;2;3)', err="arity", why="a.b(args) is the call of the entry: the usual arity rules")
C("call-too-few-is-a-projection", Q, 'f(a;b)::{a-b}\nd:::{;"f";f}\ng::d.f(10)\ng(3)', v="7", why="fewer arguments than the function takes make a projection, as for any call")
C("entry-is-a-value", Q, 'd:::{;"f";{x+1}}\ng::d.f\ng(1)', v="2")
C("passed-as-a-value", Q, 'd:::{;"f";{x+1}}\napply(h;a)::{h(a)}\napply(d.f;5)', v="6")
C("stored-in-a-dictionary", Q, 'd:::{;"f";{x+1}}\ne:::{;"g";d.f}\ne.g(1)', v="2")
C("adverb-each", Q, 'd:::{;"f";{x*x}}\nd.f\'[1 2 3]', v="[1 4 9]")
C("adverb-over", Q, 'd:::{;"f";{x+y}}\nd.f/[1 2 3 4]', v="10")
C("adverb-over-with-seed", Q, 'd:::{;"f";{x+y}}\n100 d.f/[1 2 3]', v="106")
C("adverb-scan", Q, 'd:::{;"f";{x+y}}\nd.f\\[1 2 3]', v="[1 3 6]")
C("adverb-each-pair", Q, 'd:::{;"f";{x+y}}\n[1 2 3]d.f\'[10 20 30]', v="[11 22 33]")
C("adverb-on-qualified-argument", Q, 'd:::{;"f";{x+1};"l";[1 2]}\nd.f\'d.l', v="[2 3]")
C("projection-first-open", Q, 'd:::{;"g";{x-y}}\nh::d.g(;1)\nh(5)', v="4")
C("projection-second-open", Q, 'd:::{;"g";{x-y}}\nh::d.g(10;)\nh(3)', v="7")

C("local-dictionary", Q, 'f::{[d];d:::{;"a";5};d.a}\nf()', v="5")
C("implicit-parameter-x", Q, 'f::{x.a}\nf(:{["a" 3]})', v="3")
C("implicit-parameters-y-z", Q, 'f::{y.k+z.k}\nf(0;:{["k" 1]};:{["k" 2]})', v="3")
C("implicit-x-is-a-list", Q, 'f::{x.len}\nf([1 2 3])', err="type", why="HAZARD: x.len used to be a name; it is now x@\"len\" and x is a list. The repo has no such program")
C("implicit-x-is-a-dictionary", Q, 'f::{x.len}\nf(:{["len" 3]})', v="3")
C("named-parameter", Q, 'f(p)::{p.a+1}\nf(:{["a" 3]})', v="4")
C("local-shadows-global", Q, 'd:::{;"a";1}\nf::{[d];d:::{;"a";2};d.a}\nf()+d.a', v="3")
C("dynamic-scope-is-kept", Q, 'd:::{;"a";1}\ng::{d.a}\nf::{[d];d:::{;"a";5};g()}\nf()', v="5", why="classic dynamic scoping of locals is unchanged outside modules (principle 7)")

C("locals-list-rejects-a-qualified-name", Q, 'f::{[d.a];1}', err="syntax")
C("parameter-list-rejects-a-qualified-name", Q, 'f(d.a)::{1}', err="syntax")
C("digit-segment-is-a-syntax-error", Q, 'a::[1 2]\na.5', err="syntax", why="a.5 was a variable name; a segment must start with a letter, so write a@5. Decision for the owner: syntax error rather than a silent classic variable")
C("trailing-dot-is-a-syntax-error", Q, 'd:::{;"a";1}\nd.a.', err="syntax")
C("space-breaks-the-name", Q, 'd:::{;"a";1}\nd .zz', err="name", why="d .zz is d followed by the name .zz, as today")
C("dot-after-a-call", Q, 'd:::{;"a";1}\nf::{d}\nf().zz', err="name", why="a.b attaches to a name only; (f())@\"a\" is the way to say it")
C("leading-dot-names-are-whole", Q, '.zz.a::5\n.zz.a', v="5", why="a name that starts with a dot is one name, never qualified: .srv.o .cli.h .os.argv .fpga.set")
C("leading-dot-hook-with-named-parameters", Q, '.hk.o(h)::{h+1}\n.hk.o(1)', v="2")
C("system-function-calls-are-unchanged", Q, '.fmt("{}";5)', v='"5"')
C("assignment-to-undeclared-in-function-level0", Q, 'zz::{[a];a.b::1}\nzz()', err="name", why="a is a declared local that was never assigned: undefined. At level 0 a.b::1 used to make a local variable named a.b")

# ------------------------------------------------------------------ B. the export variable and .l
CNT = 'n::0\ninc()::{n::n+1;n}\nexport:::{;"inc";inc}'
C("load-alias-basic", M, '.l("{P}m";:m)\nm.f(1)', files={"m": 'f(n)::{n+1}\nexport:::{;"f";f}'}, v="2")
C("import-returns-the-export-dictionary", M, '#.l("{P}m";:zz)', files={"m": 'f(n)::{n+1}\ng(n)::{n+2}\nexport:::{;"f";f;"g";g}'}, v="2")
C("import-nothing-in-globals-strict2", M + " strict2", '.l("{P}m";:m)\nf', files={"m": 'f(n)::{n+1}\nexport:::{;"f";f}'}, err="name", why="decision 1: nothing is written into the importer")
C("import-no-leak-of-top-level-assignment-strict2", M + " strict2", '.l("{P}m";:m)\nleak', files={"m": 'leak::1\nexport:::{;"a";1}'}, err="name")
C("private-helper-used-inside", M, '.l("{P}m";:m)\nm.f(5)', files={"m": 'h(n)::{n*2}\nf(n)::{h(n)+1}\nexport:::{;"f";f}'}, v="11")
C("private-helper-unreachable", M, '.l("{P}m";:m)\nm.h(5)', files={"m": 'h(n)::{n*2}\nf(n)::{h(n)+1}\nexport:::{;"f";f}'}, err="name")
C("private-helper-not-in-the-dictionary", M, '.l("{P}m";:m)\n:_(m@"h")', files={"m": 'h(n)::{n*2}\nf(n)::{h(n)+1}\nexport:::{;"f";f}'}, v="1")
C("private-state-unreachable", M, '.l("{P}m";:m)\nm.n', files={"m": CNT}, err="name")
C("rename-on-export", M, '.l("{P}m";:m)\nm.api(1)', files={"m": 'impl(n)::{n+10}\nexport:::{;"api";impl}'}, v="11", why="the exposed name may differ from the internal one")
C("rename-on-export-hides-the-internal-name", M, '.l("{P}m";:m)\nm.impl(1)', files={"m": 'impl(n)::{n+10}\nexport:::{;"api";impl}'}, err="name")
C("export-a-constant", M, '.l("{P}m";:m)\nm.version', files={"m": 'export:::{;"version";3}'}, v="3")
C("export-a-list", M, '.l("{P}m";:m)\nm.t@1', files={"m": 'export:::{;"t";[1 2 3]}'}, v="2")
C("export-empty-dictionary-is-fine", M, '#.l("{P}m";:zz)', files={"m": 'export:::{}'}, v="0", why="an explicit empty interface is a decision; a missing .export is probably a mistake")
C("export-key-not-a-string", M, '.l("{P}m";:zz)', files={"m": 'export:::{;1;2}'}, err="type")
C("export-not-a-dictionary", M, '.l("{P}m";:zz)', files={"m": 'export::[1 2]'}, err="type")
C("export-an-integer", M, '.l("{P}m";:zz)', files={"m": 'export::5'}, err="type")
C("export-a-string", M, '.l("{P}m";:zz)', files={"m": 'export::"a"'}, err="type")
C("export-key-is-a-symbol", M, '.l("{P}m";:zz)', files={"m": 'export:::{;:a;1}'}, err="type")
C("an-alias-for-a-classic-file-is-an-error", M, '.l("{P}m";:zz)', files={"m": 'f::{1}'}, err="type", why="a file with no `export` is classic; a second argument has nothing to bind, and the error says to write .l(path)")
C("l-of-a-classic-file-still-works", M, '.l("{P}c")\nf(2)', files={"c": 'f(n)::{n*3}'}, v="6")

C("same-file-twice-shares-state", M, '.l("{P}m";:a)\n.l("{P}m";:b)\na.inc()\nb.inc()', files={"m": CNT}, v="2", why="decision 3: the same module, cached by resolved file")
C("body-runs-once", M, '.l("{P}m";:a)\n.l("{P}m";:b)\na.v', files={"m": '.p("loaded")\nexport:::{;"v";1}'}, v="1", out="loaded")
C("cache-key-adds-the-extension", M, '.l("{P}m";:a)\n.l("{P}m.kg";:b)\na.inc()\nb.inc()', files={"m": CNT}, v="2")
C("cache-key-resolves-dot-dot", M, '.l("{P}m";:a)\n.l("{U}m";:b)\na.inc()\nb.inc()', files={"m": CNT}, v="2", why="the key is the resolved file, not the spelling")
C("state-is-private-to-the-module", M, 'n::50\n.l("{P}m";:m)\nm.inc()\nn', files={"m": CNT}, v="50")
C("module-global-beats-caller-local", M, '.l("{P}m";:m)\nf::{[n];n::99;m.get()}\nf()', files={"m": 'n::1\nget()::{n}\nexport:::{;"get";get}'}, v="1", why="decision: frames of another module are not visible to a module function (principle 4)")
C("module-inner-lambda-sees-its-own-parameters", M, '.l("{P}m";:m)\nm.adds(10;[1 2])', files={"m": 'adds(k;v)::{{x+k}\'v}\nexport:::{;"adds";adds}'}, v="[11 12]", why="inside one module, classic dynamic scope still holds")
C("two-modules-same-private-name", M, '.l("{P}a";:a)\n.l("{P}b";:b)\na.g()+b.g()*10', files={"a": 'n::1\nga()::{n}\nexport:::{;"g";ga}', "b": 'n::2\ngb()::{n}\nexport:::{;"g";gb}'}, v="21")
C("module-cannot-see-importer-globals-level0", M, 'secret::5\n.l("{P}m";:m)\nm.peek()', files={"m": 'peek()::{secret}\nexport:::{;"peek";peek}'}, v=":secret", why="the importer's names are not visible; at level 0 an unknown name still reads as its symbol, which is the classic trap")
C("module-cannot-see-importer-globals-strict1", M + " strict1", 'secret::5\n.l("{P}m";:m)\nm.peek()', files={"m": 'peek()::{secret}\nexport:::{;"peek";peek}'}, err="name")
C("callback-sees-its-own-module-globals", M, 'offset::100\ncb::{x+offset}\n.l("{P}m";:m)\nm.apply(cb;1)', files={"m": 'apply(f;a)::{f(a)}\nexport:::{;"apply";apply}'}, v="101")
C("callback-sees-the-frames-of-its-own-module", M, '.l("{P}m";:m)\ng::{[offset];offset::10;m.apply({x+offset};1)}\ng()', files={"m": 'apply(f;a)::{f(a)}\nexport:::{;"apply";apply}'}, v="11", why="the lambda was written in main, so main's frames stay visible through the module's frame")

C("relative-path-from-a-file", M, '.l("{P}a";:a)\na.bump()', files={"b": CNT, "a": '.l("{R}b";:b)\nbump()::{b.inc()}\nexport:::{;"bump";bump}'}, v="1", why="decision: ./ and ../ are relative to the importing file, never to the working directory")
C("relative-dot-dot-from-a-file", M, '.l("{P}a";:a)\na.bump()', files={"b": CNT, "a": '.l("{U}b";:b)\nbump()::{b.inc()}\nexport:::{;"bump";bump}'}, v="1")
C("relative-and-absolute-are-one-module", M, '.l("{P}a";:a)\n.l("{P}b";:b)\na.bump()\nb.inc()', files={"b": CNT, "a": '.l("{R}b";:b)\nbump()::{b.inc()}\nexport:::{;"bump";bump}'}, v="2")
C("three-levels", M, '.l("{P}a";:a)\na.f()', files={"c": 'export:::{;"v";7}', "b": '.l("{R}c";:c)\nget()::{c.v}\nexport:::{;"get";get}', "a": '.l("{R}b";:b)\nf()::{b.get()}\nexport:::{;"f";f}'}, v="7")
DIA = {"c": '.p("c")\nn::0\nbump()::{n::n+1;n}\nexport:::{;"bump";bump}',
       "a": '.l("{R}c";:c)\nexport:::{;"bump";c.bump}', "b": '.l("{R}c";:c)\nexport:::{;"bump";c.bump}'}
C("diamond-shares-one-instance", M, '.l("{P}a";:a)\n.l("{P}b";:b)\na.bump()\nb.bump()', files=DIA, v="2", out="c", why="the diamond: two importers, one module, one private counter, one load")
C("load-order-is-post-order", M, '.l("{P}b";:b)\n0', files={"c": '.p("c")\nexport:::{;"v";1}', "b": '.l("{R}c";:c)\n.p("b")\nexport:::{;"v";c.v}'}, v="0", out="c\nb")
C("cycle-is-an-error", M, '.l("{P}a";:zz)', files={"a": '.l("{R}b";:b)\nexport:::{;"v";1}', "b": '.l("{R}a";:a)\nexport:::{;"v";2}'}, err="system", why="decision: an error naming the cycle, not a half-built module: principles 4 and 5")
C("self-import-is-an-error", M, '.l("{P}a";:zz)', files={"a": '.l("{R}a";:a)\nexport:::{;"v";1}'}, err="system")

LAZY = {"m": '.p("loaded")\nexport:::{;"f";{x+1}}'}
C("lazy-import-in-a-function", M, '.p("before")\ng::{[q];.l("{P}m";:q);q.f(1)}\nr::g()\n.p("after")\nr', files=LAZY, v="2", out="before\nloaded\nafter", why="the module runs at the first call, not at definition")
C("lazy-import-is-freed-after-the-call", M, 'g::{[q];.l("{P}m";:q);q.f(1)}\ng()\ng()', files=LAZY, v="2", out="loaded\nloaded", why="the alias is local: once g returns nothing refers to the module (principle 4), so the next call loads the file again; keep the module in a global to share it")
C("lazy-import-kept-by-a-global-runs-once", M, 'keep::0\ng::{[q];.l("{P}m";:q);keep::q;q.f(1)}\ng()\ng()', files=LAZY, v="2", out="loaded")
C("lazy-import-is-local-strict2", M + " strict2", 'g::{[q];.l("{P}m";:q);q.f(1)}\ng()\nq', files=LAZY, err="name")
C("import-by-a-parameter-path", M, 'g(p)::{[q];.l(p;:q);q.f(1)}\ng("{P}m")', files={"m": 'f(n)::{n+1}\nexport:::{;"f";f}'}, v="2", why="the path is a value: the static compiler cannot follow this one (rule decl-literal)")


RO = {"m": 'f(n)::{n+1}\ng(n)::{n+2}\nexport:::{;"f";f;"g";g}'}
C("read-only-replace-an-entry", M, '.l("{P}m";:m)\nm.f::{x}', files=RO, err="type", why="decision 5: what .l returns cannot be modified from outside, so geo.scale resolves at compile time")
C("read-only-add-an-entry", M, '.l("{P}m";:m)\nm.zz::1', files=RO, err="type")
C("read-only-through-an-alias", M, '.l("{P}m";:m)\nc::m\nc.f::{x}', files=RO, err="type", why="read-only is a property of the dictionary, not of the name")
C("read-only-destructuring-target", M, '.l("{P}m";:m)\n[m.f m.g]::[1 2]', files=RO, err="type")
C("read-only-pair-join", M, '.l("{P}m";:m)\nm::m,"zz",,1', files=RO, err="type", why="GAP: , mutates a dictionary in place today; on a read-only one it must refuse")
C("read-only-reading-is-free", M, '.l("{P}m";:m)\nm.f(1)+m.g(1)', files=RO, v="5")
C("a-copy-is-writable", M, '.l("{P}m";:m)\ne:::{}\ne::e,m\ne.f::{x}\ne.f(7)', files=RO, v="7", why="joining into a new dictionary copies the entries; the copy is an ordinary dictionary")
C("a-copy-leaves-the-module-alone", M, '.l("{P}m";:m)\ne:::{}\ne::e,m\ne.f::{x}\nm.f(7)', files=RO, v="8")
C("module-writable-nested-dictionary-is-shared", M, '.l("{P}m";:a)\na.opts.n::5\n.l("{P}m";:b)\nb.opts.n', files={"m": 'opts:::{;"n";1}\nexport:::{;"opts";opts}'}, v="5", why="DECISION for the owner: read-only is shallow. Only the table is frozen; a dictionary stored in it is shared mutable state")

C("modules-are-values-in-a-dictionary", M, 'd:::{;"x";.l("{P}m";:zz)}\nd.x.f(1)', files={"m": 'f(n)::{n+1}\nexport:::{;"f";f}'}, v="2")
C("write-a-module-into-a-dictionary", M, 'd:::{}\nd.m::.l("{P}m";:zz)\nd.m.f(1)', files={"m": 'f(n)::{n+1}\nexport:::{;"f";f}'}, v="2")
C("polymorphism-by-name", M, 'reg:::{;"inc";.l("{P}a";:z1);"dbl";.l("{P}b";:z2)}\nk::"dbl"\nr::reg@k\nr.step(5)', files={"a": 'step(n)::{n+1}\nexport:::{;"step";step}', "b": 'step(n)::{n*2}\nexport:::{;"step";step}'}, v="10")
C("fold-with-a-qualified-function", M, 'reg:::{;"sum";.l("{P}a";:z1);"prod";.l("{P}b";:z2)}\nr::reg@"prod"\n1 r.step/[2 3 4]', files={"a": 'step(s;v)::{s+v}\nexport:::{;"step";step}', "b": 'step(s;v)::{s*v}\nexport:::{;"step";step}'}, v="24")
C("module-as-an-argument", M, '.l("{P}m";:m)\nuse(mod;a)::{mod.f(a)}\nuse(m;5)', files={"m": 'f(n)::{n+1}\nexport:::{;"f";f}'}, v="6")
C("module-function-in-adverb", M, '.l("{P}m";:m)\nm.f\'[1 2]', files={"m": 'f(n)::{n+1}\nexport:::{;"f";f}'}, v="[2 3]")
C("re-export-a-whole-module", M, '.l("{P}b";:b)\nb.base.f(1)', files={"a": 'f(n)::{n+1}\nexport:::{;"f";f}', "b": '.l("{R}a";:a)\nexport:::{;"base";a}'}, v="2")
C("re-export-a-function-under-another-name", M, '.l("{P}b";:b)\nb.inc(1)', files={"a": 'f(n)::{n+1}\nexport:::{;"f";f}', "b": '.l("{R}a";:a)\nexport:::{;"inc";a.f}'}, v="2")

C("l-in-a-module-loads-into-the-module", M, '.l("{P}m";:m)\nm.f(2)', files={"h": 'helper(n)::{n*3}', "m": '.l("{P}h")\nf(n)::{helper(n)}\nexport:::{;"f";f}'}, v="6", why="decision: .l loads into the scope of the file that contains the call")
C("l-in-a-module-does-not-leak-strict2", M + " strict2", '.l("{P}m";:m)\nhelper(1)', files={"h": 'helper(n)::{n*3}', "m": '.l("{P}h")\nf(n)::{helper(n)}\nexport:::{;"f";f}'}, err="name")
C("l-in-main-does-not-reach-a-module", M, '.l("{P}c")\n.l("{P}m";:m)\nm.f([2 4])', files={"c": 'mean(v)::{99}', "m": 'mean(v)::{+/v}\nf(v)::{mean(v)}\nexport:::{;"f";f}'}, v="6")
C("classic-module-block-inside-an-imported-file", M, '.l("{P}m";:m)\nm.scale(4)', files={"m": '.module(:geo)\nunit::10\nscale(n)::{n*unit}\n.module(0)\nexport:::{;"scale";scale}'}, v="40")
C("classic-module-block-in-main-is-unchanged", Q, '.module(:geo)\nunit::10\nscale(n)::{n*unit}\n.module(0)\nscale(4)', v="40")

C("error-in-the-body", M, '.l("{P}m";:zz)', files={"m": '1+"a"\nexport:::{;"a";1}'}, err="type")
C("error-in-an-exported-function", M, '.l("{P}m";:m)\nm.f(0)', files={"m": 'f(n)::{n+"a"}\nexport:::{;"f";f}'}, err="type")
C("strict2-reaches-the-module-text", M + " strict2", '.l("{P}m";:zz)', files={"m": 'a::b+1\nexport:::{;"a";a}'}, err="name", why="a module is loaded at the strict level of the importer")

C("bundled-csv-through-import", M, '.l("csv";:csv)\ncsv.split("a,b,c")', v='["a" "b" "c"]', why="klongpy/lib/csv.kg converted to a module that exports split read make write load")
C("bundled-csv-roundtrip", M, '.l("csv";:csv)\n#csv.split(csv.make(["a" "b"]))', v="2")
C("bundled-csv-keeps-its-helpers-private", M, '.l("csv";:csv)\n:_(csv@"unesc")', v="1", why="unesc, escape, unquote and ctlm are helpers")

C("static-declarations-are-inert", M + " static float", '.l("{P}m";:m)\nm.run(4.0)', files={"m": 'run(v)::{v*2.0}\n.entry("run";"float -> float")\nexport:::{;"run";run}'}, v="8.0", why="the declarations run as they stand; kgpy compile c must see through .l (DESIGN-GAPS)")

C("export-is-a-snapshot", M, '.l("{P}m";:m)\nm.inc()\nm.n', files={"m": 'n::0\ninc()::{n::n+1;n}\nexport:::{;"n";n;"inc";inc}'}, v="0", why="DECISION: .export copies values at the call. Exporting a variable exports its value then, not the variable; state is reached through functions")
C("exported-function-keeps-its-module-scope", M, '.l("{P}m";:m)\ng::m.inc\ng()+g()', files={"m": 'n::0\ninc()::{n::n+1;n}\nexport:::{;"inc";inc}'}, v="3", why="a function value taken out of a module still runs in that module: 1+2")
C("export-order-statements-after-export-still-run", M, '.l("{P}m";:m)\nm.f(1)', files={"m": 'f(n)::{n+k}\nexport:::{;"f";f}\nk::10'}, v="11", why="DECISION: .export fixes the table, not the end of the file; a lint warns when it is not the last statement")

# ---- R9 (revised): .l of a file that calls .export binds the module dictionary to the file's name, as .l("name";:name) does,
# and defines no plain name
LM = {"m": 'h(n)::{n*2}\nf(n)::{h(n)+1}\nexport:::{;"f";f}'}
C("l-of-a-module-binds-the-file-name", M, '.l("{A}m")\n{N}m.f(2)', files=LM, v="5", al=True, why="R9: .l(\"csv\") then csv.read(...) keeps working (principle 7); the file's name is the alias")
C("l-of-a-module-defines-no-plain-name", M + " strict2", '.l("{A}m")\nf(1)', files=LM, err="name", al=True, why="R9: nothing but the alias is defined (principle 4)")
C("l-of-a-module-keeps-the-private-names-private", M, '.l("{A}m")\n{N}m.h(1)', files=LM, err="name", al=True)
C("l-of-a-module-binds-a-read-only-dictionary", M, '.l("{A}m")\n{N}m.f::{x}', files=LM, err="type", al=True)
C("l-of-a-module-binds-the-keys", M, '.l("{A}m")\n{N}m.api(1)', files={"m": 'impl(n)::{n+10}\nexport:::{;"api";impl}'}, v="11", al=True)
C("l-of-a-module-binds-a-constant", M, '.l("{A}m")\n{N}m.version', files={"m": 'export:::{;"version";3}'}, v="3", al=True)
C("l-then-alias-share-the-module", M, '.l("{A}m")\n.l("{A}m";:m)\n{N}m.inc()\nm.inc()', files={"m": CNT}, v="2", al=True, why="the module is cached by real path")
C("alias-then-l-share-the-module", M, '.l("{A}m";:m)\n.l("{A}m")\nm.inc()\n{N}m.inc()', files={"m": CNT}, v="2", al=True)
C("l-twice-is-the-same-module", M, '.l("{A}m")\n.l("{A}m")\n{N}m.inc()\n{N}m.inc()', files={"m": CNT}, v="2", al=True, why="the second .l does not run the body again")
C("l-of-a-module-the-importers-global-is-not-seen", M, 'n::50\n.l("{A}m")\n{N}m.inc()', files={"m": CNT}, v="1", al=True)
C("l-of-a-module-replaces-a-global-of-the-same-name", M, '{N}m::7\n.l("{A}m")\n{N}m.f(2)', files=LM, v="5", al=True)
C("l-of-a-module-without-the-extension", M, '.l("{A}m.kg")\n{N}m.f(2)', files=LM, v="5", al=True)
C("l-of-a-module-error-in-the-body", M, '.l("{P}m")', files={"m": '1+"a"\nexport:::{;"a";1}'}, err="type")
C("export-assigned-twice-the-last-one-is-the-interface", M, '.l("{P}m";:m)\nm.b', files={"m": 'export:::{;"a";1}\nexport:::{;"b";2}'}, v="2", why="the interface is read when the module finishes loading")
C("export-assigned-twice-the-first-one-is-gone", M, '.l("{P}m";:m)\nm.a', files={"m": 'export:::{;"a";1}\nexport:::{;"b";2}'}, err="name")
C("l-of-a-module-from-a-module", M, '.l("{A}a";:a)\na.g(2)', files={"m": LM["m"], "a": '.l("{A}m")\ng(n)::{{N}m.f(n)}\nexport:::{;"g";g}'}, v="5", al=True, why="R9 inside a module: the alias is bound in the loading module's scope")
# ---- the cache holds modules weakly: a module that no value refers to is freed, and a later import runs the file again
LOADED = {"m": '.p("loaded")\nexport:::{;"v";1}'}
C("a-module-nothing-refers-to-is-freed", M, '.l("{P}m";:a)\na::0\n.l("{P}m";:b)\nb.v', files=LOADED, v="1", out="loaded\nloaded", why="principle 4: nothing is held that no value refers to; a library released with m::0 gives its memory back")
C("a-freed-module-starts-again", M, '.l("{P}m";:a)\na.inc()\na::0\n.l("{P}m";:b)\nb.inc()', files={"m": CNT}, v="1", why="the state went with the module")
C("a-module-with-a-reference-is-kept", M, '.l("{P}m";:a)\n.l("{P}m";:b)\na::0\n.l("{P}m";:c)\nc.v', files=LOADED, v="1", out="loaded", why="while any value refers to it, every importer gets the same instance")
C("a-function-of-a-module-keeps-it", M, '.l("{P}m";:a)\ng::a.inc\na::0\n.l("{P}m";:b)\ng()\nb.inc()', files={"m": CNT}, v="2", why="a function taken out of a module is a reference to it")
C("a-module-loaded-by-a-live-module-is-kept", M, '.l("{P}a";:a)\n.l("{P}c";:b)\nb.bump()\na.bump()', files=DIA, v="2", out="c", why="the diamond: c is referred to by the live module a, and by b")
C("a-module-freed-with-its-importer", M, '.l("{P}a";:a)\na::0\n.l("{P}c";:b)\nb.bump()', files=DIA, v="1", out="c\nc", why="a refers to c; once a is freed nothing does")
C("l-of-a-classic-file-in-a-module-file-is-unchanged", M, '.l("{P}a";:a)\na.g(2)', files={"c": 'f(n)::{n*3}', "a": '.l("{P}c")\ng(n)::{f(n)}\nexport:::{;"g";g}'}, v="6")
C("l-of-a-classic-module-block-is-unchanged", M, '.l("{P}c")\nscale(4)', files={"c": '.module(:geo)\nunit::10\nscale(n)::{n*unit}\n.module(0)'}, v="40", why="principle 7: classic .module blocks through .l")
C("l-of-a-classic-module-block-keeps-the-name-reachable", M, '.l("{P}c")\nunit', files={"c": '.module(:geo)\nunit::10\nscale(n)::{n*unit}\n.module(0)'}, v="10")
C("bundled-csv-with-l-binds-csv", M, '.l("csv")\ncsv.split("a,b,c")', v='["a" "b" "c"]', why="R9: .l(\"csv\") then csv.split(...) keeps working (principle 7)")
C("bundled-csv-with-l-defines-no-plain-name", M + " strict2", '.l("csv")\nsplit("a")', err="name", why="principle 4: only the alias csv is defined")
C("bundled-csv-with-l-keeps-its-helpers-private", M, '.l("csv")\ncsv.unesc("a")', err="name")
C("bundled-csv-with-l-and-import-share", M, '.l("csv")\n.l("csv";:c)\n#c.split(csv.make(["a" "b"]))', v="2")
C("bundled-nstat-with-l-binds-nstat", M + " float", '.l("nstat")\nnstat.mu([1 2 3 4])', v="2.5", why="nstat is a module: .l(\"nstat\") binds the one name nstat and its functions are entries of it")
C("bundled-nstat-with-l-defines-no-plain-name", M + " strict2", '.l("nstat")\nmu([1 2])', err="name", why="principle 4: only the alias nstat is defined")
C("bundled-nstat-with-l-leaves-the-family-names-alone", M, 'n::5\nt::6\nf::7\n.l("nstat")\n(n+t)+f', v="18", why="the distribution families n t f u r b x2 F ... are entries of nstat, not variables of the program")
C("bundled-nstat-family-is-a-dictionary-in-the-module", M + " float", '.l("nstat")\nnstat.n.cdf(0;0;1)', v="0.5", why="a family is nstat.n.cdf(x;m;s)")
C("bundled-nstat-classic-module-block-sees-the-modules-names", M + " float", '.l("nstat")\n#nstat.spl(2;[[1 1][2 4][3 9]])', v="3", why="spline.kg is a classic .module block loaded by the module; its free names (gauss, from math) are the module's own")

# ---- R2/R7: a name that starts with a dot is one name, also in a module file; it is a system name
C("dot-name-in-a-module-file-is-a-system-name", M, '.l("{P}m";:m)\n.hk.f(1)', files={"m": '.hk.f(n)::{n+1}\nexport:::{;"a";1}'}, v="2", why="a name that starts with a dot is one name for the whole program: the server looks hooks up by it (design point 4, principle 7)")
C("dot-name-from-a-module-is-visible-to-another-module", M, '.l("{P}b";:b)\nb.f()', files={"a": '.hk.g()::{7}\nexport:::{;"v";1}', "b": '.l("{R}a";:a)\nf()::{.hk.g()}\nexport:::{;"f";f}'}, v="7")
C("dot-name-set-in-the-importer-is-visible-in-a-module", M, '.hk.n::5\n.l("{P}m";:m)\nm.f()', files={"m": 'f()::{.hk.n}\nexport:::{;"f";f}'}, v="5", why="system names are visible from every module (design point 7)")
C("dot-name-variable-written-by-a-module-function", M, '.l("{P}m";:m)\nm.set(9)\n.hk.n', files={"m": '.hk.n::0\nset(v)::{.hk.n::v}\nexport:::{;"set";set}'}, v="9")
C("module-function-sees-system-functions", M, '.l("{P}m";:m)\nm.f(5)', files={"m": 'f(n)::{.fmt("{}";n)}\nexport:::{;"f";f}'}, v='"5"', why="design point 7: locals, then the module's globals, then system names")
C("module-body-sees-system-functions", M, '.l("{P}m";:zz)@"v"', files={"m": 'v::.fmt("{}";3)\nexport:::{;"v";v}'}, v='"3"')

# ---- shallow read-only: every spelling of a change that the language has
C("read-only-pair-join-list", M, '.l("{P}m";:m)\nm,["zz" 1]', files=RO, err="type", why="a read-only dictionary refuses , in every form, since , mutates its left side")
C("read-only-join-dictionary", M, '.l("{P}m";:m)\nm,:{;"zz";1}', files=RO, err="type")
C("read-only-join-replaces-an-entry", M, '.l("{P}m";:m)\nm,["f" 1]', files=RO, err="type")
C("read-only-drop-a-key", M, '.l("{P}m";:m)\n"f"_m', files=RO, err="type", why="_ drops in place today")
C("read-only-copy-by-join-into-empty", M, '.l("{P}m";:m)\ne:::{}\ne,m\ne.f(7)', files=RO, v="8", why="joining into a writable dictionary copies the entries (M11)")
C("read-only-nested-module-is-read-only", M, '.l("{P}b";:b)\nb.base.f::{x}', files={"a": 'f(n)::{n+1}\nexport:::{;"f";f}', "b": '.l("{R}a";:a)\nexport:::{;"base";a}'}, err="type", why="a module stored in another export is still the module's own read-only dictionary")
C("read-only-key-by-at-assignment-does-nothing", M, '.l("{P}m";:m)\nm@"f"::7\nm.f(1)', files=RO, v="2", why="d@\"a\"::5 does nothing today (a call target is not assignable)")

# ---- scope details
C("module-global-assigned-by-its-function", M, '.l("{P}m";:m)\nm.inc()\nm.inc()', files={"m": CNT}, v="2")
C("module-function-writes-a-new-global-in-its-module", M, '.l("{P}m";:m)\nm.set(4)\nm.get()', files={"m": 'set(v)::{k::v}\nget()::{k}\nexport:::{;"set";set;"get";get}'}, v=":k", why="level 0: a new name assigned in a function is local to the call; this pins the classic rule inside a module too")
C("module-function-writes-a-declared-global", M, '.l("{P}m";:m)\nm.set(4)\nm.get()', files={"m": 'k::0\nset(v)::{k::v}\nget()::{k}\nexport:::{;"set";set;"get";get}'}, v="4", why="an existing module global is assigned where it lives")
C("module-function-does-not-assign-the-importers-global", M, 'k::1\n.l("{P}m";:m)\nm.set(4)\nk', files={"m": 'k::0\nset(v)::{k::v}\nexport:::{;"set";set}'}, v="1")
C("module-function-with-the-importers-local-of-the-same-name", M, '.l("{P}m";:m)\nf::{[k];k::5;m.get()}\nf()', files={"m": 'k::2\nget()::{k}\nexport:::{;"get";get}'}, v="2")
C("module-function-does-not-see-its-callers-module-frames", M, '.l("{P}m";:m)\nm.outer(3)', files={"m": 'inner()::{p}\nouter(p)::{inner()}\nexport:::{;"outer";outer}'}, v=":p", why="principle 4: a function with named parameters reads its own locals and its module's globals, never its caller's locals, in a module too; level 0 reads an unknown name as its symbol")
C("module-function-does-not-see-its-callers-module-frames-strict1", M + " strict1", '.l("{P}m";:m)\nm.outer(3)', files={"m": 'inner()::{p}\nouter(p)::{inner()}\nexport:::{;"outer";outer}'}, err="name", why="the same, loud at strict level 1")
C("classic-module-function-sees-its-callers-module-frames", M, '.l("{P}m";:m)\nm.outer(3)', files={"m": 'inner::{p}\nouter::{[p];p::3;inner()}\nexport:::{;"outer";outer}'}, v="3", why="D4: a function without a parameter list keeps classic dynamic scope between the functions of one module")
C("module-function-does-not-see-the-importers-frames", M + " strict2", '.l("{P}m";:m)\nf::{[p];p::3;m.get()}\nf()', files={"m": 'get()::{p}\nexport:::{;"get";get}'}, err="name", why="principle 4: the importer's locals are not the module's names")
C("module-top-level-names-do-not-reach-the-next-module", M + " strict2", '.l("{P}a";:a)\n.l("{P}b";:b)\nb.f()', files={"a": 'shared::1\nexport:::{;"v";1}', "b": 'f()::{shared}\nexport:::{;"f";f}'}, err="name")
C("module-inside-a-function-of-a-module", M, '.l("{P}a";:a)\na.f()', files={"b": 'v::7\nget::{v}\nexport:::{;"get";get}', "a": 'f()::{[q];.l("{R}b";:q);q.get()}\nexport:::{;"f";f}'}, v="7", why="lazy import in a module function: the alias is local, the module is the same instance")
C("an-exported-lambda-keeps-its-module", M, '.l("{P}m";:m)\nm.get()', files={"m": 'v::7\nget::{v}\nexport:::{;"get";get}'}, v="7", why="a function literal in .export closes over its module's scope")
C("an-exported-projection-keeps-its-module", M, '.l("{P}m";:m)\nm.add1(1)', files={"m": 'k::10\nadd(a;b)::{a+b+k}\nexport:::{;"add1";add(;1)}'}, v="12")
C("projection-of-a-qualified-function-with-module-global", M, '.l("{P}m";:m)\np::m.add(;1)\np(2)', files={"m": 'k::10\nadd(a;b)::{a+b+k}\nexport:::{;"add";add}'}, v="13")
C("adverb-over-a-module-function-with-module-global", M, '.l("{P}m";:m)\nm.f\'[1 2]', files={"m": 'k::10\nf(a)::{a+k}\nexport:::{;"f";f}'}, v="[11 12]")
C("module-function-called-from-a-callback-of-the-importer", M, '.l("{P}m";:m)\nm.f\'[1 2]\n{m.f(x)}\'[1 2]', files={"m": 'k::10\nf(a)::{a+k}\nexport:::{;"f";f}'}, v="[11 12]")
C("dictionary-entry-call-with-module-function", M, '.l("{P}m";:m)\nd:::{;"g";m.f}\nd.g(1)', files={"m": 'k::10\nf(a)::{a+k}\nexport:::{;"f";f}'}, v="11")
C("module-state-through-two-aliases", M, '.l("{P}m";:a)\nd:::{;"x";a}\nd.x.inc()\na.inc()', files={"m": CNT}, v="2")
C("two-modules-same-exports-called-through-a-dictionary", M, 'r:::{;"p";.l("{P}a";:z1);"q";.l("{P}b";:z2)}\nr.p.g()+r.q.g()', files={"a": 'g()::{1}\nexport:::{;"g";g}', "b": 'g()::{2}\nexport:::{;"g";g}'}, v="3")
C("l-alias-x-is-not-a-name-a-module-can-take", M, '.l("{P}m";:x)', files={"m": 'export:::{;"v";5}'}, err="name", why="x, y and z are the implicit parameters and cannot be bound names")

# ---- parsing hazards
C("qualified-name-in-a-conditional", Q, 'd:::{;"a";1;"b";2}\n:[d.a=1;d.b;0]', v="2")
C("qualified-name-after-an-operator", Q, 'd:::{;"a";3}\n-d.a', v="-3")
C("qualified-name-right-to-left", Q, 'd:::{;"a";3;"b";2}\n10-d.a-d.b', v="9", why="Klong evaluates right to left: 10-(3-2)")
C("qualified-name-in-a-data-list-is-a-symbol-list", Q, 'd:::{;"a";3;"b";2}\n[d.a d.b]', v="[:d.a :d.b]", why="a list of names is data (principle 7)")
C("qualified-name-in-an-evaluated-list", Q, 'd:::{;"a";3;"b";2}\n[;d.a;d.b]', v="[3 2]")
C("qualified-name-with-a-space-before-the-dot", Q, 'd:::{;"a";1}\nd .zz', err="name", why="the name .zz is undefined: a space ends the name (design point 4)")
C("segment-may-contain-digits-after-a-letter", Q, 'd:::{;"a1b2";6}\nd.a1b2', v="6")
C("segment-may-not-start-with-a-digit-write", Q, 'd::[1 2]\nd.1::5', err="syntax")
C("a-number-with-a-dot-is-still-a-number", Q + " float", 'd:::{;"a";1}\nd.a+1.5', v="2.5")
C("quoted-dotted-symbol-is-unchanged", Q, ':a.b', v=":a.b", why="a quoted symbol is not a lookup (principle 7); also in 10-lexical")
C("quoted-dotted-symbol-of-a-dictionary-base", Q, 'd:::{;"a";1}\n:d.a', v=":d.a", why="the symbol :d.a; a quoted name never reads an entry")
C("dot-leading-name-with-segments-is-one-name", Q, '.q.w.e::4\n.q.w.e', v="4")
C("dot-leading-name-is-not-a-dictionary-lookup", Q + " strict2", '.q:::{;"w";1}\n.q.w', err="name", why="the name .q.w is one name and is undefined")


# ------------------------------------------------------------------ C. the export variable
C("export-in-the-main-program-is-a-plain-variable", Q, 'export:::{;"a";1}\nexport.a', v="1", why="a program that is run directly is not loaded as a module: `export` is an ordinary variable there")
C("export-from-a-variable", M, '.l("{P}m";:m)\nm.a', files={"m": 'd:::{;"a";5}\nexport::d'}, v="5", why="the interpreter reads the value of `export` when the file ends; only the compiler needs the literal form")
C("export-is-frozen", M, '.l("{P}m";:m)\nm.a::2', files={"m": 'export:::{;"a";1}'}, err="type", why="the interface is read-only from outside")
C("export-private-names-stay-private", M + " strict2", '.l("{P}m";:m)\nhelper', files={"m": 'helper::1\nexport:::{;"a";1}'}, err="name")
C("export-a-private-variable-by-key", M, '.l("{P}m";:m)\nm.b', files={"m": 'a::1\nb::2\nexport:::{;"b";b}'}, v="2")
C("export-assignment-inside-a-function-does-not-make-a-module", M, '.l("{P}m")', files={"m": 'f::{export:::{;"a";1}}\nf()\n1'}, v="1", why="only a top-level assignment makes the file a module; the file runs as a classic one and .l gives the value of its last statement")
C("export-name-in-a-string-does-not-make-a-module", M, '.l("{P}m")', files={"m": 's::"export:::{;1;1}"\n3'}, v="3")
C("export-name-in-a-comment-does-not-make-a-module", M, '.l("{P}m")', files={"m": 'f::{1}\n.comment("END")\nexport::1\nEND\n4'}, v="4", why="a comment block is not code")
C("export-as-a-prefix-of-another-name", M, '.l("{P}m")', files={"m": 'exports::1\n5'}, v="5", why="the name is `export`, not a name that starts with it")
C("export-with-the-qualified-name-is-not-the-variable", M, '.l("{P}m")', files={"m": 'e:::{}\ne.export::1\n6'}, v="6", why="e.export writes an entry of e; it is not an assignment to the variable export")

# ------------------------------------------------------------------ D. .l: the three forms and the errors
LM2 = {"m": 'h(n)::{n*2}\nf(n)::{h(n)+1}\ng(n)::{n*10}\nexport:::{;"f";f;"g";g}'}
C("l-stem-form-returns-the-module", M, '#.l("{A}m")', files=LM2, v="2", al=True, why=".l returns the module when the file is a module")
C("l-stem-form-binds-the-stem", M, '.l("{A}m")\n{N}m.f(2)', files=LM2, v="5", al=True, why="import x")
C("l-alias-form-binds-the-alias", M, '.l("{P}m";:s)\ns.f(2)', files=LM2, v="5", why="import x as s")
C("l-alias-form-returns-the-module", M, '#.l("{P}m";:s)', files=LM2, v="2")
C("l-alias-form-does-not-bind-the-stem", M + " strict2", '.l("{A}m";:s)\n{N}m', files=LM2, err="name", al=True)
C("l-alias-form-binds-a-read-only-module", M, '.l("{P}m";:s)\ns.f::{x}', files=LM2, err="type")
C("l-alias-form-an-alias-that-is-not-a-symbol", M, '.l("{P}m";"s")', files=LM2, err="type", why="a string is not an alias: write :s; a list of strings is the names form")
C("l-alias-form-same-module-two-aliases", M, '.l("{P}m";:p)\n.l("{P}m";:q)\np.f(1)+q.f(1)', files=LM2, v="6")
C("l-alias-form-in-a-function-is-local", M + " strict2", 'g::{[s];.l("{P}m";:s);s.f(1)}\ng()\ns', files=LM2, err="name")
C("l-alias-form-in-a-module", M, '.l("{P}a";:a)\na.v(2)', files={"m": LM2["m"], "a": '.l("{R}m";:inner)\nv(n)::{inner.f(n)}\nexport:::{;"v";v}'}, v="5", why="the alias is bound in the scope of the file that contains the call")

C("l-names-form-binds-the-names", M, '.l("{P}m";["f" "g"])\nf(1)+g(1)', files=LM2, v="13", why="from x import f, g")
C("l-names-form-binds-plain-names-only-those", M + " strict2", '.l("{P}m";["f"])\ng(1)', files=LM2, err="name")
C("l-names-form-does-not-bind-the-stem", M + " strict2", '.l("{A}m";["f"])\n{N}m', files=LM2, err="name", al=True)
C("l-names-form-returns-the-module-too", M, '#.l("{P}m";["f"])', files=LM2, v="2")
C("l-names-form-empty-list-binds-nothing", M, '#.l("{P}m";[])', files=LM2, v="2")
C("l-names-form-empty-list-binds-nothing-strict2", M + " strict2", '.l("{P}m";[])\nf', files=LM2, err="name")
C("l-names-form-a-name-listed-twice", M, '.l("{P}m";["f" "f"])\nf(1)', files=LM2, v="3")
C("l-names-form-values-are-the-entries", M, '.l("{P}m";["v" "w"])\nv+w*10', files={"m": 'export:::{;"v";3;"w";4}'}, v="43")
C("l-names-form-binds-in-the-caller-not-the-module", M, 'f::9\n.l("{P}m";["f"])\nf(1)', files=LM2, v="3", why="the entry replaces the importer's own variable")
C("l-names-form-missing-entry-is-a-name-error", M, '.l("{P}m";["f" "zz"])', files=LM2, err="name", why="the error names the entry")
C("l-names-form-private-name-is-a-missing-entry", M, '.l("{P}m";["h"])', files=LM2, err="name")
C("l-names-form-missing-entry-binds-nothing", M + " strict2", '.l("{P}m";["f" "zz"])', files=LM2, err="name")
C("l-names-form-not-a-list", M, '.l("{P}m";5)', files=LM2, err="type")
C("l-names-form-key-not-a-string", M, '.l("{P}m";["f" 1])', files=LM2, err="type")
C("l-names-form-key-is-a-symbol", M, '.l("{P}m";[:f])', files=LM2, err="type")
C("l-names-form-an-entry-that-is-not-a-valid-name", M, '.l("{P}m";["a.b"])', files={"m": 'export:::{;"a.b";1}'}, err="name", why="an entry whose key cannot be a Klong name cannot be bound as one: the error says to use the alias form")
C("l-second-argument-of-another-type", M, '.l("{P}m";3)', files=LM2, err="type", why="the second argument is a symbol (alias) or a list of strings (names)")
C("l-second-argument-a-dictionary", M, '.l("{P}m";:{;"f";1})', files=LM2, err="type")
C("l-second-argument-on-a-classic-file", M, '.l("{P}c";["f"])', files={"c": 'f(n)::{n*3}'}, err="type", why="a classic file has no interface to take names from; the error says to write .l(path)")
C("l-second-argument-names-on-a-classic-file-defines-nothing", M, '.l("{P}c";:s)', files={"c": 'f(n)::{n*3}'}, err="type")
C("l-three-arguments", M, '.l("{P}m";:s;1)', files=LM2, err="arity")
C("l-no-arguments", M, '.l()', err="arity")
C("l-path-not-a-string", M, '.l(5)', err="type")
C("l-path-not-a-string-with-alias", M, '.l(5;:s)', err="type")
C("l-stem-is-not-a-name-and-no-alias", M, '.l("{P}m")', files=LM2, err="type", why="the stem kgm87-...-m has a hyphen: it cannot be a Klong name; the error says to write .l(path;:name)")
C("l-stem-is-not-a-name-the-alias-form-works", M, '.l("{P}m";:ok)\nok.f(1)', files=LM2, v="3")
C("l-stem-is-not-a-name-the-names-form-works", M, '.l("{P}m";["f"])\nf(1)', files=LM2, v="3")
C("l-missing-file-with-alias", M, '.l("/tmp/kgm87-no-such-file-anywhere";:s)', err="system")
C("l-missing-file-with-names", M, '.l("/tmp/kgm87-no-such-file-anywhere";["f"])', err="system")
C("l-missing-file", M, '.l("/tmp/kgm87-no-such-file-anywhere")', err="system")

# a classic file loads as before
C("l-classic-defines-its-names", M, '.l("{P}c")\nf(2)', files={"c": 'f(n)::{n*3}'}, v="6")
C("l-classic-returns-its-last-value", M, '.l("{P}c")', files={"c": 'f(n)::{n*3}\n7'}, v="7")
C("l-classic-in-a-function-defines-the-names-for-the-function", M, 'g::{.l("{P}c");f(2)}\ng()', files={"c": 'f(n)::{n*3}'}, v="6")
C("l-classic-twice-runs-twice", M, '.l("{P}c")\n.l("{P}c")\n1', files={"c": '.p("run")'}, v="1", out="run\nrun", why="only a module is loaded once")
C("l-classic-is-not-cached-by-a-module-of-the-same-file-name", M, '.l("{P}c")\ng(1)', files={"c": 'g(n)::{n+1}'}, v="2")
C("l-module-twice-the-second-form-may-differ", M, '.l("{P}m";:a)\n.l("{P}m";["f"])\na.f(1)+f(1)', files=LM2, v="6", why="the module is the same: the loading forms differ only in what they bind")
C("l-module-twice-runs-the-body-once", M, '.l("{P}m";:a)\n.l("{P}m";:b)\n1', files={"m": '.p("body")\nexport:::{;"v";1}'}, v="1", out="body")
C("l-module-names-form-twice-runs-the-body-once", M, '.l("{P}m";["v"])\n.l("{P}m";["v"])\nv', files={"m": '.p("body")\nexport:::{;"v";1}'}, v="1", out="body")
C("l-module-with-a-failed-body-is-not-cached", M, '.l("{P}m";:a)', files={"m": '.p("b")\n1+"x"\nexport:::{;"v";1}'}, err="type")

# ------------------------------------------------------------------ E. search order
# A bare name is searched in the directory of the file that contains the .l (the working directory at the REPL), then the KLONGPATH
# directories, then the bundled library. ./x and ../x are relative to the importing file. The runner reads the main program from
# standard input and starts in a directory of its own, so the cases below exercise the importing file's directory with module files.
SIB = {"b": 'export:::{;"v";7}', "a": '.l("{B}b";:b)\nexport:::{;"v";b.v+1}'}
C("bare-name-from-a-module-file-is-found-beside-it", M, '.l("{P}a";:a)\na.v', files=SIB, v="8", why="the directory of the importing file comes first")
C("bare-name-from-a-module-file-names-form", M, '.l("{P}a";:a)\na.v', files={"b": 'export:::{;"v";7}', "a": '.l("{B}b";["v"])\nexport:::{;"v";v+1}'}, v="8")
C("bare-name-beside-a-classic-file", M, '.l("{P}a")\nf(1)', files={"b": 'g(n)::{n+1}', "a": '.l("{B}b")\nf(n)::{g(n)}'}, v="2", why="the same order for classic files")
C("bare-name-with-the-extension-beside-the-file", M, '.l("{P}a";:a)\na.v', files={"b": 'export:::{;"v";7}', "a": '.l("{B}b.kg";:b)\nexport:::{;"v";b.v+1}'}, v="8")
C("bare-name-not-beside-the-file-is-an-error", M, '.l("{P}a";:a)', files={"a": '.l("kgm87-no-such-sibling";:b)\nexport:::{;"v";1}'}, err="system")
C("bare-name-from-the-main-program-is-not-found-in-a-temporary-directory", M, '.l("{B}m";:m)', files={"m": 'export:::{;"v";1}'}, err="system", why="the main program has no file: its directory is the working directory of the runner, not /tmp")
C("dot-slash-is-relative-to-the-importing-file", M, '.l("{P}a";:a)\na.v', files={"b": 'export:::{;"v";7}', "a": '.l("{R}b";:b)\nexport:::{;"v";b.v+1}'}, v="8")
C("dot-dot-slash-is-relative-to-the-importing-file", M, '.l("{P}a";:a)\na.v', files={"b": 'export:::{;"v";7}', "a": '.l("{U}b";:b)\nexport:::{;"v";b.v+1}'}, v="8")
C("absolute-path-is-itself", M, '.l("{P}m";:m)\nm.v', files={"m": 'export:::{;"v";2}'}, v="2")
C("l-by-a-bundled-name-finds-the-library", M, '.l("csv")\n#csv.split("a,b")', v="2", why="no local file of that name: the bundled library is last")
C("l-by-a-bundled-name-with-an-alias", M, '.l("csv";:c)\n#c.split("a,b")', v="2")
C("l-by-a-bundled-name-with-an-alias-does-not-bind-the-stem", M + " strict2", '.l("csv";:c)\ncsv', err="name")
C("l-by-a-bundled-name-with-names", M, '.l("csv";["split"])\n#split("a,b,c")', v="3")
C("l-by-a-bundled-name-missing-entry", M, '.l("csv";["nope"])', err="name")
C("l-by-a-bundled-name-private-entry", M, '.l("csv";["unesc"])', err="name")
C("l-by-a-bundled-classic-library-with-an-alias-is-an-error", M, '.l("math";:m)', err="type", why="math is not a module")
C("two-files-are-two-modules", M, '.l("{P}m";:a)\n.l("{P}n";:b)\na.inc()\nb.inc()', files={"m": CNT, "n": CNT}, v="1", why="the key is the resolved file")

# ------------------------------------------------------------------ F. host functions are values (tag static): see 97-static.cases for the declarations
SX = "strings dicts sysio qualified modules static"
RD = {"read": 'adc::.extern("read_adc";"int -> float")\ndac::.extern("write_dac";"int float -> int")\nwide::.extern("read_wide";"int int int int -> int")\nexport:::{;"adc";adc;"dac";dac;"wide";wide}'}
C("extern-module-call-without-a-host-function", SX, '.l("{P}read";:read)\nread.adc(3)', files=RD, err="system", why="a target's functions live in an ordinary module file used as .l(\"read\") then read.adc(3)")
C("extern-module-wrong-arity", SX, '.l("{P}read";:read)\nread.adc(1;2)', files=RD, err="arity")
C("extern-module-two-parameters-wrong-arity", SX, '.l("{P}read";:read)\nread.dac(1)', files=RD, err="arity")
C("extern-module-wide-function-without-a-host-function", SX, '.l("{P}read";:read)\nread.wide(1;2;3;4)', files=RD, err="system")
C("extern-module-wide-function-too-many-arguments", SX, '.l("{P}read";:read)\nread.wide(1;2;3;4;5)', files=RD, err="arity")
C("extern-module-wide-function-fewer-arguments-is-a-projection", SX, '.l("{P}read";:read)\np::read.wide(1;2;3)\np(4)', files=RD, err="system")
C("extern-module-missing-entry", SX, '.l("{P}read";:read)\nread.nope', files=RD, err="name")
C("extern-module-is-read-only", SX, '.l("{P}read";:read)\nread.adc::{x}', files=RD, err="type")
C("extern-module-names-form", SX, '.l("{P}read";["adc"])\nadc(1)', files=RD, err="system")
C("extern-module-the-c-symbol-is-not-an-entry", SX, '.l("{P}read";:read)\nread.read_adc', files=RD, err="name")
C("extern-module-a-function-value-can-be-passed", SX, '.l("{P}read";:read)\napply(f;a)::{f(a)}\napply(read.adc;1)', files=RD, err="system")
C("extern-module-the-same-symbol-in-two-modules", SX, '.l("{P}a";:a)\n.l("{P}b";:b)\n1', files={"a": 'f::.extern("tick";"-> int")\nexport:::{;"f";f}', "b": 'g::.extern("tick";"-> int")\nexport:::{;"g";g}'}, v="1", why="the same symbol declared twice with the same signature is one declaration")
C("extern-module-the-same-symbol-with-different-signatures", SX, '.l("{P}a";:a)\n.l("{P}b";:b)', files={"a": 'f::.extern("tick";"-> int")\nexport:::{;"f";f}', "b": 'g::.extern("tick";"-> float")\nexport:::{;"g";g}'}, err="type")
C("extern-module-different-names-for-one-symbol", SX, '.l("{P}a";:a)\na.f()', files={"a": 'f::.extern("tick";"-> int")\nh::.extern("tick";"-> int")\nexport:::{;"f";f;"h";h}'}, err="system")
C("extern-in-a-module-does-not-define-the-symbol-in-the-importer", SX + " strict2", '.l("{P}a";:a)\ntick', files={"a": 'f::.extern("tick";"-> int")\nexport:::{;"f";f}'}, err="name")
C("extern-statement-form-in-a-module-file-is-an-error", SX, '.l("{P}a";:a)', files={"a": '.extern("tick";"-> int")\nexport:::{;"v";1}'}, err="type", why="the discarded form declared a global in version 1; the value must be kept")
C("extern-dotted-symbol-in-a-module-file", SX, '.l("{P}a";:a)', files={"a": 'f::.extern("read.adc";"int -> float")\nexport:::{;"f";f}'}, err="type", why="the dotted-extern special case is gone: a symbol is a C identifier")
C("extern-and-a-user-dictionary-of-the-same-name-do-not-interact", SX, 'read:::{;"adc";5}\n.l("{P}read";:rd)\nread.adc', files=RD, v="5", why="the extern no longer creates a dictionary")

# ------------------------------------------------------------------ writer
def kstr(s): return '"' + s.replace('"', '""') + '"'

def render(c):
    idx = CASES.index(c)
    # a case that binds a module with .l names the file with letters and digits only: the alias is the file's name, {N}x in the text
    pre = ("/tmp/kgm87a%d" % idx) if c["al"] else "/tmp/kgm87-" + c["name"] + "-"
    def mac(t, in_file=False):
        t = t.replace("{A}", "/tmp/kgm87a%d" % idx).replace("{N}", "kgm87a%d" % idx)
        u = "../tmp/kgm87-" if in_file else "/tmp/../tmp/kgm87-"
        return (t.replace("{P}", pre).replace("{R}", "./kgm87-" + c["name"] + "-").replace("{U}", u + c["name"] + "-")
                 .replace("{B}", "kgm87-" + c["name"] + "-"))
    lines = []
    paths = []
    if c["files"]:
        lines.append("kgw(p;t)::{[c];c::.oc(p);.tc(c);{.p(x)}'t;.tc(0);.cc(c)}")
        for fname, src in c["files"].items():
            path = pre + fname + ".kg"
            paths.append(path)
            body = "[" + " ".join(kstr(mac(l, True)) for l in src.split("\n")) + "]"
            lines.append("kgw(%s;%s)" % (kstr(path), body))
    main = [mac(l) for l in c["main"].split("\n")]
    if paths:
        last = main[-1]
        main = main[:-1] + ["kgr::" + last] + [".df(%s)" % kstr(p) for p in paths] + ["kgr"]
    lines += main
    return lines, paths

out = []
out.append("# Modules: the export variable, .l of modules, host functions in modules and qualified names a.b. Generated by conformance/tools/gen_modules.py: edit the generator, not this file.")
out.append("# docs/design-principles.md decides every open question; a `why` line names the principle.")
out.append("#")
out.append("# Tags: qualified = a.b on dictionaries (reading, writing, calling); modules = the export variable and .l of a module file")
out.append("# (a case writes its module files into /tmp with .oc .tc .p .cc and deletes them with .df, hence strings dicts sysio).")
out.append("# Macros in the comments: {P} is the absolute path, {R} the ./ path, {U} the ../tmp/ path and {B} the bare name (found beside the importing file) of the case's own files.")
out.append("# A case that fails leaves /tmp/kgm87-* files behind (the runner stops at the first error). File names are unique per case.")
out.append("# Error kinds: a missing key is `name`; a base that is not a dictionary, writing to a read-only dictionary and every wrong argument of .l or .extern are `type`;")
out.append("# every failure of the module graph (missing file, cycle, a bare name that no directory of the search order has) is `system`.")
out.append("")
names = set()
for c in CASES:
    assert c["name"] not in names, c["name"]; names.add(c["name"])
    assert (c["v"] is None) != (c["err"] is None) or c["v"] is None and c["err"] is None
    out.append("@ modules.%s %s" % (c["name"], c["tags"]))
    if c["why"]: out.append("# " + c["why"])
    for fname, src in c["files"].items():
        for i, l in enumerate(src.split("\n")):
            out.append("# %s.kg: %s" % (fname, l))
    lines, _ = render(c)
    for l in lines: out.append("> " + l if l else ">")
    if c["err"] is not None: out.append("! " + c["err"])
    elif c["v"] is not None:
        for l in c["v"].split("\n"): out.append("= " + l)
    if c["out"] is not None:
        for l in c["out"].split("\n"): out.append("~ " + l)
    out.append("")
sys.stdout.write("\n".join(out))
sys.stderr.write("%d cases\n" % len(CASES))
