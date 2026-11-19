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
    {B}name   kgm87-c-name         (a bare name: searched in the library path only)
"""
import re, sys

CASES = []
def C(name, tags, main, files=None, v=None, err=None, out=None, why=None, nodel=False, al=False):
    CASES.append(dict(name=name, tags=tags, main=main, files=files or {}, v=v, err=err, out=out, why=why, al=al))

Q = "strings dicts qualified"            # qualified names on ordinary dictionaries
M = "strings dicts sysio qualified modules"   # needs .import/.export and files

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

# ------------------------------------------------------------------ B. .import and .export
CNT = 'n::0\ninc()::{n::n+1;n}\n.export(:{;"inc";inc})'
C("import-basic", M, 'm::.import("{P}m")\nm.f(1)', files={"m": 'f(n)::{n+1}\n.export(:{;"f";f})'}, v="2")
C("import-returns-the-export-dictionary", M, '#.import("{P}m")', files={"m": 'f(n)::{n+1}\ng(n)::{n+2}\n.export(:{;"f";f;"g";g})'}, v="2")
C("import-nothing-in-globals-strict2", M + " strict2", 'm::.import("{P}m")\nf', files={"m": 'f(n)::{n+1}\n.export(:{;"f";f})'}, err="name", why="decision 1: nothing is written into the importer")
C("import-no-leak-of-top-level-assignment-strict2", M + " strict2", 'm::.import("{P}m")\nleak', files={"m": 'leak::1\n.export(:{;"a";1})'}, err="name")
C("private-helper-used-inside", M, 'm::.import("{P}m")\nm.f(5)', files={"m": 'h(n)::{n*2}\nf(n)::{h(n)+1}\n.export(:{;"f";f})'}, v="11")
C("private-helper-unreachable", M, 'm::.import("{P}m")\nm.h(5)', files={"m": 'h(n)::{n*2}\nf(n)::{h(n)+1}\n.export(:{;"f";f})'}, err="name")
C("private-helper-not-in-the-dictionary", M, 'm::.import("{P}m")\n:_(m@"h")', files={"m": 'h(n)::{n*2}\nf(n)::{h(n)+1}\n.export(:{;"f";f})'}, v="1")
C("private-state-unreachable", M, 'm::.import("{P}m")\nm.n', files={"m": CNT}, err="name")
C("rename-on-export", M, 'm::.import("{P}m")\nm.api(1)', files={"m": 'impl(n)::{n+10}\n.export(:{;"api";impl})'}, v="11", why="the exposed name may differ from the internal one")
C("rename-on-export-hides-the-internal-name", M, 'm::.import("{P}m")\nm.impl(1)', files={"m": 'impl(n)::{n+10}\n.export(:{;"api";impl})'}, err="name")
C("export-a-constant", M, 'm::.import("{P}m")\nm.version', files={"m": '.export(:{;"version";3})'}, v="3")
C("export-a-list", M, 'm::.import("{P}m")\nm.t@1', files={"m": '.export(:{;"t";[1 2 3]})'}, v="2")
C("export-empty-dictionary-is-fine", M, '#.import("{P}m")', files={"m": '.export(:{})'}, v="0", why="an explicit empty interface is a decision; a missing .export is probably a mistake")
C("export-key-not-a-string", M, '.import("{P}m")', files={"m": '.export(:{;1;2})'}, err="type")
C("export-not-a-dictionary", M, '.import("{P}m")', files={"m": '.export([1 2])'}, err="type")
C("export-twice", M, '.import("{P}m")', files={"m": '.export(:{;"a";1})\n.export(:{;"b";2})'}, err="system")
C("export-arity", M, '.import("{P}m")', files={"m": '.export(1;2)'}, err="arity", why="the torch backend's .export(fn;input;path) must be renamed")
C("export-in-the-main-program-is-inert", Q, 'd::.export(:{;"a";1})\nd.a', v="1", why="a file can be run directly as a script: .export gives its argument back and does nothing else")
C("no-export-is-an-error", M, '.import("{P}m")', files={"m": 'f::{1}'}, err="system", why="decision: a forgotten .export must not give an empty module; .l runs a file for effect (principles 4 and 5)")
C("l-of-a-classic-file-still-works", M, '.l("{P}c")\nf(2)', files={"c": 'f(n)::{n*3}'}, v="6")

C("path-not-a-string", M, '.import(5)', err="type")
C("import-arity", M, '.import("a";["x"];3)', err="arity")
C("import-no-arguments", M, '.import()', err="arity")
C("same-file-twice-shares-state", M, 'a::.import("{P}m")\nb::.import("{P}m")\na.inc()\nb.inc()', files={"m": CNT}, v="2", why="decision 3: the same module, cached by resolved file")
C("body-runs-once", M, 'a::.import("{P}m")\nb::.import("{P}m")\na.v', files={"m": '.p("loaded")\n.export(:{;"v";1})'}, v="1", out="loaded")
C("cache-key-adds-the-extension", M, 'a::.import("{P}m")\nb::.import("{P}m.kg")\na.inc()\nb.inc()', files={"m": CNT}, v="2")
C("cache-key-resolves-dot-dot", M, 'a::.import("{P}m")\nb::.import("{U}m")\na.inc()\nb.inc()', files={"m": CNT}, v="2", why="the key is the resolved file, not the spelling")
C("state-is-private-to-the-module", M, 'n::50\nm::.import("{P}m")\nm.inc()\nn', files={"m": CNT}, v="50")
C("module-global-beats-caller-local", M, 'm::.import("{P}m")\nf::{[n];n::99;m.get()}\nf()', files={"m": 'n::1\nget()::{n}\n.export(:{;"get";get})'}, v="1", why="decision: frames of another module are not visible to a module function (principle 4)")
C("module-inner-lambda-sees-its-own-parameters", M, 'm::.import("{P}m")\nm.adds(10;[1 2])', files={"m": 'adds(k;v)::{{x+k}\'v}\n.export(:{;"adds";adds})'}, v="[11 12]", why="inside one module, classic dynamic scope still holds")
C("two-modules-same-private-name", M, 'a::.import("{P}a")\nb::.import("{P}b")\na.g()+b.g()*10', files={"a": 'n::1\nga()::{n}\n.export(:{;"g";ga})', "b": 'n::2\ngb()::{n}\n.export(:{;"g";gb})'}, v="21")
C("module-cannot-see-importer-globals-level0", M, 'secret::5\nm::.import("{P}m")\nm.peek()', files={"m": 'peek()::{secret}\n.export(:{;"peek";peek})'}, v=":secret", why="the importer's names are not visible; at level 0 an unknown name still reads as its symbol, which is the classic trap")
C("module-cannot-see-importer-globals-strict1", M + " strict1", 'secret::5\nm::.import("{P}m")\nm.peek()', files={"m": 'peek()::{secret}\n.export(:{;"peek";peek})'}, err="name")
C("callback-sees-its-own-module-globals", M, 'offset::100\ncb::{x+offset}\nm::.import("{P}m")\nm.apply(cb;1)', files={"m": 'apply(f;a)::{f(a)}\n.export(:{;"apply";apply})'}, v="101")
C("callback-sees-the-frames-of-its-own-module", M, 'm::.import("{P}m")\ng::{[offset];offset::10;m.apply({x+offset};1)}\ng()', files={"m": 'apply(f;a)::{f(a)}\n.export(:{;"apply";apply})'}, v="11", why="the lambda was written in main, so main's frames stay visible through the module's frame")

C("relative-path-from-a-file", M, 'a::.import("{P}a")\na.bump()', files={"b": CNT, "a": 'b::.import("{R}b")\nbump()::{b.inc()}\n.export(:{;"bump";bump})'}, v="1", why="decision: ./ and ../ are relative to the importing file, never to the working directory")
C("relative-dot-dot-from-a-file", M, 'a::.import("{P}a")\na.bump()', files={"b": CNT, "a": 'b::.import("{U}b")\nbump()::{b.inc()}\n.export(:{;"bump";bump})'}, v="1")
C("relative-and-absolute-are-one-module", M, 'a::.import("{P}a")\nb::.import("{P}b")\na.bump()\nb.inc()', files={"b": CNT, "a": 'b::.import("{R}b")\nbump()::{b.inc()}\n.export(:{;"bump";bump})'}, v="2")
C("three-levels", M, 'a::.import("{P}a")\na.f()', files={"c": '.export(:{;"v";7})', "b": 'c::.import("{R}c")\nget()::{c.v}\n.export(:{;"get";get})', "a": 'b::.import("{R}b")\nf()::{b.get()}\n.export(:{;"f";f})'}, v="7")
DIA = {"c": '.p("c")\nn::0\nbump()::{n::n+1;n}\n.export(:{;"bump";bump})',
       "a": 'c::.import("{R}c")\n.export(:{;"bump";c.bump})', "b": 'c::.import("{R}c")\n.export(:{;"bump";c.bump})'}
C("diamond-shares-one-instance", M, 'a::.import("{P}a")\nb::.import("{P}b")\na.bump()\nb.bump()', files=DIA, v="2", out="c", why="the diamond: two importers, one module, one private counter, one load")
C("load-order-is-post-order", M, 'b::.import("{P}b")\n0', files={"c": '.p("c")\n.export(:{;"v";1})', "b": 'c::.import("{R}c")\n.p("b")\n.export(:{;"v";c.v})'}, v="0", out="c\nb")
C("cycle-is-an-error", M, '.import("{P}a")', files={"a": 'b::.import("{R}b")\n.export(:{;"v";1})', "b": 'a::.import("{R}a")\n.export(:{;"v";2})'}, err="system", why="decision: an error naming the cycle, not a half-built module: principles 4 and 5")
C("self-import-is-an-error", M, '.import("{P}a")', files={"a": 'a::.import("{R}a")\n.export(:{;"v";1})'}, err="system")

LAZY = {"m": '.p("loaded")\n.export(:{;"f";{x+1}})'}
C("lazy-import-in-a-function", M, '.p("before")\ng::{[q];q::.import("{P}m");q.f(1)}\nr::g()\n.p("after")\nr', files=LAZY, v="2", out="before\nloaded\nafter", why="the module runs at the first call, not at definition")
C("lazy-import-is-freed-after-the-call", M, 'g::{[q];q::.import("{P}m");q.f(1)}\ng()\ng()', files=LAZY, v="2", out="loaded\nloaded", why="the alias is local: once g returns nothing refers to the module (principle 4), so the next call loads the file again; keep the module in a global to share it")
C("lazy-import-kept-by-a-global-runs-once", M, 'q::0\ng::{q::.import("{P}m");q.f(1)}\ng()\ng()', files=LAZY, v="2", out="loaded")
C("lazy-import-is-local-strict2", M + " strict2", 'g::{[q];q::.import("{P}m");q.f(1)}\ng()\nq', files=LAZY, err="name")
C("import-by-a-parameter-path", M, 'g(p)::{[q];q::.import(p);q.f(1)}\ng("{P}m")', files={"m": 'f(n)::{n+1}\n.export(:{;"f";f})'}, v="2", why="the path is a value: the static compiler cannot follow this one (rule decl-literal)")


RO = {"m": 'f(n)::{n+1}\ng(n)::{n+2}\n.export(:{;"f";f;"g";g})'}
C("read-only-replace-an-entry", M, 'm::.import("{P}m")\nm.f::{x}', files=RO, err="type", why="decision 5: what .import returns cannot be modified from outside, so geo.scale resolves at compile time")
C("read-only-add-an-entry", M, 'm::.import("{P}m")\nm.zz::1', files=RO, err="type")
C("read-only-through-an-alias", M, 'm::.import("{P}m")\nc::m\nc.f::{x}', files=RO, err="type", why="read-only is a property of the dictionary, not of the name")
C("read-only-destructuring-target", M, 'm::.import("{P}m")\n[m.f m.g]::[1 2]', files=RO, err="type")
C("read-only-pair-join", M, 'm::.import("{P}m")\nm::m,"zz",,1', files=RO, err="type", why="GAP: , mutates a dictionary in place today; on a read-only one it must refuse")
C("read-only-reading-is-free", M, 'm::.import("{P}m")\nm.f(1)+m.g(1)', files=RO, v="5")
C("a-copy-is-writable", M, 'm::.import("{P}m")\ne:::{}\ne::e,m\ne.f::{x}\ne.f(7)', files=RO, v="7", why="joining into a new dictionary copies the entries; the copy is an ordinary dictionary")
C("a-copy-leaves-the-module-alone", M, 'm::.import("{P}m")\ne:::{}\ne::e,m\ne.f::{x}\nm.f(7)', files=RO, v="8")
C("module-writable-nested-dictionary-is-shared", M, 'a::.import("{P}m")\na.opts.n::5\nb::.import("{P}m")\nb.opts.n', files={"m": 'opts:::{;"n";1}\n.export(:{;"opts";opts})'}, v="5", why="DECISION for the owner: read-only is shallow. Only the table is frozen; a dictionary stored in it is shared mutable state")

C("modules-are-values-in-a-dictionary", M, 'd:::{;"x";.import("{P}m")}\nd.x.f(1)', files={"m": 'f(n)::{n+1}\n.export(:{;"f";f})'}, v="2")
C("write-a-module-into-a-dictionary", M, 'd:::{}\nd.m::.import("{P}m")\nd.m.f(1)', files={"m": 'f(n)::{n+1}\n.export(:{;"f";f})'}, v="2")
C("polymorphism-by-name", M, 'reg:::{;"inc";.import("{P}a");"dbl";.import("{P}b")}\nk::"dbl"\nr::reg@k\nr.step(5)', files={"a": 'step(n)::{n+1}\n.export(:{;"step";step})', "b": 'step(n)::{n*2}\n.export(:{;"step";step})'}, v="10")
C("fold-with-a-qualified-function", M, 'reg:::{;"sum";.import("{P}a");"prod";.import("{P}b")}\nr::reg@"prod"\n1 r.step/[2 3 4]', files={"a": 'step(s;v)::{s+v}\n.export(:{;"step";step})', "b": 'step(s;v)::{s*v}\n.export(:{;"step";step})'}, v="24")
C("module-as-an-argument", M, 'm::.import("{P}m")\nuse(mod;a)::{mod.f(a)}\nuse(m;5)', files={"m": 'f(n)::{n+1}\n.export(:{;"f";f})'}, v="6")
C("module-function-in-adverb", M, 'm::.import("{P}m")\nm.f\'[1 2]', files={"m": 'f(n)::{n+1}\n.export(:{;"f";f})'}, v="[2 3]")
C("re-export-a-whole-module", M, 'b::.import("{P}b")\nb.base.f(1)', files={"a": 'f(n)::{n+1}\n.export(:{;"f";f})', "b": 'a::.import("{R}a")\n.export(:{;"base";a})'}, v="2")
C("re-export-a-function-under-another-name", M, 'b::.import("{P}b")\nb.inc(1)', files={"a": 'f(n)::{n+1}\n.export(:{;"f";f})', "b": 'a::.import("{R}a")\n.export(:{;"inc";a.f})'}, v="2")

C("l-in-a-module-loads-into-the-module", M, 'm::.import("{P}m")\nm.f(2)', files={"h": 'helper(n)::{n*3}', "m": '.l("{P}h")\nf(n)::{helper(n)}\n.export(:{;"f";f})'}, v="6", why="decision: .l loads into the scope of the file that contains the call")
C("l-in-a-module-does-not-leak-strict2", M + " strict2", 'm::.import("{P}m")\nhelper(1)', files={"h": 'helper(n)::{n*3}', "m": '.l("{P}h")\nf(n)::{helper(n)}\n.export(:{;"f";f})'}, err="name")
C("l-in-main-does-not-reach-a-module", M, '.l("{P}c")\nm::.import("{P}m")\nm.f([2 4])', files={"c": 'mean(v)::{99}', "m": 'mean(v)::{+/v}\nf(v)::{mean(v)}\n.export(:{;"f";f})'}, v="6")
C("classic-module-block-inside-an-imported-file", M, 'm::.import("{P}m")\nm.scale(4)', files={"m": '.module(:geo)\nunit::10\nscale(n)::{n*unit}\n.module(0)\n.export(:{;"scale";scale})'}, v="40")
C("classic-module-block-in-main-is-unchanged", Q, '.module(:geo)\nunit::10\nscale(n)::{n*unit}\n.module(0)\nscale(4)', v="40")

C("error-in-the-body", M, '.import("{P}m")', files={"m": '1+"a"\n.export(:{;"a";1})'}, err="type")
C("error-in-an-exported-function", M, 'm::.import("{P}m")\nm.f(0)', files={"m": 'f(n)::{n+"a"}\n.export(:{;"f";f})'}, err="type")
C("strict2-reaches-the-module-text", M + " strict2", '.import("{P}m")', files={"m": 'a::b+1\n.export(:{;"a";a})'}, err="name", why="a module is loaded at the strict level of the importer")

C("bundled-csv-through-import", M, 'csv::.import("csv")\ncsv.split("a,b,c")', v='["a" "b" "c"]', why="klongpy/lib/csv.kg converted to a module that exports split read make write load")
C("bundled-csv-roundtrip", M, 'csv::.import("csv")\n#csv.split(csv.make(["a" "b"]))', v="2")
C("bundled-csv-keeps-its-helpers-private", M, 'csv::.import("csv")\n:_(csv@"unesc")', v="1", why="unesc, escape, unquote and ctlm are helpers")

C("static-declarations-are-inert", M + " static float", 'm::.import("{P}m")\nm.run(4.0)', files={"m": 'run(v)::{v*2.0}\n.entry("run";"float -> float")\n.export(:{;"run";run})'}, v="8.0", why="the declarations run as they stand; kgpy compile c must see through .import (DESIGN-GAPS)")

C("export-is-a-snapshot", M, 'm::.import("{P}m")\nm.inc()\nm.n', files={"m": 'n::0\ninc()::{n::n+1;n}\n.export(:{;"n";n;"inc";inc})'}, v="0", why="DECISION: .export copies values at the call. Exporting a variable exports its value then, not the variable; state is reached through functions")
C("exported-function-keeps-its-module-scope", M, 'm::.import("{P}m")\ng::m.inc\ng()+g()', files={"m": 'n::0\ninc()::{n::n+1;n}\n.export(:{;"inc";inc})'}, v="3", why="a function value taken out of a module still runs in that module: 1+2")
C("export-order-statements-after-export-still-run", M, 'm::.import("{P}m")\nm.f(1)', files={"m": 'f(n)::{n+k}\n.export(:{;"f";f})\nk::10'}, v="11", why="DECISION: .export fixes the table, not the end of the file; a lint warns when it is not the last statement")

# ---- R6: selection. `@` is dictionary indexing (classic); the two-argument form is loud
SEL = {"m": 'f(n)::{n+1}\ng(n)::{n*10}\n.export(:{;"f";f;"g";g})'}
VAL = {"m": '.export(:{;"v";3;"w";4})'}
C("select-by-at", M, '[f g]::.import("{P}m")@["f" "g"]\nf(1)+g(1)', files=SEL, v="12", why="@ with a list of keys is dictionary indexing (principle 3)")
C("select-by-at-rename", M, '[inc dbl]::.import("{P}m")@["g" "f"]\ninc(1)', files=SEL, v="10", why="the names on the left are the importer's own")
C("select-by-at-strict2", M + " strict2", '[f g]::.import("{P}m")@["f" "g"]\nf(1)+g(1)', files=SEL, v="12")
C("select-by-at-leaves-the-rest-private", M + " strict2", '[f]::.import("{P}m")@["f"]\ng', files=SEL, err="name", why="only what was selected is bound in the importer (principle 4)")
C("at-staying-classic-a-missing-name-is-undefined", M, '[f zz]::.import("{P}m")@["f" "zz"]\n:_(zz)', files={"m": 'f(n)::{n+1}\n.export(:{;"f";f})'}, v="1", why="pin: @ keeps its classic answer :undefined for a missing key (a.b and the two-argument form are the loud ones)")
C("import-then-at", M, 'g::(.import("{P}m"))@"f"\ng(1)', files=SEL, v="2")
C("select-two-arguments", M, '[f g]::.import("{P}m";["f" "g"])\nf(1)+g(1)', files=SEL, v="12", why="R6: the second argument lists the keys; the result is the list of their values, in that order")
C("select-two-arguments-order", M, '[a b]::.import("{P}m";["g" "f"])\na(1)', files=SEL, v="10", why="the order of the keys, not of the export")
C("select-two-arguments-rename", M, '[inc dbl]::.import("{P}m";["f" "g"])\ninc(1)+dbl(1)', files=SEL, v="12")
C("select-two-arguments-values", M, '.import("{P}m";["w" "v"])', files=VAL, v="[4 3]")
C("select-two-arguments-count", M, '#.import("{P}m";["w" "v" "w"])', files=VAL, v="3", why="a key may be listed twice")
C("select-two-arguments-empty-list", M, '#.import("{P}m";[])', files=VAL, v="0")
C("select-two-arguments-missing-key", M, '.import("{P}m";["v" "zz"])', files=VAL, err="name", why="R6: fails, naming the missing key and the file (principle 5)")
C("select-two-arguments-private-name", M, '.import("{P}m";["h"])', files={"m": 'h::1\n.export(:{;"v";3})'}, err="name", why="a private name is a missing key")
C("select-two-arguments-leaves-the-rest-private", M + " strict2", '[f]::.import("{P}m";["f"])\ng', files=SEL, err="name")
C("select-two-arguments-not-a-list", M, '.import("{P}m";5)', files=VAL, err="type")
C("select-two-arguments-key-not-a-string", M, '.import("{P}m";["v" 1])', files=VAL, err="type")
C("select-two-arguments-three-arguments", M, '.import("{P}m";["v"];1)', files=VAL, err="arity")
C("select-two-arguments-needs-an-export", M, '.import("{P}m";["v"])', files={"m": 'v::1'}, err="system", why="the same rule as the one-argument form")
C("select-two-arguments-shares-the-module", M, '[i]::.import("{P}m";["inc"])\nj::.import("{P}m")\ni()\nj.inc()', files={"m": CNT}, v="2", why="the two-argument form uses the cache of the one-argument form")
C("select-two-arguments-is-not-a-dictionary", M, '.import("{P}m";["v"])@0', files=VAL, v="3", why="the result is a list of values")

# ---- R8: path resolution. ./ and ../ are relative to the importing file; a bare name searches the library path only
C("bare-name-does-not-search-the-importing-files-directory", M, '.import("{P}a")', files={"b": '.export(:{;"v";1})', "a": 'b::.import("{B}b")\n.export(:{;"v";b.v})'}, err="system", why="R8, principle 4: a bare name is not looked up beside the importing file; write ./b")
C("bare-name-does-not-search-the-working-directory", M, '.import("{B}m")', files={"m": '.export(:{;"v";1})'}, err="system", why="R8, principle 4: the working directory is a hidden input; the file is in /tmp and the bare name is not found whatever the directory the runner started in")
C("relative-path-to-a-missing-file", M, '.import("{P}a")', files={"a": 'b::.import("./kgm87-nosuch-file-anywhere")\n.export(:{;"v";1})'}, err="system")
C("missing-file", M, '.import("/tmp/kgm87-no-such-file-anywhere")', err="system")
C("missing-file-two-arguments", M, '.import("/tmp/kgm87-no-such-file-anywhere";["f"])', err="system")

# ---- R9 (revised): .l of a file that calls .export binds the module dictionary to the file's name, as name::.import("name") does,
# and defines no plain name
LM = {"m": 'h(n)::{n*2}\nf(n)::{h(n)+1}\n.export(:{;"f";f})'}
C("l-of-a-module-binds-the-file-name", M, '.l("{A}m")\n{N}m.f(2)', files=LM, v="5", al=True, why="R9: .l(\"csv\") then csv.read(...) keeps working (principle 7); the file's name is the alias")
C("l-of-a-module-defines-no-plain-name", M + " strict2", '.l("{A}m")\nf(1)', files=LM, err="name", al=True, why="R9: nothing but the alias is defined (principle 4)")
C("l-of-a-module-keeps-the-private-names-private", M, '.l("{A}m")\n{N}m.h(1)', files=LM, err="name", al=True)
C("l-of-a-module-binds-a-read-only-dictionary", M, '.l("{A}m")\n{N}m.f::{x}', files=LM, err="type", al=True)
C("l-of-a-module-binds-the-keys", M, '.l("{A}m")\n{N}m.api(1)', files={"m": 'impl(n)::{n+10}\n.export(:{;"api";impl})'}, v="11", al=True)
C("l-of-a-module-binds-a-constant", M, '.l("{A}m")\n{N}m.version', files={"m": '.export(:{;"version";3})'}, v="3", al=True)
C("l-then-import-share-the-module", M, '.l("{A}m")\nm::.import("{A}m")\n{N}m.inc()\nm.inc()', files={"m": CNT}, v="2", al=True, why="the module is cached by real path")
C("import-then-l-share-the-module", M, 'm::.import("{A}m")\n.l("{A}m")\nm.inc()\n{N}m.inc()', files={"m": CNT}, v="2", al=True)
C("l-twice-is-the-same-module", M, '.l("{A}m")\n.l("{A}m")\n{N}m.inc()\n{N}m.inc()', files={"m": CNT}, v="2", al=True, why="the second .l does not run the body again")
C("l-of-a-module-the-importers-global-is-not-seen", M, 'n::50\n.l("{A}m")\n{N}m.inc()', files={"m": CNT}, v="1", al=True)
C("l-of-a-module-replaces-a-global-of-the-same-name", M, '{N}m::7\n.l("{A}m")\n{N}m.f(2)', files=LM, v="5", al=True)
C("l-of-a-module-without-the-extension", M, '.l("{A}m.kg")\n{N}m.f(2)', files=LM, v="5", al=True)
C("l-of-a-module-error-in-the-body", M, '.l("{P}m")', files={"m": '1+"a"\n.export(:{;"a";1})'}, err="type")
C("l-of-a-module-export-twice", M, '.l("{P}m")', files={"m": '.export(:{;"a";1})\n.export(:{;"b";2})'}, err="system")
C("l-of-a-module-from-a-module", M, 'a::.import("{A}a")\na.g(2)', files={"m": LM["m"], "a": '.l("{A}m")\ng(n)::{{N}m.f(n)}\n.export(:{;"g";g})'}, v="5", al=True, why="R9 inside a module: the alias is bound in the loading module's scope")
# ---- the cache holds modules weakly: a module that no value refers to is freed, and a later import runs the file again
LOADED = {"m": '.p("loaded")\n.export(:{;"v";1})'}
C("a-module-nothing-refers-to-is-freed", M, 'a::.import("{P}m")\na::0\nb::.import("{P}m")\nb.v', files=LOADED, v="1", out="loaded\nloaded", why="principle 4: nothing is held that no value refers to; a library released with m::0 gives its memory back")
C("a-freed-module-starts-again", M, 'a::.import("{P}m")\na.inc()\na::0\nb::.import("{P}m")\nb.inc()', files={"m": CNT}, v="1", why="the state went with the module")
C("a-module-with-a-reference-is-kept", M, 'a::.import("{P}m")\nb::.import("{P}m")\na::0\nc::.import("{P}m")\nc.v', files=LOADED, v="1", out="loaded", why="while any value refers to it, every importer gets the same instance")
C("a-function-of-a-module-keeps-it", M, 'a::.import("{P}m")\ng::a.inc\na::0\nb::.import("{P}m")\ng()\nb.inc()', files={"m": CNT}, v="2", why="a function taken out of a module is a reference to it")
C("a-selected-function-keeps-the-module", M, '[i]::.import("{P}m")@["inc"]\nj::.import("{P}m")\ni()\nj.inc()', files={"m": CNT}, v="2")
C("a-module-imported-by_a_live_module-is-kept", M, 'a::.import("{P}a")\nb::.import("{P}c")\nb.bump()\na.bump()', files=DIA, v="2", out="c", why="the diamond: c is referred to by the live module a, and by b")
C("a-module-freed-with-its-importer", M, 'a::.import("{P}a")\na::0\nb::.import("{P}c")\nb.bump()', files=DIA, v="1", out="c\nc", why="a refers to c; once a is freed nothing does")
C("l-of-a-classic-file-in-a-module-file-is-unchanged", M, 'a::.import("{P}a")\na.g(2)', files={"c": 'f(n)::{n*3}', "a": '.l("{P}c")\ng(n)::{f(n)}\n.export(:{;"g";g})'}, v="6")
C("l-of-a-classic-module-block-is-unchanged", M, '.l("{P}c")\nscale(4)', files={"c": '.module(:geo)\nunit::10\nscale(n)::{n*unit}\n.module(0)'}, v="40", why="principle 7: classic .module blocks through .l")
C("l-of-a-classic-module-block-keeps-the-name-reachable", M, '.l("{P}c")\nunit', files={"c": '.module(:geo)\nunit::10\nscale(n)::{n*unit}\n.module(0)'}, v="10")
C("bundled-csv-with-l-binds-csv", M, '.l("csv")\ncsv.split("a,b,c")', v='["a" "b" "c"]', why="R9: .l(\"csv\") then csv.split(...) keeps working (principle 7)")
C("bundled-csv-with-l-defines-no-plain-name", M + " strict2", '.l("csv")\nsplit("a")', err="name", why="principle 4: only the alias csv is defined")
C("bundled-csv-with-l-keeps-its-helpers-private", M, '.l("csv")\ncsv.unesc("a")', err="name")
C("bundled-csv-with-l-and-import-share", M, '.l("csv")\nc::.import("csv")\n#c.split(csv.make(["a" "b"]))', v="2")
C("bundled-nstat-with-l-binds-nstat", M + " float", '.l("nstat")\nnstat.mu([1 2 3 4])', v="2.5", why="nstat is a module: .l(\"nstat\") binds the one name nstat and its functions are entries of it")
C("bundled-nstat-with-l-defines-no-plain-name", M + " strict2", '.l("nstat")\nmu([1 2])', err="name", why="principle 4: only the alias nstat is defined")
C("bundled-nstat-with-l-leaves-the-family-names-alone", M, 'n::5\nt::6\nf::7\n.l("nstat")\n(n+t)+f', v="18", why="the distribution families n t f u r b x2 F ... are entries of nstat, not variables of the program")
C("bundled-nstat-family-is-a-dictionary-in-the-module", M + " float", '.l("nstat")\nnstat.n.cdf(0;0;1)', v="0.5", why="a family is nstat.n.cdf(x;m;s)")
C("bundled-nstat-classic-module-block-sees-the-modules-names", M + " float", '.l("nstat")\n#nstat.spl(2;[[1 1][2 4][3 9]])', v="3", why="spline.kg is a classic .module block loaded by the module; its free names (gauss, from math) are the module's own")

# ---- R2/R7: a name that starts with a dot is one name, also in a module file; it is a system name
C("dot-name-in-a-module-file-is-a-system-name", M, 'm::.import("{P}m")\n.hk.f(1)', files={"m": '.hk.f(n)::{n+1}\n.export(:{;"a";1})'}, v="2", why="a name that starts with a dot is one name for the whole program: the server looks hooks up by it (design point 4, principle 7)")
C("dot-name-from-a-module-is-visible-to-another-module", M, 'b::.import("{P}b")\nb.f()', files={"a": '.hk.g()::{7}\n.export(:{;"v";1})', "b": 'a::.import("{R}a")\nf()::{.hk.g()}\n.export(:{;"f";f})'}, v="7")
C("dot-name-set-in-the-importer-is-visible-in-a-module", M, '.hk.n::5\nm::.import("{P}m")\nm.f()', files={"m": 'f()::{.hk.n}\n.export(:{;"f";f})'}, v="5", why="system names are visible from every module (design point 7)")
C("dot-name-variable-written-by-a-module-function", M, 'm::.import("{P}m")\nm.set(9)\n.hk.n', files={"m": '.hk.n::0\nset(v)::{.hk.n::v}\n.export(:{;"set";set})'}, v="9")
C("module-function-sees-system-functions", M, 'm::.import("{P}m")\nm.f(5)', files={"m": 'f(n)::{.fmt("{}";n)}\n.export(:{;"f";f})'}, v='"5"', why="design point 7: locals, then the module's globals, then system names")
C("module-body-sees-system-functions", M, '.import("{P}m")@"v"', files={"m": 'v::.fmt("{}";3)\n.export(:{;"v";v})'}, v='"3"')

# ---- shallow read-only: every spelling of a change that the language has
C("read-only-pair-join-list", M, 'm::.import("{P}m")\nm,["zz" 1]', files=RO, err="type", why="a read-only dictionary refuses , in every form, since , mutates its left side")
C("read-only-join-dictionary", M, 'm::.import("{P}m")\nm,:{;"zz";1}', files=RO, err="type")
C("read-only-join-replaces-an-entry", M, 'm::.import("{P}m")\nm,["f" 1]', files=RO, err="type")
C("read-only-drop-a-key", M, 'm::.import("{P}m")\n"f"_m', files=RO, err="type", why="_ drops in place today")
C("read-only-copy-by-join-into-empty", M, 'm::.import("{P}m")\ne:::{}\ne,m\ne.f(7)', files=RO, v="8", why="joining into a writable dictionary copies the entries (M11)")
C("read-only-nested-module-is-read-only", M, 'b::.import("{P}b")\nb.base.f::{x}', files={"a": 'f(n)::{n+1}\n.export(:{;"f";f})', "b": 'a::.import("{R}a")\n.export(:{;"base";a})'}, err="type", why="a module stored in another export is still the module's own read-only dictionary")
C("read-only-key-by-at-assignment-does-nothing", M, 'm::.import("{P}m")\nm@"f"::7\nm.f(1)', files=RO, v="2", why="d@\"a\"::5 does nothing today (a call target is not assignable)")

# ---- .export checks
C("export-key-not-a-string-symbol", M, '.import("{P}m")', files={"m": '.export(:{;:a;1})'}, err="type")
C("export-no-arguments", M, '.import("{P}m")', files={"m": '.export()'}, err="arity", why="the torch backend's .export(fn;input;path) is renamed")
C("export-a-string", M, '.import("{P}m")', files={"m": '.export("a")'}, err="type")
C("export-three-arguments-is-the-old-torch-form", M, '.import("{P}m")', files={"m": '.export(1;2;3)'}, err="arity")
C("export-value-is-the-dictionary", M, '#.import("{P}m")', files={"m": '.export(:{;"a";1;"b";2;"c";3})'}, v="3")
C("export-does-not-return-the-private-names", M, 'm::.import("{P}m")\n:_(m@"b")', files={"m": 'b::2\n.export(:{;"a";1})'}, v="1")
C("export-from-a-function-in-the-module", M, 'm::.import("{P}m")\nm.a', files={"m": 'mk()::{.export(:{;"a";1})}\nmk()'}, v="1", why="the call of .export is what counts, wherever it runs while the body runs")
C("export-a-qualified-value", M, 'm::.import("{P}m")\nm.v', files={"m": 'd:::{;"k";4}\n.export(:{;"v";d.k})'}, v="4", why="a qualified name is an expression inside .export")

# ---- scope details
C("module-global-assigned-by-its-function", M, 'm::.import("{P}m")\nm.inc()\nm.inc()', files={"m": CNT}, v="2")
C("module-function-writes-a-new-global-in-its-module", M, 'm::.import("{P}m")\nm.set(4)\nm.get()', files={"m": 'set(v)::{k::v}\nget()::{k}\n.export(:{;"set";set;"get";get})'}, v=":k", why="level 0: a new name assigned in a function is local to the call; this pins the classic rule inside a module too")
C("module-function-writes-a-declared-global", M, 'm::.import("{P}m")\nm.set(4)\nm.get()', files={"m": 'k::0\nset(v)::{k::v}\nget()::{k}\n.export(:{;"set";set;"get";get})'}, v="4", why="an existing module global is assigned where it lives")
C("module-function-does-not-assign-the-importers-global", M, 'k::1\nm::.import("{P}m")\nm.set(4)\nk', files={"m": 'k::0\nset(v)::{k::v}\n.export(:{;"set";set})'}, v="1")
C("module-function-with-the-importers-local-of-the-same-name", M, 'm::.import("{P}m")\nf::{[k];k::5;m.get()}\nf()', files={"m": 'k::2\nget()::{k}\n.export(:{;"get";get})'}, v="2")
C("module-function-does-not-see-its-callers-module-frames", M, 'm::.import("{P}m")\nm.outer(3)', files={"m": 'inner()::{p}\nouter(p)::{inner()}\n.export(:{;"outer";outer})'}, v=":p", why="principle 4: a function with named parameters reads its own locals and its module's globals, never its caller's locals, in a module too; level 0 reads an unknown name as its symbol")
C("module-function-does-not-see-its-callers-module-frames-strict1", M + " strict1", 'm::.import("{P}m")\nm.outer(3)', files={"m": 'inner()::{p}\nouter(p)::{inner()}\n.export(:{;"outer";outer})'}, err="name", why="the same, loud at strict level 1")
C("classic-module-function-sees-its-callers-module-frames", M, 'm::.import("{P}m")\nm.outer(3)', files={"m": 'inner::{p}\nouter::{[p];p::3;inner()}\n.export(:{;"outer";outer})'}, v="3", why="D4: a function without a parameter list keeps classic dynamic scope between the functions of one module")
C("module-function-does-not-see-the-importers-frames", M + " strict2", 'm::.import("{P}m")\nf::{[p];p::3;m.get()}\nf()', files={"m": 'get()::{p}\n.export(:{;"get";get})'}, err="name", why="principle 4: the importer's locals are not the module's names")
C("module-top-level-names-do-not-reach-the-next-module", M + " strict2", 'a::.import("{P}a")\nb::.import("{P}b")\nb.f()', files={"a": 'shared::1\n.export(:{;"v";1})', "b": 'f()::{shared}\n.export(:{;"f";f})'}, err="name")
C("module-inside-a-function-of-a-module", M, 'a::.import("{P}a")\na.f()', files={"b": 'v::7\nget::{v}\n.export(:{;"get";get})', "a": 'f()::{[q];q::.import("{R}b");q.get()}\n.export(:{;"f";f})'}, v="7", why="lazy import in a module function: the alias is local, the module is the same instance")
C("an-exported-lambda-keeps-its-module", M, 'm::.import("{P}m")\nm.get()', files={"m": 'v::7\nget::{v}\n.export(:{;"get";get})'}, v="7", why="a function literal in .export closes over its module's scope")
C("an-exported-projection-keeps-its-module", M, 'm::.import("{P}m")\nm.add1(1)', files={"m": 'k::10\nadd(a;b)::{a+b+k}\n.export(:{;"add1";add(;1)})'}, v="12")
C("projection-of-a-qualified-function-with-module-global", M, 'm::.import("{P}m")\np::m.add(;1)\np(2)', files={"m": 'k::10\nadd(a;b)::{a+b+k}\n.export(:{;"add";add})'}, v="13")
C("adverb-over-a-module-function-with-module-global", M, 'm::.import("{P}m")\nm.f\'[1 2]', files={"m": 'k::10\nf(a)::{a+k}\n.export(:{;"f";f})'}, v="[11 12]")
C("module-function-called-from-a-callback-of-the-importer", M, 'm::.import("{P}m")\nm.f\'[1 2]\n{m.f(x)}\'[1 2]', files={"m": 'k::10\nf(a)::{a+k}\n.export(:{;"f";f})'}, v="[11 12]")
C("dictionary-entry-call-with-module-function", M, 'm::.import("{P}m")\nd:::{;"g";m.f}\nd.g(1)', files={"m": 'k::10\nf(a)::{a+k}\n.export(:{;"f";f})'}, v="11")
C("module-state-through-two-aliases", M, 'a::.import("{P}m")\nd:::{;"x";a}\nd.x.inc()\na.inc()', files={"m": CNT}, v="2")
C("two-modules-same-exports-called-through-a-dictionary", M, 'r:::{;"p";.import("{P}a");"q";.import("{P}b")}\nr.p.g()+r.q.g()', files={"a": 'g()::{1}\n.export(:{;"g";g})', "b": 'g()::{2}\n.export(:{;"g";g})'}, v="3")
C("module-held-in-a-global-called-x-works-at-top-level", M, 'x::.import("{P}m")\nx.v', files={"m": '.export(:{;"v";5})'}, v="5")

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

# ---- static: dotted .extern (R10)
S = "strings dicts qualified static"
C("extern-dotted-declares-a-dictionary", S, '.extern("read.adc";"int -> float")\n#read', v="1", why="R10: .extern(\"read.adc\") declares the entry adc of a dictionary read (principle 7: the existing programs keep their names)")
C("extern-dotted-share-one-dictionary", S, '.extern("read.adc";"int -> float")\n.extern("read.dac";"float -> int")\n#read', v="2")
C("extern-dotted-call-without-a-host-function", S, '.extern("read.adc";"int -> float")\nread.adc(0)', err="system", why="no host function: the same error as before")
C("extern-dotted-is-not-a-variable-named-with-a-dot", S, '.extern("read.adc";"int -> float")\n(read@"adc")~read.adc', v="1")
C("extern-dotted-missing-entry", S, '.extern("read.adc";"int -> float")\nread.dac', err="name")
C("extern-dotted-first-part-not-a-dictionary", S, 'read::5\n.extern("read.adc";"int -> float")', err="type", why="R10 detail: the first part names a dictionary; a name that holds something else is an error (principle 5)")

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
out.append("# Modules: .import, .export and qualified names a.b. Generated by conformance/tools/gen_modules.py: edit the generator, not this file.")
out.append("# docs/design-principles.md decides every open question; a `why` line names the principle.")
out.append("#")
out.append("# Tags: qualified = a.b on dictionaries (reading, writing, calling); modules = .import and .export")
out.append("# (a case writes its module files into /tmp with .oc .tc .p .cc and deletes them with .df, hence strings dicts sysio).")
out.append("# Macros in the comments: {P} is the absolute path, {R} the ./ path, {U} the ../tmp/ path and {B} the bare name of the case's own files.")
out.append("# A case that fails leaves /tmp/kgm87-* files behind (the runner stops at the first error). File names are unique per case.")
out.append("# Error kinds: a missing key is `name`; a base that is not a dictionary and writing to a read-only dictionary are `type`;")
out.append("# every failure of the module graph (missing file, cycle, no .export, a bare name not on the library path) is `system`.")
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
