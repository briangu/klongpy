#!/usr/bin/env python3
"""
Random programs into the native binary under a heap limit and a step limit (ticket 102, step 9).
Every run must end with exit status 0 or 1 (a value or a Klong error): never a panic, an abort,
a stack overflow or a hang.

    python3 native/scripts/fuzz.py [--seed N] [--count N] [--klong PATH]

Three generators: token soup (any order of lexemes), mutations of the programs of the
conformance cases, and a small grammar of well-formed expressions over strings, chars,
dictionaries, modules and the file functions (ticket 103). The binary runs with --memfs: the
file functions see an empty file system in memory, never the real one. A cargo-fuzz target with
coverage feedback is still to do.
"""
import argparse
import glob
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEXEMES = (
    "x y z a b f g .f 0 1 2 3 10 -1 1.5 1e3 1e400 9223372036854775807 0cx 0c \"s\" \"\" :a :b :undefined "
    "+ - * % ! ^ & | < > = @ , # _ $ ~ ? :# :_ := :- :@ :$ :^ :+ :% :: ' / \\ :~ :* :' :\\ :/ \\~ \\* @' "
    "( ) [ ] { } ; :[ :| :{ :( \n [] {x} {x+y} {[a];a::x;a} f:: g:: a:: (;) (1;) .p( .strict( .mem() !100 !100000 "
    "[1 2 3] [[1 2] [3 4]] ∇ é :\" \"\"\" "
    # ticket 103: strings and chars, dictionaries, modules, files
    "\"abc\" \"héllo\" \"a\"\"b\" \"1.5\" \"-12\" \"é\" 0cé 0c0 0c\" :{} :{[1 2]} :{[\"k\" [1 2]]} :{;\"k\";1} :{[1.5 2]} d "
    "d:: d,d d,[1 d] d?\"k\" d@1 1_d {x}'d :{[1 :{[2 :{}]}]} 5$ -5$ 5.2$ 1e300$ 0:$ 1.0:$ :a:$ 0ca:$ \"\":$ "
    ".module(:m) .module(0) .l(\"f\") .l(\"math\") .ic(\"f\") .oc(\"f\") .ac(\"f\") .cc( .tc( .fc( .tc(0) .fc(0) "
    ".r() .rl() .rs( .mi() .fl() .df(\"f\") .w( .d( .E( .fmt(\"{} {:5.2}\"; .fmt(\"{1}{0:-4}\"; .pf( .ver() .rn() .pc() "
    ".cin .cout .cerr .e .a .strict(2) "
).split(" ")


def atom(rng):
    return rng.choice(["0", "1", "-3", "2.5", "1e10", "\"\"", "\"abc\"", "\"héllo wörld\"", "\"a,b,,c\"", "0cx", "0cé", ":s", "[]",
                       "[1 2 3]", "[\"a\" \"bc\"]", "[1 \"a\" 0cb :c [2.5]]", ":{}", ":{[1 2] [\"a\" \"b\"] [:k [1 2]]}", "d", "s", "(!5)",
                       ":undefined", "1000000", "-1"])


def expr(rng, depth=0):
    """A well-formed expression that leans on strings, dictionaries and the file functions."""
    if depth > 3 or rng.random() < 0.25:
        return atom(rng)
    e = lambda: expr(rng, depth + 1)
    k = rng.randrange(22)
    if k < 6:
        op = rng.choice(["+", "-", "*", "%", "!", "^", "&", "|", "<", ">", "=", "@", ",", "#", "_", "$", "~", "?", ":#", ":_",
                         ":=", ":-", ":@", ":$", ":^", ":+", ":%"])
        return f"({e()}){op}({e()})"
    if k < 10:
        return rng.choice(["#", "|", "*", "?", "=", "<", ">", "$", "~", "@", ",", "^", "&", "_", "-", "!", "%", ":#", ":_"]) + f"({e()})"
    if k < 13:
        adv = rng.choice(["'", "/", "\\", ":'", "\\~", ":~"])
        return rng.choice(["{x,y}", "{x}", ",", "{$x}", "{#x}", "{x?0ca}", "+", "{d,x}", "{.w(x)}"]) + adv + f"({e()})"
    if k < 15:
        return f":{{;{e()};{e()};{e()};{e()}}}"
    if k == 15:
        return f"d,({e()}),,{e()}"
    if k == 16:
        return rng.choice([f".fmt({e()};{e()})", f".fmt(\"{{}}|{{:6}}|{{:-6.2}}\";[1 \"a\" 2.5]),{e()}", f".rs({e()})", f".E({e()})", f".rs(${e()})"])
    if k in (17, 18, 19):
        # statement sequences: as they are at top level, inside a function call when nested
        if k == 17:
            f = rng.choice(["\"f\"", "\"g.kg\"", "\"\"", "(" + e() + ")"])
            seq = rng.choice([f".tc(.oc({f}));.w({e()});.p({e()});.cc(.tc(0))", f".fc(.ic({f}));s::.r();s::s,.rl();.mi();.cc(.fc(0));s",
                              f".l({f})", f".df({f})", f".tc(.ac({f}));.d({e()});.cc(.tc(0))", f".ic({f})", f".cc(.oc({f}))",
                              f".tc(.oc(\"g.kg\"));.p(\"s::s,1;t::{{x}}\");.p({e()});.cc(.tc(0));.l(\"g\")"])
        elif k == 18:
            seq = f".module(:m{rng.randrange(3)});s::{e()};g::{{s,x}};.module(0);g({e()})"
        else:
            seq = f"d::{e()};s::{e()};{e()}"
        return seq if depth == 0 else "{" + seq + "}()"
    if k == 20:
        return f"{rng.choice(['5', '-7', '0', '6.2', '-4.1', '100000000', '0.5'])}$({e()})"
    return f"({e()}):$({e()})"


def grammar(rng):
    return ";".join(["d:::{[1 2] [\"a\" \"b\"]}", "s::\"héllo\""] + [expr(rng) for _ in range(rng.randint(1, 4))])


def corpus():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "conformance", "cases", "*.cases"))):
        with open(path, encoding="utf-8") as f:
            prog = []
            for line in f:
                if line.startswith("> "):
                    prog.append(line[2:].rstrip("\n"))
                elif prog:
                    out.append("\n".join(prog))
                    prog = []
    return out


def soup(rng):
    return "".join(rng.choice(LEXEMES) + rng.choice(["", "", " "]) for _ in range(rng.randint(1, 40)))


def mutate(rng, text):
    for _ in range(rng.randint(1, 4)):
        i = rng.randint(0, len(text))
        op = rng.random()
        if op < 0.4:
            text = text[:i] + rng.choice(LEXEMES) + text[i:]
        elif op < 0.7:
            text = text[:i] + text[i + rng.randint(1, 3):]
        elif op < 0.85:
            j = rng.randint(0, len(text))
            text = text[:i] + text[min(i, j):max(i, j)] + text[i:]
        else:
            text = text[:i] + rng.choice("([{:\"'") * rng.randint(1, 300) + text[i:]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--klong", default=os.path.join(ROOT, "native", "target", "release", "klong"))
    ap.add_argument("--extra", default="", help="more options for the binary, e.g. '--stack 64K --stack-budget 64K'")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    base = corpus()
    def one():
        r = rng.random()
        if r < 0.3 or not base:
            return soup(rng)
        if r < 0.6:
            return mutate(rng, rng.choice(base))
        if r < 0.9:
            return grammar(rng)
        return mutate(rng, grammar(rng))

    programs = [one() for _ in range(a.count)]

    def run(src):
        strict = str(len(src) % 3)
        argv = [a.klong, "--memfs", "--heap-limit", "1M", "--steps", "200000", "--strict", strict, "--conformance"] + a.extra.split()
        try:
            p = subprocess.run(argv, input=src.encode("utf-8"), capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            return src, "timeout", ""
        return src, p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace")[-300:]

    bad = 0
    kinds = {}
    with ThreadPoolExecutor(os.cpu_count() or 2) as pool:
        for src, code, out in pool.map(run, programs):
            key = "value" if code == 0 else out.strip().splitlines()[0] if code == 1 and out.startswith("error: ") else f"BAD {code}"
            kinds[key] = kinds.get(key, 0) + 1
            if code not in (0, 1) or (code == 1 and "error: internal" in out):
                bad += 1
                if bad <= 10:
                    print(f"--- status {code}\n{src!r}\n{out}")
    for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"{n:6d}  {k}")
    print(f"{len(programs)} programs, seed {a.seed}: {bad} bad")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
