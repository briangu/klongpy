#!/usr/bin/env python3
"""
Random programs into the native binary under a heap limit and a step limit (ticket 102, step 9).
Every run must end with exit status 0 or 1 (a value or a Klong error): never a panic, an abort,
a stack overflow or a hang.

    python3 native/scripts/fuzz.py [--seed N] [--count N] [--klong PATH]

Two generators: token soup (any order of lexemes) and mutations of the programs of the
conformance cases. A cargo-fuzz target with coverage feedback is ticket 103's.
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
).split(" ")


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
    a = ap.parse_args()
    rng = random.Random(a.seed)
    base = corpus()
    programs = [soup(rng) if rng.random() < 0.5 or not base else mutate(rng, rng.choice(base)) for _ in range(a.count)]

    def run(src):
        strict = str(len(src) % 3)
        argv = [a.klong, "--heap-limit", "1M", "--steps", "200000", "--strict", strict, "--conformance"]
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
