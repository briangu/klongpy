#!/usr/bin/env python3
"""
The Klong library and the reference Klong suites through the native binary (ticket 103).

    .venv/bin/python native/scripts/suites.py [--klong PATH] [--no-klongpy]

1. Every file of klongpy/lib is loaded with `.l` at strict levels 0, 1 and 2.
2. The library calls of tests/test_lib_strict2.py (CALLS) are evaluated at every level and
   compared with KlongPy's value (as printed; numbers within 1e-12 relative).
3. Every tests/kgtests/language/{test,gen}*.kg suite is run through tests/kgtests/runner.kg at
   every level: the number of `t(...)` lines and how many the runner reported as failed.

Exit status 0 when everything agrees. Run from anywhere; the binary runs in the repository root.
"""
import ast
import glob
import os
import re
import subprocess
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "klongpy", "lib")
LEVELS = (0, 1, 2)
# Suites that cannot run without Python: (file, why)
DESKTOP = {"test_reassign.kg": "uses .pyf (Python interop)"}


def klong_bin(argv):
    if "--klong" in argv:
        return argv[argv.index("--klong") + 1]
    return os.environ.get("KLONG_NATIVE") or os.path.join(ROOT, "native", "target", "release", "klong")


def run(binary, level, program, cwd=ROOT, timeout=300):
    p = subprocess.run([binary, "--strict", str(level), "-e", program], capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def calls():
    """CALLS of tests/test_lib_strict2.py, read without importing the test module."""
    with open(os.path.join(ROOT, "tests", "test_lib_strict2.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CALLS":
            return ast.literal_eval(node.value)
    return []


def numbers(text):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", text)]


def same(a, b):
    if a == b:
        return True
    na, nb = numbers(a), numbers(b)
    if len(na) != len(nb) or re.sub(r"-?\d+\.?\d*(?:e[-+]?\d+)?", "#", a) != re.sub(r"-?\d+\.?\d*(?:e[-+]?\d+)?", "#", b):
        return False
    return all(abs(x - y) <= 1e-12 * max(1.0, abs(x), abs(y)) for x, y in zip(na, nb))


def main(argv):
    binary = klong_bin(argv)
    bad = 0

    print("== library files: .l at strict levels 0, 1, 2")
    for path in sorted(glob.glob(os.path.join(LIB, "*.kg"))):
        name = os.path.basename(path)
        res = []
        for level in LEVELS:
            code, out, err = run(binary, level, f'.helpdb::[];.l("{name}");"loaded"', cwd=LIB)
            ok = code == 0 and out.strip().endswith('"loaded"')
            res.append("loads" if ok else "FAILS: " + (err.strip().splitlines() or ["?"])[0])
            bad += not ok
        print(f"  {name:12} " + ("loads at every level" if all(r == "loads" for r in res) else "; ".join(res)))

    print("== library calls against KlongPy (tests/test_lib_strict2.py CALLS)")
    ref = None
    if "--no-klongpy" not in argv:
        sys.path.insert(0, ROOT)
        warnings.simplefilter("ignore")
        from klongpy import KlongInterpreter
        from klongpy.core import kg_write

        def ref(fname, text):
            import contextlib
            import io
            k = KlongInterpreter()
            k[".helpdb"] = []
            cwd = os.getcwd()
            os.chdir(LIB)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    k(f'.l("{fname}")')
                    return kg_write(k(text), k._backend)
            finally:
                os.chdir(cwd)
    agree = total = 0
    for fname, text in calls():
        got = []
        for level in LEVELS:
            code, out, err = run(binary, level, f'.helpdb::[];.l("{fname}");{text}', cwd=LIB)
            got.append(out.strip() if code == 0 else "error: " + (err.strip().splitlines() or ["?"])[0])
        want = ref(fname, text) if ref else got[0]
        total += 1
        if all(same(g, want) for g in got):
            agree += 1
        else:
            bad += 1
            print(f"  DIFF {fname} {text}: KlongPy {want!r}, native {got!r}")
    print(f"  {agree} of {total} calls agree at every level" + ("" if ref else " (levels compared with each other only)"))

    print("== reference suites (tests/kgtests/language) through runner.kg")
    print(f"  {'file':28} {'t lines':>8}  level 0 / 1 / 2: failed")
    for path in sorted(glob.glob(os.path.join(ROOT, "tests", "kgtests", "language", "*.kg"))):
        name = os.path.basename(path)
        if not name.startswith(("test", "gen")):
            continue
        with open(path, encoding="utf-8") as f:
            lines = sum(1 for line in f if line.startswith("t("))
        if name in DESKTOP:
            print(f"  {name:28} {lines:8d}  not run: {DESKTOP[name]}")
            continue
        cells = []
        rel = os.path.relpath(path, ROOT)
        for level in LEVELS:
            code, out, err = run(binary, level, f'fullpath::"{rel}";.l("tests/kgtests/runner.kg");err')
            failed = out.count("failed: ")
            if code != 0:
                cells.append("ERROR " + (err.strip().splitlines() or ["?"])[0][:80])
                bad += 1
            else:
                cells.append(str(failed))
                bad += failed != 0 or not out.strip().endswith("0")
        print(f"  {name:28} {lines:8d}  " + " / ".join(cells))
    print("all agree" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
