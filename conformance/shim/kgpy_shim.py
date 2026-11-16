#!/usr/bin/env python3
"""
KlongPy as an executable of the conformance protocol (conformance/README.md, "Runner protocol").

    kgpy_shim.py [--strict N]          program text on stdin
    kgpy_shim.py --dump-ast FILE       canonical parse tree of FILE

Exit status 0 and the printed result on stdout (nothing for a program with no value),
or exit status 1 and `error: <kind>` on stdout. What the program prints with .p .d .w goes
to stderr. A native runtime implements the same few lines; this file is the reference.
"""
import contextlib
import io
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate(source, strict=None):
    """Return (status, text, console): status "result" with the printed value, or "error" with the kind."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from klongpy import KlongInterpreter
    from klongpy.core import kg_write

    out = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(out):
        warnings.simplefilter("ignore")
        try:
            klong = KlongInterpreter()
            if strict:
                klong(f".strict({strict})")
            r = klong(source)
            text = "" if r is None else kg_write(r, klong._backend, display=False)
            return "result", text, out.getvalue()
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                return "error", "interrupt", out.getvalue()
            return "error", getattr(e, "kind", "internal"), out.getvalue()


def dump_ast(path):
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from klongpy import KlongInterpreter
    from klongpy.dump import dump, parse_program

    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sys.stdout.write(dump(parse_program(KlongInterpreter(parser="token"), text)))
            return 0
        except BaseException as e:
            print(f"error: {getattr(e, 'kind', 'internal')}")
            return 1


def main(argv):
    if argv[:1] == ["--dump-ast"] and len(argv) == 2:
        return dump_ast(argv[1])
    strict = None
    if argv[:1] == ["--strict"] and len(argv) >= 2:
        strict = int(argv[1])
    elif argv:
        print("usage: kgpy_shim.py [--strict N] | --dump-ast FILE", file=sys.stderr)
        return 2
    source = sys.stdin.buffer.read().decode("utf-8")
    status, text, console = evaluate(source, strict)
    sys.stderr.write(console)
    if status == "result":
        if text:
            sys.stdout.write(text + "\n")
        return 0
    sys.stdout.write(f"error: {text}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
