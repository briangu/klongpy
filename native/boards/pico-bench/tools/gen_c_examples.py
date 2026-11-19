#!/usr/bin/env python3
"""Generate what the firmware needs of the compiled examples: the C of examples/embedded/{mavg,pid,fir}.kg, their Klong text, and the
reference results the native host gives for the fixed inputs of the board check.

    gen_c_examples.py OUT_DIR

writes into OUT_DIR (the build directory; nothing here is checked in):

    mavg.c mavg.h  pidc.c pidc.h  fir.c fir.h  klong_rt.h    the static-profile compiler's C, as `klong compile c` writes it (32-bit
                                                             kg_int, 32-bit kg_float: a Cortex-M0+ has neither a 64-bit divide
                                                             nor an FPU)
    klong_c_data.c klong_c_data.h                            the three sources as C strings (they are loaded by the interpreter
                                                             on the board, from flash, without a copy), the fixed inputs of the
                                                             check, the host's results for them and the values pinned by the
                                                             .expect files

It needs the `klong` binary (`cd native && cargo build --release`; or its path in $KLONG) and the extension of native/klong-py (its
directory in $KLONGPY_NATIVE_PATH, or klong_native importable) for the expected values, and NumPy, which the extension needs. KlongPy
itself is not imported. The tool fails, and so the build does, when the host no longer gives the values the .expect files pin, so the
board is never checked against a stale reference.
"""
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EMBEDDED = os.path.join(ROOT, "examples", "embedded")

# The board runs each example on these inputs. The first ones are the calls the .expect file pins.
PID_TAIL = [0.31, 0.5, 0.3, 0.7, 0.2]


def klist(xs):
    return "[" + " ".join(repr(float(x)) for x in xs) + "]"


def expect_lines(name):
    with open(os.path.join(EMBEDDED, name + ".expect")) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#")]


def mavg_case():
    pins = [(float(m.group(1)), float(m.group(2))) for m in (re.match(r"step\((.*)\) => (.*)$", ln) for ln in expect_lines("mavg")) if m]
    assert len(pins) == 5, pins
    inputs = [4.0 * (i + 1) for i in range(64)]
    assert [p[0] for p in pins] == inputs[:5], "the first inputs are the calls of mavg.expect"
    return dict(name="mavg", prefix="mavg", entry="step", inputs=inputs, setup=".c.in::" + klist(inputs), expr="step'.c.in", prelude="",
                cleanup="buf::0;pos::0;step::0", pins=[(i, p[1]) for i, p in enumerate(pins)], externs=None)


def pid_case():
    pins = [(float(m.group(1)), float(m.group(2))) for m in (re.match(r"step\((.*)\) => (.*)$", ln) for ln in expect_lines("pid")) if m]
    assert len(pins) == 3, pins
    inputs = [p[0] for p in pins] + [PID_TAIL[i % len(PID_TAIL)] for i in range(29)]
    # the externs of pid.expect: read.adc is 0.3 and write.pwm is 0 (the board registers C stand-ins for the interpreter that give the same as the compiled code's)
    return dict(name="pid", prefix="pidc", entry="step", inputs=inputs, setup=".c.in::" + klist(inputs), expr="step'.c.in",
                prelude="", cleanup="kp::0;ki::0;kd::0;dt::0;lo::0;hi::0;integ::0;prev::0;clamp::0;step::0",
                pins=[(i, p[1]) for i, p in enumerate(pins)], externs={"read.adc": lambda channel: 0.3, "write.pwm": lambda channel, duty: 0})


def fir_case():
    pins = []
    for ln in expect_lines("fir"):
        m = re.match(r"\*run\(0\.5\*!256\) => (.*)$", ln)
        if m:
            pins.append((0, float(m.group(1))))
        m = re.match(r"run\(0\.5\*!256\)@(\d+) => (.*)$", ln)
        if m:
            pins.append((int(m.group(1)), float(m.group(2))))
    assert len(pins) == 2, pins
    return dict(name="fir", prefix="fir", entry="run", inputs=[0.5 * i for i in range(256)], setup=".c.in::0.5*!256", expr="run(.c.in)", prelude="",
                cleanup="raw::0;taps::0;idx::0;run::0", pins=pins, externs=None)


def cstr(text):
    """A C string literal, one source line to a line."""
    out = []
    for line in text.splitlines(keepends=True):
        s = ""
        for b in line.encode("utf-8"):
            c = chr(b)
            if c == "\\" or c == '"':
                s += "\\" + c
            elif c == "\n":
                s += "\\n"
            elif 32 <= b < 127 and c != "?":
                s += c
            else:
                s += "\\%03o" % b
        out.append('    "' + s + '"')
    return "\n".join(out) if out else '    ""'


def cdoubles(xs):
    return ", ".join(repr(float(x)) for x in xs)


def native_extension():
    """The extension module of native/klong-py: $KLONGPY_NATIVE_PATH (a directory that holds it), else klong_native on the path."""
    d = os.environ.get("KLONGPY_NATIVE_PATH")
    if d:
        sys.path.insert(0, d)
    try:
        import klong_native
    except ImportError:
        sys.exit("gen_c_examples: the extension klong_native is needed (build native/klong-py and put its directory on KLONGPY_NATIVE_PATH)")
    return klong_native


def klong_binary():
    """The `klong` binary: $KLONG, else native/target/release/klong."""
    path = os.environ.get("KLONG") or os.path.join(ROOT, "native", "target", "release", "klong")
    if not os.access(path, os.X_OK):
        sys.exit("gen_c_examples: %s is not an executable `klong` (cd native && cargo build --release, or set KLONG)" % path)
    return path


def reference(case):
    """The host's results for the case: the expression the board's interpreter runs, and the entry called point by point."""
    import numpy as np

    kn = native_extension()
    path = os.path.join(EMBEDDED, case["name"] + ".kg")
    with open(path) as f:
        text = f.read()
    externs = case["externs"] or {}
    klong = kn.Klong(2)
    for name, fn in externs.items():
        # the stand-ins are bound by the declaration that follows: .extern takes the function registered under its name
        klong.register(name, fn, fn.__code__.co_argcount)
    klong.run(text, path)
    klong.run(case["setup"])
    got = klong.eval(case["expr"])
    ref = [float(x) for x in np.asarray(got, dtype=float).ravel()]
    prog = kn.StaticProgram(path, text, externs)
    if case["name"] == "fir":
        again = [float(x) for x in np.asarray(prog.call_entry(case["entry"], [list(case["inputs"])]), dtype=float)]
    else:
        again = [float(prog.call_entry(case["entry"], [x])) for x in case["inputs"]]
    assert len(ref) == len(again) and all(abs(a - b) <= 1e-12 * max(1.0, abs(a)) for a, b in zip(ref, again)), "the expression and call_entry disagree"
    for i, v in case["pins"]:
        assert math.isclose(ref[i], v, rel_tol=1e-9, abs_tol=1e-12), "%s: the host gives %r at %d but %s.expect pins %r" % (case["name"], ref[i], i, case["name"], v)
    return ref


def emit(klong, kg, out, prefix):
    """`klong compile c` of one program with 32-bit ints and floats: the files in out; returns the size report."""
    r = subprocess.run([klong, "compile", "c", kg, "-o", out, "--prefix", prefix], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("gen_c_examples: klong compile c %s failed:\n%s%s" % (kg, r.stdout, r.stderr))
    # the first line says what was written, then the size report
    return r.stdout.split("\n", 1)[1].rstrip("\n") if "\n" in r.stdout else ""


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = os.path.abspath(argv[1])
    os.makedirs(out, exist_ok=True)
    klong = klong_binary()

    cases = [mavg_case(), pid_case(), fir_case()]
    for c in cases:
        c["report"] = emit(klong, os.path.join(EMBEDDED, c["name"] + ".kg"), out, c["prefix"])
        c["ref"] = reference(c)
        with open(os.path.join(EMBEDDED, c["name"] + ".kg")) as f:
            c["src"] = f.read()
        print(c["report"])
    with open(os.path.join(out, "klong_c_data.h"), "w") as f:
        f.write("""/* Generated by native/boards/pico-bench/tools/gen_c_examples.py from examples/embedded/{mavg,pid,fir}.kg and their .expect files. Do not edit. */
#ifndef KLONG_C_DATA_H
#define KLONG_C_DATA_H
#include <stddef.h>
#include <stdint.h>

typedef struct klong_c_example {
    const char *name;    /* mavg, pid, fir */
    const char *src;     /* the Klong source, byte for byte (in flash); src_len bytes */
    size_t src_len;
    const char *prelude; /* Klong the interpreter runs before the source (the stand-ins of the externs) */
    const char *setup;   /* Klong the interpreter runs after it, not timed: .c.in is the fixed input ... */
    const char *expr;    /* ... and this is the call, which is timed */
    const char *cleanup; /* Klong that lets go of what the source and the setup defined (the globals are set to 0) */
    size_t nin;
    const double *in;    /* the same inputs, for the compiled code */
    size_t nout;
    const double *ref;   /* what KlongPy answered to expr on the host, when the firmware was built */
    size_t npin;
    const int32_t *pin_idx; /* the results the .expect file pins: their positions in ref ... */
    const double *pin_val;  /* ... and their values */
} klong_c_example;

#define KLONG_C_NEXAMPLES 3
extern const klong_c_example klong_c_examples[KLONG_C_NEXAMPLES];

#endif
""")
    with open(os.path.join(out, "klong_c_data.c"), "w") as f:
        f.write("/* Generated by native/boards/pico-bench/tools/gen_c_examples.py from examples/embedded/{mavg,pid,fir}.kg and their .expect files. Do not edit. */\n")
        f.write('#include "klong_c_data.h"\n\n')
        for c in cases:
            n = c["name"]
            f.write("static const char src_%s[] =\n%s;\n" % (n, cstr(c["src"])))
            f.write("static const char setup_%s[] =\n%s;\n" % (n, cstr(c["setup"])))
            f.write("static const char expr_%s[] =\n%s;\n" % (n, cstr(c["expr"])))
            f.write("static const double in_%s[%d] = {%s};\n" % (n, len(c["inputs"]), cdoubles(c["inputs"])))
            f.write("static const double ref_%s[%d] = {%s};\n" % (n, len(c["ref"]), cdoubles(c["ref"])))
            f.write("static const int32_t pin_idx_%s[%d] = {%s};\n" % (n, len(c["pins"]), ", ".join(str(i) for i, _ in c["pins"])))
            f.write("static const double pin_val_%s[%d] = {%s};\n\n" % (n, len(c["pins"]), cdoubles([v for _, v in c["pins"]])))
        f.write("const klong_c_example klong_c_examples[KLONG_C_NEXAMPLES] = {\n")
        for c in cases:
            n = c["name"]
            f.write('    {"%s", src_%s, sizeof src_%s - 1, "%s", setup_%s, expr_%s, "%s", %d, in_%s, %d, ref_%s, %d, pin_idx_%s, pin_val_%s},\n' % (
                n, n, n, c["prelude"], n, n, c["cleanup"], len(c["inputs"]), n, len(c["ref"]), n, len(c["pins"]), n, n))
        f.write("};\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
