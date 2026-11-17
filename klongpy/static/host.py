"""
Running a static program under KlongPy (the host half of .extern and .entry; what the
differential harness builds on).

    klong = load_program("examples/embedded/pid.kg",
                         externs={"read.adc": lambda ch: 0.3, "write.pwm": lambda ch, duty: 0})
    call_entry(klong, "step", 0.31)        # 0.12005

load_program runs the file at strict level 2 with the Python functions bound
as its externs; call_entry calls an entry the way the C caller would, with
the arguments and the result checked against the ``.entry`` signature.
"""
import os
import re

import numpy as np

from ..errors import KlongNameError, KlongRangeError, KlongShapeError, KlongSystemError, KlongTypeError
from ..lexer import Source
from .signature import Mismatch, conform, is_fixed

__all__ = ["load_program", "call_entry", "set_param", "uses_fixed"]

_FIXED_USE = re.compile(r"fixed<|\.fixed\(")


def uses_fixed(klong):
    """Does the loaded program use fixed-point numbers? Such a program runs through call_entry on the typed-IR evaluator."""
    src = getattr(klong, "_static_source", None)
    return bool(src and _FIXED_USE.search(src[1]))


def load_program(path, externs=None, *, text=None, strict=2, klong=None, backend=None):
    """
    Run a program and return the interpreter that holds its state.

        path      the file to run (or, with text=, the name the text is known by)
        externs   {name: callable}: the stand-ins, bound before the program runs
        text      the program text, instead of reading path
        strict    the strict level (2: what the static profile requires)
        klong     an existing KlongInterpreter to run in, instead of a new one
        backend   backend name for a new interpreter

    A program may run on any backend ("numpy", "torch"). What runs inside is the
    backend's own (torch computes in float32, so results can differ from the
    compiled program's double arithmetic by about 1e-7); what crosses the
    boundary is numpy: a stand-in is called with Python numbers and numpy
    int64/float64 arrays, and call_entry returns the same, and both are checked
    against the signature in that form. The checker never depends on the backend.

    Top-level definitions are evaluated in order, exactly as kgpy does;
    ``.extern`` binds each declared name to its stand-in and ``.entry``
    validates each signature. Errors are KlongError, positioned in the file.
    """
    from ..interpreter import KlongInterpreter
    if klong is None:
        klong = KlongInterpreter(backend=backend, strict=strict)
    if externs:
        klong.externs.update(externs)
    if text is None:
        with open(path, "r") as f:
            text = f.read()
    klong._static_source = (os.fspath(path), text, dict(externs or {}))
    klong._static_eval = None
    klong(Source(os.fspath(path), text))
    return klong


def set_param(klong, name, value):
    """
    Set the parameter ``name`` (declared with ``.param``) between runs: the host side of the register that the
    C emitter exposes as ``<prog>_set_<name>`` and the hardware as an s_axilite argument. The value is checked
    against the declared type and converted ("int" takes an int, "float" an int or a float, "fixed<W,I>" an int,
    a float or a fixed-point value, truncating toward minus infinity).
    """
    decl = klong.externs.params.get(name)
    if decl is None:
        raise KlongNameError(f"{name} is not a parameter of the program", hint=f'declare it: .param("{name}";"int")')
    try:
        v = conform(decl.ty, value, f"the value of parameter {name}")
    except Mismatch as m:
        raise (KlongShapeError if m.kind == "shape" else KlongTypeError)(m.message) from None
    klong[name] = v if is_fixed(decl.ty.elem) else (int(v) if decl.ty.elem == "int" else float(v))
    return v


def _evaluator(klong):
    """The typed-IR evaluator of a program that uses fixed-point numbers (compiled on first use)."""
    ev = getattr(klong, "_static_eval", None)
    if ev is None:
        from .compile import compile_program
        from .eval_ir import IrEvaluator
        path, text, externs = klong._static_source
        program = compile_program(path, text=text, externs=externs).program
        ev = klong._static_eval = IrEvaluator(program)
    return ev


def _host_state(ty, v):
    from . import fixed as fx
    if is_fixed(ty.elem):
        return fx.FixedArray(v, fx.spec_of(ty.elem))
    return v.item() if not ty.shape else np.array(v)


def _raw(ty, v):
    return np.asarray(v.raw if hasattr(v, "raw") else v, dtype=np.float64 if ty.elem == "float" else np.int64)


def _call_evaluated(klong, name, sig, values, raw):
    """Run an entry of a fixed-point program on the evaluator; the state globals live in klong between calls."""
    from . import fixed as fx
    from .eval_ir import Trap
    ev = _evaluator(klong)
    entry = next((e for e in ev.program.entries if e.decl.name == name and str(e.decl.signature) == str(sig)), None)
    if entry is None:
        raise KlongNameError(f"{name} with the signature \"{sig}\" is not an entry of the compiled program")
    for g in ev.program.globals:
        if g.state:
            ev.state[g.name] = np.array(_raw(g.ty, klong[g.name]))
    try:
        r = ev.run(entry, [_raw(t, v) for t, v in zip(sig.params, values)])
    except Trap as t:
        raise (KlongRangeError if t.code == 1 else KlongSystemError)(f"entry {name}: {t}") from None
    finally:
        for g in ev.program.globals:
            if g.state:
                klong[g.name] = _host_state(g.ty, ev.state[g.name])
    if sig.result.elem == "void":
        return None
    if is_fixed(sig.result.elem):
        spec = fx.spec_of(sig.result.elem)
        return np.asarray(r, dtype=np.int64) if raw else fx.to_float(r, spec)
    return r


def call_entry(klong, name, *args, signature=None, raw=False):
    """
    Call the entry ``name`` of a loaded program with Python values, as the
    host of the compiled program would: one argument per parameter of the
    ``.entry`` signature (numbers, lists or numpy arrays), checked and
    converted to the declared types; the result is checked the same way and
    returned as a Python int or float, a numpy array, or None for ``void``.

    signature picks one entry when the function has several; otherwise the
    first whose parameters fit the arguments is used.

    A program that uses fixed-point numbers runs on the typed-IR evaluator
    (klongpy/static/eval_ir.py): the interpreter has no fixed-point type. A
    fixed-point argument is a number, a list or a FixedArray; a fixed-point
    result is returned as a float (exact up to 53 bits), or with raw=True as
    the scaled integers (int64) that compiled code works on.
    """
    table = klong.externs
    cands = [e for e in table.entries if e.name == name]
    if not cands:
        raise KlongNameError(f"{name} is not an entry of the program", hint=f"declare it: .entry(\"{name}\";\"... -> ...\")")
    if signature is not None:
        cands = [table.entry(name, signature)]
    err = None
    for e in cands:
        sig = e.signature
        if len(args) != sig.arity:
            err = KlongTypeError(f"entry {name} is declared \"{sig}\" and takes {sig.arity} argument(s), got {len(args)}")
            continue
        try:
            values = [conform(ty, a, f"argument {i + 1} of entry {name}") for i, (ty, a) in enumerate(zip(sig.params, args))]
        except Mismatch as m:
            err = (KlongShapeError if m.kind == "shape" else KlongTypeError)(f"{m.message} (declared \"{sig}\")")
            continue
        if uses_fixed(klong):
            return _call_evaluated(klong, name, sig, values, raw)
        values = [klong._backend.kg_asarray(v) if ty.shape else v for ty, v in zip(sig.params, values)]
        r = klong[name](*values)
        try:
            return conform(sig.result, r, f"the result of entry {name}")
        except Mismatch as m:
            raise (KlongShapeError if m.kind == "shape" else KlongTypeError)(f"{m.message} (declared \"{sig}\")") from None
    raise err
