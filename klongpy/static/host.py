"""
Running a static program under KlongPy (the host half of ticket 094; what the
differential harness of ticket 093 builds on).

    klong = load_program("examples/embedded/pid.kg",
                         externs={"read.adc": lambda ch: 0.3, "write.pwm": lambda ch, duty: 0})
    call_entry(klong, "step", 0.31)        # 0.12005

load_program runs the file at strict level 2 with the Python functions bound
as its externs; call_entry calls an entry the way the C caller would, with
the arguments and the result checked against the ``.entry`` signature.
"""
import os

from ..errors import KlongNameError, KlongShapeError, KlongTypeError
from ..lexer import Source
from .signature import Mismatch, conform

__all__ = ["load_program", "call_entry"]


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
    klong(Source(os.fspath(path), text))
    return klong


def call_entry(klong, name, *args, signature=None):
    """
    Call the entry ``name`` of a loaded program with Python values, as the
    host of the compiled program would: one argument per parameter of the
    ``.entry`` signature (numbers, lists or numpy arrays), checked and
    converted to the declared types; the result is checked the same way and
    returned as a Python int or float, a numpy array, or None for ``void``.

    signature picks one entry when the function has several; otherwise the
    first whose parameters fit the arguments is used.
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
        values = [klong._backend.kg_asarray(v) if ty.shape else v for ty, v in zip(sig.params, values)]
        r = klong[name](*values)
        try:
            return conform(sig.result, r, f"the result of entry {name}")
        except Mismatch as m:
            raise (KlongShapeError if m.kind == "shape" else KlongTypeError)(f"{m.message} (declared \"{sig}\")") from None
    raise err
