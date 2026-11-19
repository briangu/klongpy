"""
The static profile as a Python API: thin wrappers over the native engine's extension (klong_native), under the names the package
has always had (klongpy.static.check_source, compile_program, emit_program, emit_hls_program, load_program, call_entry, set_param).

    check_source(text, name, ...) -> StaticReport      the checker (klong_static::check_source)
    compile_program(path, ...) -> Compiled              check, evaluate the compile-time definitions, lower (typed IR as text)
    emit_program(path, out_dir, ...) -> Emitted         the same, then C, <prefix>.h and klong_rt.h
    emit_hls_program(path, out_dir, ...) -> HlsOutput   the C and the Vitis HLS files around it (fpga_source)
    load_program(path, externs=...) -> LoadedProgram    a program run on the native interpreter at strict level 2
    call_entry(program, name, *args), set_param(program, name, value)

What the extension does not return, and these wrappers therefore do not have (each is an "engine needs" item):
the typed IR as objects (Compiled.program is its text, `ir_text`), check_source(globals=...) (the values of compile-time
definitions the caller evaluated), the rule tables as data (RULES), a StaticReport whose functions carry their parse nodes, the
interpreter error of a compile-time evaluation as an exception of its own kind.
"""
import os
import re
import shutil

from ..errors import KlongStaticError, KlongSystemError
from ..native_ext import native
from ..source import Source
from .diagnostics import StaticDiagnostic
from .signature import EntryDecl, ExternDecl, parse_signature

__all__ = ["check_source", "check_file", "validate_global", "StaticReport", "GlobalInfo", "FunctionInfo", "compile_program",
           "emit_program", "runtime_header", "Compiled", "Emitted", "emit_hls_program", "HlsOutput", "load_program",
           "call_entry", "set_param", "c_prefix", "emit_main", "emit_hls_main", "check_main"]


def _library_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")


def _diagnostics(tuples, source):
    return [StaticDiagnostic(rule, (a, b), source, message, hint, function) for rule, a, b, _line, _col, message, hint, function in tuples]


class GlobalInfo:
    """A compile-time value: its name, its type text when the checker knows it, whether a function assigns it (state), whether
    its value was known to the checker and whether `.param` declared it."""

    def __init__(self, name, ty, assigned, resolved, param):
        self.name, self.ty, self.assigned, self.resolved, self.param = name, ty, assigned, resolved, param


class FunctionInfo:
    """A run-time function: whether an entry reaches it, whether the profile holds for it, and the argument types it was checked
    with (`specializations`: {tuple of type text or None: result type text or None})."""

    def __init__(self, name, reachable, in_profile, specializations):
        self.name, self.reachable, self.in_profile = name, reachable, in_profile
        self.specializations = specializations


class StaticReport:
    """What the checker found: ok, diagnostics (StaticDiagnostic in source order), source, entries ([EntryDecl]), externs
    ({name: ExternDecl}), globals ({name: GlobalInfo}), functions ({name: FunctionInfo}), host_only, unresolved."""

    def __init__(self, source, diags, info):
        entries, externs, globals_, functions = info
        self.source = source
        self.diagnostics = _diagnostics(diags, source)
        self.entries = [EntryDecl(n, parse_signature(s)) for n, s in entries]
        self.externs = {n: ExternDecl(n, parse_signature(s, allow_str=True), m) for n, s, m in externs}
        self.globals = {n: GlobalInfo(n, ty, a, r, p) for n, ty, a, r, p in globals_}
        self.functions = {n: FunctionInfo(n, r, i, {tuple(a): b for a, b in sp}) for n, r, i, sp in functions}

    @property
    def ok(self):
        return not self.diagnostics

    @property
    def host_only(self):
        return [f.name for f in self.functions.values() if not f.reachable]

    @property
    def unresolved(self):
        return [g.name for g in self.globals.values() if not g.resolved]

    def errors(self):
        return [d.error() for d in self.diagnostics]

    def render(self, color=False):
        return "\n".join(d.render(color) for d in self.diagnostics)

    def summary(self):
        def plural(n, word, many=None):
            return f"{n} {word if n == 1 else (many or word + 's')}"
        fns = [f for f in self.functions.values() if f.reachable]
        return ", ".join([plural(len(fns), "function"), plural(len(self.entries), "entry", "entries"),
                          plural(len(self.externs), "extern"), plural(len(self.globals), "global")])

    def raise_first(self):
        """Raise the first diagnostic as a KlongStaticError; the error carries all of them as ``diagnostics``."""
        if self.diagnostics:
            e = self.diagnostics[0].error()
            e.diagnostics = list(self.diagnostics)
            raise e
        return self


def check_source(text, name="<string>", *, int_bits=32, globals=None, base_dir=None, resolve=True, all_functions=False):
    """Check a program text against the static profile; ``report.ok`` is True when it is inside. Nothing in the program is
    evaluated and no other file is run. `globals` (values the caller evaluated) is not supported by the extension."""
    if int_bits not in (32, 64):
        raise ValueError(f"int_bits must be 32 or 64, not {int_bits!r}")
    if globals:
        raise NotImplementedError("check_source(globals=...): the extension cannot take the values of compile-time definitions")
    source = text if isinstance(text, Source) else Source(name, text)
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(name)) if os.path.isfile(name) else os.getcwd()
    diags, info = native().check_source(source.name, source.text, int_bits == 64, base_dir, resolve, _library_dir(), all_functions)
    return StaticReport(source, diags, info)


def check_file(path, **options):
    """Check a file. OSError and UnicodeDecodeError propagate."""
    with open(path, "r") as f:
        text = f.read()
    return check_source(text, path, **options)


def validate_global(*args, **kwargs):
    raise NotImplementedError("validate_global: the extension has no counterpart (engine needs it)")


def runtime_header():
    """The path of klong_rt.h, shipped with the package."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "klong_rt.h")


def c_prefix(path):
    """The default C prefix of a program: its file name without the extension, as a C identifier."""
    from .signature import _C_KEYWORDS
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    stem = re.sub(r"[^A-Za-z0-9_]", "_", stem) or "prog"
    if not stem[0].isalpha():
        stem = "prog_" + stem
    if stem == "kg" or stem.startswith("kg_") or stem in _C_KEYWORDS:
        stem = "prog_" + stem
    return stem


class Compiled:
    """What the driver produced: the typed IR as text (`ir_text`)."""

    def __init__(self, ir_text):
        self.ir_text = ir_text


class Emitted:
    """The texts of the emitted program (`c`, `h`, `prefix`) and its size report (`report`: a dict of the numbers; `report_text()`)."""

    def __init__(self, prefix, c, h, report, text):
        self.prefix, self.c, self.h = prefix, c, h
        self.report = dict(report[0])
        self.call_chain, self.entries, self.externs, self.need_math = report[1], report[2], report[3], report[4]
        self._text = text

    def report_text(self):
        return self._text


def _raise(name, text, diags, error):
    source = Source(name, text)
    if diags:
        ds = _diagnostics(diags, source)
        e = ds[0].error()
        e.diagnostics = ds
        raise e
    raise KlongSystemError(error or "the compile failed")


def _read(path, text):
    if text is None:
        with open(path, "r") as f:
            text = f.read()
    return os.fspath(path), text


def compile_program(path, *, text=None, int_bits=32, externs=None, name=None):
    """Check, evaluate the compile-time definitions, check again and lower. Raises KlongStaticError (the first diagnostic; all of
    them in ``diagnostics``) when the program is outside the profile."""
    path, text = _read(path, text)
    ok, ir_text, diags, error = native().dump_ir_source(path, text, int_bits == 64, name)
    if not ok:
        _raise(path, text, diags, error)
    return Compiled(ir_text)


def emit_program(path, out_dir=None, *, text=None, int_bits=32, float_bits=32, line_directives=False, prefix=None, externs=None,
                 hls=None):
    """Compile a program to C. With out_dir, writes <prefix>.c, <prefix>.h and klong_rt.h there."""
    path, text = _read(path, text)
    ext = native()
    ok, pre, c, h, report, diags, error = ext.compile_c_source(path, text, int_bits == 64, float_bits == 64, line_directives, prefix)
    if not ok:
        if not diags and error is not None and not error.startswith(("[", " ")) and ("prefix" in error or "ValueError" in error):
            raise ValueError(error)
        _raise(path, text, diags, error)
    out = Emitted(pre, c, h, report[:5], report[5])
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, pre + ".c"), "w") as f:
            f.write(c)
        with open(os.path.join(out_dir, pre + ".h"), "w") as f:
            f.write(h)
        with open(os.path.join(out_dir, "klong_rt.h"), "w") as f:
            f.write(ext.runtime_header())
    return out


class HlsOutput:
    """The generated files: ``files`` {name: text}, ``manifest`` (a dict), ``entries`` (one dict per entry)."""

    def __init__(self, prefix, files, manifest):
        self.prefix = prefix
        self.files = files
        self.manifest = manifest

    @property
    def entries(self):
        return self.manifest["entries"]

    def write(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for name, text in self.files.items():
            with open(os.path.join(out_dir, name), "w") as f:
                f.write(text)


class HlsRefusal(KlongStaticError):
    """The extension refused a program: outside the profile, or an entry that cannot be a kernel. ``text`` is what it rendered; the
    message, the rule and the hint are read back from it. (Engine need: fpga_source returning the diagnostics as tuples, as
    compile_c_source does.) A refusal that is not a diagnostic (a bad option) is a ValueError, as it always was."""

    def __init__(self, text):
        m = re.search(r"static error: (.*) \[([\w-]+)\]\n", text + "\n")
        h = re.search(r"\n\s*hint: (.*)", text)
        KlongStaticError.__init__(self, f"{m.group(1)} [{m.group(2)}]" if m else text, hint=h.group(1) if h else None)
        self.rule = m.group(2) if m else None
        self.text = text


def emit_hls_program(path, out_dir=None, *, text=None, int_bits=32, prefix=None, clock_mhz=None, externs=None,
                     line_directives=False, wrapped=False, lanes=None, reassociate=False):
    """Compile a program to C and the Vitis HLS files around it (fpga_source). With out_dir, writes them there."""
    import json
    path, text = _read(path, text)
    if wrapped and (lanes is not None or reassociate):
        raise ValueError("--lanes and --reassociate belong to the streamable back end, not to --wrapped")
    ok, files, entries, error = native().fpga_source(path, text, int_bits == 64, prefix, clock_mhz or 300, wrapped, lanes, reassociate)
    if not ok:
        error = error or "the program cannot be a kernel"
        if "static error:" not in error:
            raise ValueError(error)
        raise HlsRefusal(error)
    files = dict(files)
    manifest = json.loads(files["manifest.json"])
    out = HlsOutput(prefix or manifest.get("prefix") or c_prefix(path), files, manifest)
    if out_dir is not None:
        out.write(out_dir)
    return out


# ---- running a static program -----------------------------------------------------------------------------------------------------

_FIXED_USE = re.compile(r"fixed<|\.fixed\(")


def uses_fixed(klong):
    """Does the loaded program use fixed-point numbers? Such a program runs through call_entry on the engine's typed-IR evaluator
    (StaticProgram): the interpreter has no fixed-point type."""
    src = getattr(klong, "_static_source", None)
    return bool(src and _FIXED_USE.search(src[1]))


def _host_out(v):
    """A value of the extension as Python has it: a fixed-point (elem, raw) as a FixedArray, nested lists as NumPy arrays."""
    import numpy as np
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
        from . import fixed as fx
        return fx.FixedArray(np.array(v[1], dtype=np.int64), fx.spec_of(v[0]))
    if isinstance(v, list):
        return np.array(v)
    return v


def _program_of(klong):
    """The StaticProgram of a loaded program that uses fixed-point numbers (built on first use)."""
    p = getattr(klong, "_static_native", None)
    if p is None:
        path, text, externs = klong._static_source
        p = klong._static_native = native().StaticProgram(path, text, externs)
    return p


def load_program(path, externs=None, *, text=None, strict=2, klong=None, backend=None):
    """
    Run a program and return the interpreter that holds its state.

        path      the file to run (or, with text=, the name the text is known by)
        externs   {name: callable}: the stand-ins, bound before the program runs
        text      the program text, instead of reading path
        strict    the strict level (2: what the static profile requires)
        klong     an existing KlongInterpreter to run in, instead of a new one
        backend   backend name for a new interpreter

    Top-level definitions are evaluated in order, exactly as kgpy does; ``.extern`` binds each declared name to its stand-in and
    ``.entry`` validates each signature. Errors are KlongError, positioned in the file.
    """
    from ..interpreter import KlongInterpreter
    if klong is None:
        klong = KlongInterpreter(backend=backend, strict=strict)
    if externs:
        klong.externs.update(externs)
    path, text = _read(path, text)
    klong._static_source = (path, text, dict(externs or {}))
    klong._static_native = None
    klong(Source(path, text))
    return klong


def call_entry(klong, name, *args, signature=None, raw=False):
    """
    Call the entry ``name`` of a loaded program with Python values, as the host of the compiled program would: one argument per
    parameter of the ``.entry`` signature (numbers, lists or numpy arrays), checked and converted to the declared types; the result
    is checked the same way and returned as a Python int or float, a numpy array, or None for ``void``.

    signature picks one entry when the function has several; otherwise the first whose parameters fit the arguments is used.

    A program that uses fixed-point numbers runs on the engine's typed-IR evaluator (StaticProgram): the interpreter has no
    fixed-point type. A fixed-point argument is a number, a list or a FixedArray; a fixed-point result is returned as a float (exact
    up to 53 bits), or with raw=True as the scaled integers (int64) that compiled code works on.
    """
    if uses_fixed(klong):
        r = _native_call(lambda: _program_of(klong).call_entry(name, list(args), signature, raw))
        return _host_out(r) if not raw else (np_int64(r) if isinstance(r, list) else r)
    from ..errors import KlongNameError, KlongShapeError, KlongTypeError
    from .signature import Mismatch, conform
    cands = [e for e in klong.externs.entries if e.name == name]
    if not cands:
        raise KlongNameError(f"{name} is not an entry of the program", hint=f"declare it: .entry(\"{name}\";\"... -> ...\")")
    if signature is not None:
        cands = [klong.externs.entry(name, signature)]
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


def _native_call(call):
    """Run a call of the extension; its error is raised as the KlongError of its kind."""
    ext = native()
    try:
        return call()
    except ext.NativeError as e:
        from ..rust_engine import _error
        raise _error(e)


def np_int64(r):
    import numpy as np
    return np.array(r, dtype=np.int64)


def set_param(klong, name, value):
    """Set the parameter ``name`` (declared with ``.param``) between runs: the value is checked against the declared type and
    converted ("int" takes an int, "float" an int or a float, "fixed<W,I>" an int, a float or a fixed-point value, truncating toward
    minus infinity)."""
    from ..errors import KlongNameError, KlongShapeError, KlongTypeError
    from .signature import Mismatch, conform, is_fixed
    decl = klong.externs.params.get(name)
    if decl is None:
        raise KlongNameError(f"{name} is not a parameter of the program", hint=f'declare it: .param("{name}";"int")')
    try:
        v = conform(decl.ty, value, f"the value of parameter {name}")
    except Mismatch as m:
        raise (KlongShapeError if m.kind == "shape" else KlongTypeError)(m.message) from None
    klong[name] = v if is_fixed(decl.ty.elem) else (int(v) if decl.ty.elem == "int" else float(v))
    if uses_fixed(klong):
        _native_call(lambda: _program_of(klong).set_param(name, v))
    return v


def check_main(*args, **kwargs):
    from ..tools import check_main as main
    return main(*args, **kwargs)


def emit_main(*args, **kwargs):
    from ..tools import emit_main as main
    return main(*args, **kwargs)


def emit_hls_main(*args, **kwargs):
    from ..tools import emit_hls_main as main
    return main(*args, **kwargs)
