"""
The driver of the static compiler (tickets/design/static-compile.md section 1).

    compile_program(path, ...) -> Compiled      stages one to four: check, evaluate, check again, lower
    emit_program(path, out_dir, ...) -> Emitted the same, then emit C and write <prefix>.c, <prefix>.h, klong_rt.h
    emit_main(...)                              the body of kgpy --emit-c and --dump-ir

Stage one runs the top-level definitions under KlongPy (numpy backend, strict
level 2), exactly as kgpy would run the file; the values of the globals are
then handed to the checker, which decides every rule, and to lowering. A
compile-time call of an extern needs a host stand-in (``externs=``); without
one the interpreter's error names the extern.
"""
import os
import re
import shutil
import sys
import threading

from ..errors import KlongError
from .checker import check_source
from .emit_c import emit
from .host import load_program
from .infer import lower
from .ir import dump
from .signature import _C_KEYWORDS

__all__ = ["compile_program", "emit_program", "emit_main", "Compiled", "runtime_header", "c_prefix", "emit_hls_program",
           "emit_hls_main"]


_CHDIR = threading.RLock()


def runtime_header():
    """The path of klong_rt.h, shipped with the package."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "klong_rt.h")


def c_prefix(path):
    """The default C prefix of a program: its file name without the extension, as a C identifier."""
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    stem = re.sub(r"[^A-Za-z0-9_]", "_", stem) or "prog"
    if not stem[0].isalpha():
        stem = "prog_" + stem
    if stem == "kg" or stem.startswith("kg_") or stem in _C_KEYWORDS:
        stem = "prog_" + stem
    return stem


class Compiled:
    """What the driver produced: the final StaticReport, the stage-one values and the ir.Program."""
    def __init__(self, report, values, program):
        self.report = report
        self.values = values
        self.program = program


def compile_program(path, *, text=None, int_bits=32, externs=None, name=None):
    """
    Check, evaluate stage one, check again with the values, and lower.

        path      the program file (or, with text=, the name the text is known by)
        externs   {name: callable}: host stand-ins for externs that compile-time code calls
        name      the program's name in the IR (default: from the file name)

    Raises KlongStaticError (the first diagnostic; all of them are in its
    ``diagnostics``) when the program is outside the profile, and whatever
    KlongError the interpreter raises while it evaluates stage one.
    """
    if text is None:
        with open(path, "r") as f:
            text = f.read()
    path = os.fspath(path)
    base_dir = os.path.dirname(os.path.abspath(path)) if os.path.isfile(path) else None
    first = check_source(text, path, int_bits=int_bits, base_dir=base_dir)
    # form errors are decided without running anything: stop before stage one if there are any
    first.raise_first()
    # .l("file") is looked up from the directory of the program, as the checker does
    with _CHDIR:        # the working directory is the process's: two threads compiling must not interleave their chdirs
        cwd = os.getcwd()
        try:
            if base_dir:
                os.chdir(base_dir)
            klong = load_program(os.path.abspath(os.path.join(cwd, path)) if base_dir else path, externs=externs, text=text,
                                 strict=2, backend="numpy")
        finally:
            os.chdir(cwd)
    values = {}
    for gname in first.globals:
        values[gname] = klong[gname]
    report = check_source(text, path, int_bits=int_bits, globals=values, base_dir=base_dir)
    report.raise_first()
    program = lower(report, values, name or c_prefix(path), int_bits)
    return Compiled(report, values, program)


def emit_program(path, out_dir=None, *, text=None, int_bits=32, float_bits=32, line_directives=False, prefix=None,
                 externs=None, hls=None):
    """
    Compile a program to C. With out_dir, writes <prefix>.c, <prefix>.h and
    klong_rt.h there. Returns the Emitted (texts and size report), with the
    Compiled as its ``compiled`` attribute.
    """
    compiled = compile_program(path, text=text, int_bits=int_bits, externs=externs, name=prefix)
    out = emit(compiled.program, prefix=prefix, float_bits=float_bits, line_directives=line_directives, hls=hls)
    out.compiled = compiled
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, out.prefix + ".c"), "w") as f:
            f.write(out.c)
        with open(os.path.join(out_dir, out.prefix + ".h"), "w") as f:
            f.write(out.h)
        shutil.copyfile(runtime_header(), os.path.join(out_dir, "klong_rt.h"))
    return out


def emit_main(files, out_dir=None, int_bits=32, float_bits=32, line_directives=False, prefix=None, dump_ir=False,
              emit_c=True, stdout=None, stderr=None, color=False):
    """
    The body of ``kgpy --emit-c FILE... -o DIR`` (and of ``--dump-ir``, with
    emit_c=False). Prints the size report of each program (or its IR) on
    stdout and errors on stderr. Returns 0, 1 when a program is outside the
    profile or fails at compile time, 2 when a file cannot be read.
    """
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    status = 0
    if prefix and len(files) > 1:
        print("--prefix names one program: give one FILE", file=err)
        return 2
    for path in files:
        try:
            if emit_c:
                res = emit_program(path, out_dir or ".", int_bits=int_bits, float_bits=float_bits,
                                   line_directives=line_directives, prefix=prefix)
                program = res.compiled.program
            else:
                program = compile_program(path, int_bits=int_bits, name=prefix).program
        except (OSError, UnicodeDecodeError) as e:
            print(f"{path}: cannot read: {getattr(e, 'strerror', None) or e}", file=err)
            status = max(status, 2)
            continue
        except KlongError as e:
            for d in getattr(e, "diagnostics", None) or [e]:
                print(d.render(color), file=err)
            status = 1 if status != 1 else status
            continue
        except ValueError as e:
            print(f"{path}: {e}", file=err)
            status = 1
            continue
        if dump_ir:
            out.write(dump(program))
        if emit_c:
            where = out_dir or "."
            print(f"{path}: wrote {os.path.join(where, res.prefix + '.c')}, {res.prefix}.h and klong_rt.h", file=out)
            print(res.report_text(), file=out)
    return status


def emit_hls_program(path, out_dir=None, *, text=None, int_bits=32, prefix=None, clock_mhz=None, externs=None,
                     line_directives=False, wrapped=False, lanes=None, reassociate=False):
    """
    Compile a program to C and the Vitis HLS files around it (emit_hls). With out_dir, writes the C files and the HLS
    files there. Returns the HlsOutput, with the Emitted as its ``emitted`` attribute.

        wrapped      stage A: the plain C of --emit-c is the core, called once per record (kgpy --hls-wrapped)
        lanes        N cores on one stream (stage B only)
        reassociate  sum floats as a balanced tree (stage B only; changes the rounding)
    """
    from .emit_c import HLS_WORK_LIMIT, HlsOptions, hls_work
    from .emit_hls import DEFAULT_CLOCK_MHZ, emit_hls
    if wrapped and (lanes is not None or reassociate):
        raise ValueError("--lanes and --hls-reassociate belong to the streamable back end, not to --hls-wrapped")
    options = None
    if not wrapped:
        compiled = compile_program(path, text=text, int_bits=int_bits, externs=externs, name=prefix)
        work = max([hls_work(compiled.program, e) for e in compiled.program.entries] or [0])
        options = HlsOptions(reassociate=reassociate, mode="pipeline" if work <= HLS_WORK_LIMIT else "block")
    emitted = emit_program(path, None, text=text, int_bits=int_bits, float_bits=32, prefix=prefix, externs=externs,
                           line_directives=line_directives, hls=options)
    out = emit_hls(emitted.compiled.program, emitted, clock_mhz=clock_mhz or DEFAULT_CLOCK_MHZ,
                   source_name=os.path.basename(os.fspath(path)), wrapped=wrapped, lanes=lanes, reassociate=reassociate,
                   mode=options.mode if options else None)
    out.emitted = emitted
    out.files[emitted.prefix + ".c"] = emitted.c
    out.files[emitted.prefix + ".h"] = emitted.h
    with open(runtime_header(), "r") as f:
        out.files["klong_rt.h"] = f.read()
    if out_dir is not None:
        out.write(out_dir)
    return out


def emit_hls_main(files, out_dir=None, int_bits=32, prefix=None, clock_mhz=None, synth=False, stdout=None, stderr=None,
                  color=False, wrapped=False, lanes=None, reassociate=False):
    """
    The body of ``kgpy --emit-hls FILE... -o DIR [--synth]``. Prints what it wrote and, with synth, runs C synthesis
    on the build machine ($KLONG_HW_BUILD; unset: this machine) and prints the estimates. Returns 0, 1 when a program
    is outside the profile or cannot be a kernel (or its synthesis fails), 2 when a file cannot be read.
    """
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    status = 0
    if prefix and len(files) > 1:
        print("--prefix names one program: give one FILE", file=err)
        return 2
    where = out_dir or "."
    for path in files:
        try:
            res = emit_hls_program(path, where, int_bits=int_bits, prefix=prefix, clock_mhz=clock_mhz, wrapped=wrapped,
                                   lanes=lanes, reassociate=reassociate)
        except (OSError, UnicodeDecodeError) as e:
            print(f"{path}: cannot read: {getattr(e, 'strerror', None) or e}", file=err)
            status = max(status, 2)
            continue
        except KlongError as e:
            for d in getattr(e, "diagnostics", None) or [e]:
                print(d.render(color), file=err)
            status = 1 if status != 1 else status
            continue
        except ValueError as e:
            print(f"{path}: {e}", file=err)
            status = 1
            continue
        names = sorted(res.files)
        print(f"{path}: wrote {', '.join(os.path.join(where, n) for n in names)}", file=out)
        for ent in res.entries:
            print(f"  kernel {ent['top']}: entry {ent['name']} ({ent['signature']}); record in {ent['in_bytes']} bytes, "
                  f"out {ent['out_bytes']} bytes", file=out)
        if synth:
            from ..hw.synth import synth_main
            status = max(status, synth_main(res, where, out, err))
    return status
