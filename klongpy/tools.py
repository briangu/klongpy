"""
The command-line tools of KlongPy, as thin calls into the native (Rust) engine's extension.

    lint_main         kgpy lint FILE...
    check_main        kgpy check FILE... [--streamable]
    dump_ir_main      kgpy check --dump-ir FILE...
    emit_main         kgpy compile c FILE... -o DIR [--dump-ir]
    emit_hls_main     kgpy fpga source FILE... -o DIR [--synth]
    ast_main          kgpy ast FILE

Each is argument handling and one call into the extension (klong_native: lint_files, check_files, dump_ir_files, compile_c_files,
fpga_source, ast_text), which returns the status and the text for stdout and stderr; nothing here reads a
program, checks one or compiles one. What is Python's: the names the program may use that the Python package binds (the system
functions of this interpreter, see builtin_names), writing the files of `fpga source`, and C synthesis on the build machine.
"""
import os
import sys

from .native_ext import native

__all__ = ["lint_main", "check_main", "dump_ir_main", "emit_main", "emit_hls_main", "ast_main", "builtin_names",
           "library_dir"]


def library_dir():
    """The directory of the .kg library the package ships (searched after KLONGPATH by `.l`)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")


_builtins = None


def builtin_names():
    """The names a fresh interpreter binds: the engine's own system functions and variables and those KlongPy registers on it
    (`.py`, `.srv`, `.fpga`, ...). Read from a live interpreter, once."""
    global _builtins
    if _builtins is None:
        from .interpreter import KlongInterpreter
        from .repl import FRONT_END_NAMES
        _builtins = tuple(sorted(set(KlongInterpreter().names()) | set(FRONT_END_NAMES)))
    return _builtins


def _streams(stdout, stderr):
    return stdout or sys.stdout, stderr or sys.stderr


def _write(result, stdout=None, stderr=None):
    """Write what the extension returned, (status, stdout text, stderr text), and answer the status."""
    status, out_text, err_text = result
    out, err = _streams(stdout, stderr)
    out.write(out_text)
    err.write(err_text)
    return status


def lint_main(files, select=None, ignore=None, verbose=False, define=None, stdout=None, stderr=None):
    """The body of ``kgpy lint FILE...``: prints the findings and a summary; 1 if an error was found, 2 if a file could not be
    read (and no error), else 0. A file named - is read from stdin."""
    return _write(native().lint_files(list(files), select, ignore, define, verbose, library_dir(), builtin_names()), stdout, stderr)


def check_main(files, int_bits=32, verbose=False, stdout=None, stderr=None, color=False, streamable=False):
    """The body of ``kgpy check FILE... [--streamable]``: 1 if any file is outside the static profile, 2 if one could not be
    read, else 0."""
    return _write(native().check_files(list(files), int_bits == 64, verbose, library_dir(), streamable), stdout, stderr)


def dump_ir_main(files, int_bits=32, stdout=None, stderr=None):
    """The body of ``kgpy check --dump-ir FILE...``: the typed IR of each program."""
    return _write(native().dump_ir_files(list(files), int_bits == 64, library_dir()), stdout, stderr)


def emit_main(files, out_dir=None, int_bits=32, float_bits=32, line_directives=False, prefix=None, dump_ir=False, stdout=None,
              stderr=None, color=False):
    """The body of ``kgpy compile c FILE... -o DIR`` (and --dump-ir): 0, 1 when a program is outside the profile or fails at
    compile time, 2 when a file cannot be read."""
    return _write(native().compile_c_files(list(files), out_dir, int_bits == 64, float_bits == 64, line_directives, prefix, dump_ir, library_dir()),
                  stdout, stderr)


def ast_main(path, stdout=None, stderr=None):
    """The body of ``kgpy ast FILE``: the canonical parse tree, one line per statement; 2 when the file cannot be read, 1 when the
    text does not parse (the error is printed as any KlongPy error is)."""
    out, err = _streams(stdout, stderr)
    try:
        with open(path, "r") as f:
            text = f.read()
    except OSError as e:
        print(f"kgpy ast: cannot read {path}: {e.strerror or e}", file=err)
        return 2
    ext = native()
    try:
        out.write(ext.ast_text(text, path))
    except ext.NativeError as e:
        from .rust_engine import _error
        print(_error(e, path).render(False), file=err)
        return 1
    return 0


def emit_hls_main(files, out_dir=None, int_bits=32, prefix=None, clock_mhz=None, synth=False, stdout=None, stderr=None,
                  color=False, wrapped=False, lanes=None, reassociate=False):
    """The body of ``kgpy fpga source FILE... -o DIR [--synth]``: writes what Vitis HLS needs for each program (the files
    `fpga_source` returns) and prints what it wrote; with synth, runs C synthesis on the build machine ($KLONG_FPGA_BUILD; unset:
    here) and prints the estimates. Returns 0, 1 when a program is outside the profile or cannot be a kernel (or its synthesis
    fails), 2 when a file cannot be read."""
    from .static.native import HlsRefusal, emit_hls_program
    out, err = _streams(stdout, stderr)
    status = 0
    if prefix and len(files) > 1:
        print("--prefix names one program: give one FILE", file=err)
        return 2
    where = out_dir or "."
    for path in files:
        try:
            res = emit_hls_program(path, None, int_bits=int_bits, prefix=prefix, clock_mhz=clock_mhz, wrapped=wrapped, lanes=lanes,
                             reassociate=reassociate)
        except (OSError, UnicodeDecodeError) as e:
            print(f"{path}: cannot read: {getattr(e, 'strerror', None) or e}", file=err)
            status = max(status, 2)
            continue
        except HlsRefusal as e:
            print(e.text, file=err)
            status = 1
            continue
        except ValueError as e:
            print(f"{path}: {e}", file=err)
            status = 1
            continue
        res.write(where)
        names = sorted(res.files)
        print(f"{path}: wrote {', '.join(os.path.join(where, n) for n in names)}", file=out)
        for ent in res.entries:
            print(f"  kernel {ent['top']}: entry {ent['name']} ({ent['signature']}); record in {ent['in_bytes']} bytes, "
                  f"out {ent['out_bytes']} bytes", file=out)
        if synth:
            from .hw.synth import synth_main
            status = max(status, synth_main(res, where, out, err))
    return status
