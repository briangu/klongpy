#!/usr/bin/env python3
"""
Conformance suite runner (format and protocol: conformance/README.md).

    python conformance/run.py                         every case against KlongPy, in process
    python conformance/run.py --cmd "./klong --conformance"
                                                      every case against an external executable
    python conformance/run.py --tags core,strings     only what a runtime that has these tags can run
    python conformance/run.py --list                  list the cases that would run
    python conformance/run.py --bless FILE            fill in the expectations of cases marked `?`
    python conformance/run.py --trees --cmd CMD       golden parse trees (tests/golden) against CMD

The case file parser (`load_file`, `load_dir`) and the executors have no
dependency on KlongPy; only the in-process executor imports it.
"""
import argparse
import concurrent.futures
import importlib.util
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CASES_DIR = os.path.join(HERE, "cases")
PENDING_DIR = os.path.join(HERE, "pending")
GOLDEN_DIR = os.path.join(ROOT, "tests", "golden")

TIERS = ("core", "strings", "dicts", "sysio", "timers", "ipc", "desktop")
MARKS = ("width64", "bigint", "unicode", "device", "sysio", "device")
CAPABILITIES = ("float", "string-fmt", "strict1", "strict2")
IMPLEMENTATION = ("numpy-only", "torch-only", "kgpy-only")
ALL_TAGS = TIERS + MARKS + CAPABILITIES + IMPLEMENTATION
KINDS = ("syntax", "name", "arity", "type", "shape", "range", "system", "memory", "interrupt")
META_KEYS = ("reason", "actual", "ticket")

# What KlongPy (numpy backend) declares. `device` needs a heap limit and a console;
# the other tiers have no cases in this suite yet.
KLONGPY_TAGS = frozenset(("core", "strings", "dicts", "float", "string-fmt", "strict1", "strict2",
                          "width64", "bigint", "unicode", "device", "sysio", "numpy-only", "kgpy-only"))
# KlongPy on the torch backend: floats are 32 bits wide by default and a tensor cannot
# hold :undefined, so it does not declare `numpy-only` (README, "Tags").
KLONGPY_TORCH_TAGS = (KLONGPY_TAGS - {"numpy-only"}) | {"torch-only"}


def klongpy_tags(backend="numpy"):
    """The tags KlongPy declares when it runs on the given backend."""
    return KLONGPY_TORCH_TAGS if backend == "torch" else KLONGPY_TAGS


class FormatError(Exception):
    pass


class Case:
    """One case. `expect` is ("value", text), ("error", kind) or ("none", None)."""

    def __init__(self, file, line, name, tags):
        self.file = file
        self.line = line
        self.name = name
        self.tags = tags
        self.source_lines = []
        self.value_lines = []
        self.out_lines = []
        self.error = None
        self.unfilled = False
        self.meta = {}

    @property
    def source(self):
        return "\n".join(self.source_lines)

    @property
    def expect(self):
        if self.error is not None:
            return ("error", self.error)
        if self.value_lines:
            return ("value", "\n".join(self.value_lines))
        return ("none", None)

    @property
    def expect_out(self):
        return "\n".join(self.out_lines) if self.out_lines else None

    @property
    def strict_level(self):
        for n in (2, 1):
            if f"strict{n}" in self.tags:
                return n
        return None

    @property
    def where(self):
        return f"{os.path.relpath(self.file, ROOT)}:{self.line}"


def _text(line):
    """The text after a one-character sigil and at most one space."""
    rest = line[1:]
    return rest[1:] if rest.startswith(" ") else rest


def parse_text(text, path="<text>"):
    cases = []
    cur = None
    for n, raw in enumerate(text.split("\n"), 1):
        line = raw.rstrip("\r")
        if line == "" or line.startswith("#"):
            continue
        sig = line[0]

        def bad(msg):
            return FormatError(f"{os.path.relpath(path, ROOT) if os.path.isabs(path) else path}:{n}: {msg}")

        if sig == "@":
            parts = line[1:].split()
            if not parts:
                raise bad("case without a name")
            name, tags = parts[0], parts[1:]
            for t in tags:
                if t not in ALL_TAGS:
                    raise bad(f"unknown tag {t!r} (known: {' '.join(ALL_TAGS)})")
            tiers = [t for t in tags if t in TIERS]
            if not tiers or (len(tiers) > 1 and "core" in tiers):
                raise bad(f"case {name} needs one or more tier tags from {' '.join(TIERS)} "
                          f"(`core` alone, or the other tiers it needs; core is implied by them)")
            cur = Case(path, n, name, tuple(tags))
            cases.append(cur)
            continue
        if cur is None:
            raise bad("text before the first @ line")
        if sig == ">":
            cur.source_lines.append(_text(line))
        elif sig == "=":
            cur.value_lines.append(_text(line))
        elif sig == "~":
            cur.out_lines.append(_text(line))
        elif sig == "!":
            kind = _text(line).strip()
            if kind not in KINDS:
                raise bad(f"unknown error kind {kind!r} (known: {' '.join(KINDS)})")
            if cur.error is not None:
                raise bad("two ! lines")
            cur.error = kind
        elif sig == "?":
            cur.unfilled = True
        elif sig == ":":
            key, _, val = _text(line).partition(" ")
            if key not in META_KEYS:
                raise bad(f"unknown metadata key {key!r} (known: {' '.join(META_KEYS)})")
            cur.meta[key] = val.strip()
        else:
            raise bad(f"unknown line type {sig!r}")
    names = set()
    for c in cases:
        if c.name in names:
            raise FormatError(f"{path}:{c.line}: duplicate case name {c.name}")
        names.add(c.name)
        if not c.source_lines:
            raise FormatError(f"{path}:{c.line}: case {c.name} has no source")
        if c.error is not None and c.value_lines:
            raise FormatError(f"{path}:{c.line}: case {c.name} has both = and !")
    return cases


def load_file(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return parse_text(f.read(), path)


def case_files(directory):
    out = []
    for dp, _, fns in os.walk(directory):
        out.extend(os.path.join(dp, f) for f in fns if f.endswith(".cases"))
    return sorted(out)


def load_dir(directory):
    cases = []
    for p in case_files(directory):
        cases.extend(load_file(p))
    return cases


# ---------------------------------------------------------------------------
# Executors. run(case) -> Result
# ---------------------------------------------------------------------------

class Result:
    """kind: "value" (text may be ""), "error" (text is the error kind) or "crash" (text explains)."""

    def __init__(self, kind, text="", out=""):
        self.kind = kind
        self.text = text
        self.out = out

    def as_expect(self):
        if self.kind == "value":
            return ("value", self.text) if self.text != "" else ("none", None)
        return (self.kind, self.text)


def _load_shim():
    path = os.path.join(HERE, "shim", "kgpy_shim.py")
    spec = importlib.util.spec_from_file_location("kgpy_shim", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, ROOT)
    spec.loader.exec_module(mod)
    return mod


class InProcess:
    """KlongPy in this process: a fresh interpreter per case, through the same function the shim uses."""
    name = "KlongPy (in process)"

    def __init__(self, backend=None):
        self.shim = _load_shim()
        self.backend = backend  # None: the interpreter's default

    def run(self, case):
        status, text, out = self.shim.evaluate(case.source, case.strict_level, backend=self.backend)
        return Result("value" if status == "result" else "error", text, out)


class External:
    """
    An executable speaking the one-shot protocol: argv `[--strict N]`, program text on stdin;
    stdout is the printed result (exit 0) or `error: <kind>` (exit 1); stderr is the program's console output.
    """

    def __init__(self, cmd, timeout=20.0):
        self.cmd = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        self.timeout = timeout
        self.name = " ".join(self.cmd)

    def run(self, case):
        argv = list(self.cmd)
        if case.strict_level:
            argv += ["--strict", str(case.strict_level)]
        try:
            p = subprocess.run(argv, input=case.source.encode("utf-8"), capture_output=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return Result("crash", f"timeout after {self.timeout}s")
        except OSError as e:
            return Result("crash", f"cannot run {argv[0]}: {e}")
        return parse_response(p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace"))


def parse_response(code, stdout, stderr):
    if stdout.endswith("\n"):
        stdout = stdout[:-1]
    if code == 0:
        return Result("value", stdout, stderr)
    if code == 1 and stdout.startswith("error: "):
        return Result("error", stdout[len("error: "):].strip(), stderr)
    return Result("crash", f"exit status {code}, stdout {stdout[:200]!r}, stderr {stderr[-200:]!r}", stderr)


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------

def check(case, result):
    """Return None if result satisfies the case, else a one-line-per-field explanation."""
    want = case.expect
    got = result.as_expect()
    problems = []
    if result.kind == "crash":
        problems.append(f"crashed: {result.text}")
    elif want != got:
        problems.append(f"expected {_show(want)}")
        problems.append(f"     got {_show(got)}")
    if case.expect_out is not None and result.kind != "crash":
        if result.out != case.expect_out and result.out.rstrip("\n") != case.expect_out:
            problems.append(f"console output expected {case.expect_out!r}, got {result.out!r}")
    return "\n".join(problems) or None


def _show(e):
    kind, text = e
    if kind == "value":
        return text
    if kind == "error":
        return f"error: {text}"
    return "(no value)"


def selected(cases, tags):
    return [c for c in cases if all(t in tags for t in c.tags)]


def run_cases(cases, executor, jobs=1):
    """Return a list of (case, Result, problem-or-None) in case order."""
    if jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
            results = list(pool.map(executor.run, cases))
    else:
        results = [executor.run(c) for c in cases]
    return [(c, r, check(c, r)) for c, r in zip(cases, results)]


# ---------------------------------------------------------------------------
# Blessing (authoring aid): fill in `?` cases from the executor's answer. Review the diff.
# ---------------------------------------------------------------------------

def bless(path, executor):
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    cases = [c for c in parse_text(text, path) if c.unfilled]
    if not cases:
        return 0
    lines = text.split("\n")
    inserts = {}
    for c in cases:
        r = executor.run(c)
        new = []
        if r.kind == "error":
            new.append(f"! {r.text}")
        elif r.kind == "crash":
            new.append(f"# CRASH {r.text}")
        else:
            new.extend(("= " + t).rstrip() if t else "=" for t in r.text.split("\n")) if r.text else None
        if r.out:
            new.extend(("~ " + t).rstrip() if t else "~" for t in r.out.rstrip("\n").split("\n"))
        inserts[c.line] = new
    # replace the `?` line of each case
    out, cur = [], None
    for n, line in enumerate(lines, 1):
        if line.startswith("@"):
            cur = n
        if line.startswith("?") and cur in inserts:
            out.extend(inserts[cur])
        else:
            out.append(line)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(out))
    return len(cases)


# ---------------------------------------------------------------------------
# Golden parse trees
# ---------------------------------------------------------------------------

def golden_pairs(base=GOLDEN_DIR):
    """Yield (name, source path, golden text) for every tests/golden/<group>/<name>.sexp with its source."""
    spec = importlib.util.spec_from_file_location("golden_regen", os.path.join(base, "regen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for group, name, src, golden in mod.cases():
        with open(golden, "r", encoding="utf-8") as f:
            yield f"{group}/{name}", src, f.read()


def run_tree(cmd, src, want, timeout=20.0):
    """Run `cmd --dump-ast SRC`; None if it produced the golden tree (or, for `(error ...)`, a syntax error)."""
    argv = (shlex.split(cmd) if isinstance(cmd, str) else list(cmd)) + ["--dump-ast", src]
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"crashed: {e}"
    out = p.stdout.decode("utf-8", "replace")
    if want.startswith("(error "):
        if p.returncode == 1 and out.strip() == "error: syntax":
            return None
        return f"expected `error: syntax` with exit status 1, got status {p.returncode}, {out[:120]!r}"
    if p.returncode == 0 and out == want:
        return None
    return f"expected tree {want[:120]!r}, got status {p.returncode}, {out[:120]!r}"


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="case files or directories (default: conformance/cases)")
    ap.add_argument("--cmd", help="external executable (default: KlongPy in process)")
    ap.add_argument("--tags", help="comma-separated tags the implementation supports "
                                   "(default for KlongPy: %s)" % ",".join(sorted(KLONGPY_TAGS)))
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes for --cmd")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pending", action="store_true", help="run conformance/pending (each is expected to fail)")
    ap.add_argument("--bless", metavar="FILE", help="fill in cases marked `?` using KlongPy (or --cmd)")
    ap.add_argument("--trees", action="store_true", help="check golden parse trees with `--cmd CMD --dump-ast FILE`")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    executor = External(args.cmd, args.timeout) if args.cmd else InProcess()
    if args.trees:
        if not args.cmd:
            ap.error("--trees needs --cmd")
        bad = n = 0
        for name, src, want in golden_pairs():
            n += 1
            p = run_tree(args.cmd, src, want, args.timeout)
            if p:
                bad += 1
                print(f"FAIL tree {name}: {p}")
        print(f"{n - bad} of {n} trees match")
        return 1 if bad else 0
    if args.bless:
        n = bless(args.bless, executor)
        print(f"filled {n} cases in {args.bless}")
        return 0

    tags = set(args.tags.split(",")) if args.tags else set(KLONGPY_TAGS)
    paths = args.paths or [PENDING_DIR if args.pending else CASES_DIR]
    cases = []
    for p in paths:
        cases.extend(load_dir(p) if os.path.isdir(p) else load_file(p))
    todo = selected(cases, tags)
    if args.list:
        for c in todo:
            print(f"{c.where}\t{c.name}\t{' '.join(c.tags)}")
        return 0
    rows = run_cases(todo, executor, args.jobs)
    failed = [(c, p) for c, _, p in rows if p]
    if args.pending:
        # a case whose reason starts with "native:" passes in KlongPy on purpose:
        # it waits for the native runtime, which does not give the result yet
        waiting = [c for c, _, p in rows if not p and c.meta.get("reason", "").startswith("native:")]
        fixed = [c for c, _, p in rows if not p and c not in waiting]
        for c in fixed:
            print(f"NOW PASSES {c.where} {c.name}: move it to cases/")
        rest = len(rows) - len(waiting)
        print(f"{rest - len(fixed)} of {rest} pending cases still fail as expected"
              + (f"; {len(waiting)} more pass in KlongPy and wait for the native runtime" if waiting else ""))
        return 0
    for c, p in failed:
        print(f"FAIL {c.where} {c.name}\n  " + p.replace("\n", "\n  "))
    print(f"{len(rows) - len(failed)} passed, {len(failed)} failed, {len(cases) - len(todo)} skipped by tags "
          f"({executor.name})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
