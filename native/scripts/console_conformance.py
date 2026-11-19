#!/usr/bin/env python3
"""
The conformance suite through the serial console: every case is sent to
a board with `^A run` and judged by what comes back over the wire, instead of starting the
`kg` binary once per case. What a board would run gives the same answers, or this says where
it does not.

    .venv/bin/python native/scripts/console_conformance.py [--tags a,b] [--jobs N] [FILE.cases ...]
    .venv/bin/python native/scripts/console_conformance.py --port /dev/ttyUSB0 [--baud 115200]

Without --port it starts simulated boards (`native/target/release/klong-sim --pty`, or the
path in KLONG_SIM), one per job, and talks to each through its pseudo-terminal with
native/tools/kgupload.py, exactly as it talks to a real serial port. With --port it uses that
one board (one job).

Per case, as the contract of conformance/README.md asks (a fresh interpreter, the strict
level before the program is parsed): `.x(0)` resets the board, `.strict(N)` is run when the
case carries a strict tag, then the program is run; the value is the printed value of the `ok
value` reply, the console output is what came before it, an error is the kind of the `err`
reply.

Default tags: what the default build declares (native/scripts/conformance.py). --narrow: the
tags of a build with int32,float32. Options after `--` go to klong-sim (for example
`-- --heap-limit 1M --stack-budget 64K --crlf`).
"""
import importlib.util
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DECLARED = "core,float,string-fmt,width64,strict1,strict2,numpy-only,device,strings,dicts,unicode,sysio,static"
NARROW = "core,strings,dicts,unicode,sysio,static,strict1,strict2,numpy-only,device"


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class ConsoleExecutor:
    """Runs cases on boards reached through their consoles; one board per thread."""

    def __init__(self, conf, kg, sim=None, sim_args=(), port=None, baud=None, timeout=20.0):
        self.conf = conf
        self.kg = kg
        self.sim = sim
        self.sim_args = list(sim_args)
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.local = threading.local()
        self.all = []
        self.lock = threading.Lock()
        self.name = f"console of {port or sim}"

    def _open(self):
        proc = None
        if self.port:
            path = self.port
        else:
            proc = subprocess.Popen([self.sim, "--pty", *self.sim_args], stdout=subprocess.PIPE)
            line = proc.stdout.readline().decode()
            if not line.startswith("pty: "):
                proc.kill()
                raise self.kg.ProtocolError(f"{self.sim} did not name its pseudo-terminal: {line!r}")
            path = line[5:].strip()
        board = self.kg.Board(self.kg.Port.open(path, self.baud), timeout=self.timeout, run_timeout=self.timeout, retries=1)
        board.sync()
        with self.lock:
            self.all.append((proc, board))
        return proc, board

    def _drop(self):
        proc, board = self.local.board
        self.local.board = None
        board.port.close()
        if proc:
            proc.kill()
            proc.wait()
            proc.stdout.close()
        with self.lock:
            self.all = [(p, b) for p, b in self.all if b is not board]

    def run(self, case):
        Result = self.conf.Result
        try:
            if getattr(self.local, "board", None) is None:
                self.local.board = self._open()
            board = self.local.board[1]
            board.reset()
            if case.strict_level:
                board.run(f".strict({case.strict_level})")
            value, out = board.run(case.source)
            return Result("value", value or "", out)
        except self.kg.BoardError as e:
            return Result("error", e.kind, e.output)
        except self.kg.ProtocolError as e:
            # a board that stopped answering is replaced (a simulated one) before the next case
            if getattr(self.local, "board", None) is not None and not self.port:
                self._drop()
            return Result("crash", str(e))

    def close(self):
        for proc, board in self.all:
            board.port.close()
            if proc:
                proc.kill()
                proc.wait()
                proc.stdout.close()
        self.all = []


def run(tags, files=None, jobs=None, sim=None, sim_args=(), port=None, baud=None, show=print):
    """Run the selected cases; returns (passed, total, failures as text lines, seconds)."""
    conf = _load("conformance_run", os.path.join(ROOT, "conformance", "run.py"))
    kg = _load("kgupload", os.path.join(ROOT, "native", "tools", "kgupload.py"))
    sim = sim or os.environ.get("KLONG_SIM") or os.path.join(ROOT, "native", "target", "release", "klong-sim")
    jobs = 1 if port else (jobs or min(8, os.cpu_count() or 2))
    executor = ConsoleExecutor(conf, kg, sim=sim, sim_args=sim_args, port=port, baud=baud)
    total = passed = 0
    failures = []
    start = time.perf_counter()
    try:
        show(f"{'file':28} {'pass':>5} {'run':>5} {'in file':>8}")
        for path in files or conf.case_files(conf.CASES_DIR):
            every = conf.load_file(path)
            cases = conf.selected(every, tags)
            rows = conf.run_cases(cases, executor, jobs)
            ok = sum(1 for _, _, p in rows if not p)
            total += len(cases)
            passed += ok
            show(f"{os.path.basename(path):28} {ok:5d} {len(cases):5d} {len(every):8d}")
            for case, result, problem in rows:
                if problem:
                    line = f"FAIL {case.where} {case.name}: " + problem.replace("\n", " | ")
                    failures.append(line)
                    show("    " + line)
    finally:
        executor.close()
    took = time.perf_counter() - start
    show(f"{'total':28} {passed:5d} {total:5d}   in {took:.1f} s through {executor.name}, {jobs} job{'s' if jobs != 1 else ''}")
    return passed, total, failures, took


def main(argv):
    sim_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, sim_args = argv[:i], argv[i + 1:]
    tags = DECLARED
    jobs = port = baud = None
    files = []
    it = iter(argv)
    for a in it:
        if a == "--tags":
            tags = next(it)
        elif a == "--narrow":
            tags = NARROW
        elif a == "--jobs":
            jobs = int(next(it))
        elif a == "--port":
            port = next(it)
        elif a == "--baud":
            baud = int(next(it))
        elif a.startswith("-"):
            print(__doc__)
            return 2
        else:
            files.append(a)
    if port and baud is None:
        baud = 115200  # the boards' console speed, as in kgupload.py
    passed, total, _, _ = run(set(tags.split(",")), files or None, jobs, sim_args=sim_args, port=port, baud=baud)
    return 0 if passed == total and total > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
