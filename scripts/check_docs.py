"""Check the transcript examples in the documentation.

    python scripts/check_docs.py                  # README.md and docs/*.md of this repository
    python scripts/check_docs.py FILE.md ...      # these files only

Two kinds of fenced block are run:

* a block whose first line starts with ``?> `` runs in one fresh REPL interpreter.
  Each ``?> `` line is an input; the lines up to the next one are what it must
  print followed by the value the REPL shows. A failure is shown as
  ``error: <first line of the message>``. Blocks that need a server or a web
  port (``.clid(``, ``.srv(88``, ``.web(``, ``.ws(``) are skipped in this form.
* a block whose first line starts with ``server?> `` or ``client?> `` runs in two
  interpreters in this process, connected over localhost. ``server?> `` lines
  run in the first, ``client?> `` lines in the second. The port 8888 written in
  the block is replaced by a free port, in the inputs and in the expected text.
  After each input the script waits a moment so that callbacks have run.

The script exits with status 1 when an example does not match. Run it from
anywhere with the repository's interpreter, for example
``.venv/bin/python scripts/check_docs.py``.
"""
import contextlib
import glob
import io
import os
import re
import socket
import sys
import time
import warnings

warnings.simplefilter("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from klongpy.repl import cleanup_repl, create_repl  # noqa: E402
from klongpy.writer import kg_write  # noqa: E402

SKIP_SINGLE = r"\.clid?\(|\.srv\(88|\.web\(|\.ws\("
PAIR = re.compile(r"^(server|client)\?> ")
SETTLE = 0.15       # seconds to let callbacks run after each two-interpreter step
DOC_PORT = "8888"


def blocks(text):
    """(kind, body) for every runnable fenced block: kind is 'single' or 'pair'."""
    for m in re.finditer(r"```[a-z]*\n(.*?)```", text, re.S):
        body = m.group(1)
        first = body.lstrip()
        if first.startswith("?> ") and not re.search(SKIP_SINGLE, body):
            yield "single", body
        elif PAIR.match(first):
            yield "pair", body


def parse(body, prefixes):
    """[[who, input, [expected lines]]]; ``who`` is None for the single form."""
    steps = []
    for line in body.splitlines():
        for who, prefix in prefixes.items():
            if line.startswith(prefix):
                steps.append([who, line[len(prefix):], []])
                break
        else:
            if steps:
                steps[-1][2].append(line)
    for st in steps:
        while st[2] and st[2][-1] == "":
            st[2].pop()
    return steps


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run_step(klong, expr, buf):
    buf.seek(0)
    buf.truncate()
    try:
        r = klong(expr)
        return (buf.getvalue() + kg_write(r, klong._backend, display=False)).split("\n")
    except Exception as ex:
        return (buf.getvalue() + "error: " + str(ex).split("\n")[0]).split("\n")


def check_single(path, body, report):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        klong, loops = create_repl()
        try:
            for _, expr, exp in parse(body, {None: "?> "}):
                got = run_step(klong, expr, buf)
                report(path, expr, exp, got)
        finally:
            cleanup_repl(loops)


def check_pair(path, body, report):
    port = str(free_port())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        server, sloops = create_repl()
        client, cloops = create_repl()
        env = {"server": server, "client": client}
        try:
            for who, expr, exp in parse(body, {"server": "server?> ", "client": "client?> "}):
                expr = expr.replace(DOC_PORT, port)
                exp = [line.replace(DOC_PORT, port) for line in exp]
                got = run_step(env[who], expr, buf)
                time.sleep(SETTLE)
                report(path, expr, exp, got)
        finally:
            for k in (server, client):
                try:
                    k['.system']['closeEvent'].trigger()
                except Exception:
                    pass
            cleanup_repl(sloops)
            cleanup_repl(cloops)


def main(argv):
    paths = argv or [os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))
    counts = {"single": 0, "pair": 0, "bad": 0}

    def make_report(kind):
        def report(path, expr, exp, got):
            counts[kind] += 1
            if got != exp:
                counts["bad"] += 1
                sys.__stdout__.write(f"MISMATCH {os.path.relpath(path, ROOT)}: {expr}\n  expected {exp}\n  got      {got}\n")
        return report

    for path in paths:
        with open(path) as f:
            text = f.read()
        for kind, body in blocks(text):
            (check_single if kind == "single" else check_pair)(path, body, make_report(kind))
    total = counts["single"] + counts["pair"]
    print(f"{total} examples checked ({counts['single']} in one interpreter, {counts['pair']} in two), "
          f"{counts['bad']} mismatches")
    return 1 if counts["bad"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
