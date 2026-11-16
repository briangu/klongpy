"""Check the transcript examples in the documentation.

    python scripts/check_docs.py                  # README.md and docs/*.md of this repository
    python scripts/check_docs.py FILE.md ...      # these files only

Two kinds of fenced block are run:

* a block whose first line starts with ``?> `` runs in one fresh REPL interpreter.
  Each ``?> `` line is an input, with any ``.. `` lines directly after it as
  the rest of a multi-line entry; the lines up to the next ``?> `` are what it
  must print followed by the value the REPL shows. The last value is kept in
  ``it``, as in the REPL. A failure is shown as
  ``error: <first line of the message>``. Blocks that need a server or a web
  port (``.clid(``, ``.srv(88``, ``.web(``, ``.ws(``) are skipped in this form.
* a block whose first line starts with ``server?> `` or ``client?> `` runs in two
  interpreters in this process, connected over localhost. ``server?> `` lines
  run in the first, ``client?> `` lines in the second. The port 8888 written in
  the block is replaced by a free port, in the inputs and in the expected text.
  After each input the script waits a moment so that callbacks have run.

In both forms ``/tmp/`` in a block is replaced by a fresh temporary directory
for the run, so examples that write files do not touch the real ``/tmp``, and
the port 8888 by a free port. A ``?> `` block that starts a server or opens a
client is run only when a directive on it asks for it (see ``requires``).

A comment on the line before a fenced block changes how it is run::

    <!-- check-docs: continue -->               run in the interpreters of the previous block
    <!-- check-docs: requires duckdb pandas --> run only if these packages import, else skip
    <!-- check-docs: skip: reason -->           do not run; the reason is required

Directives combine with ``;`` (``continue; requires duckdb``); a ``skip`` reason runs to the end of the comment. ``requires`` on
a block applies to the blocks that continue it. Use ``skip`` only for an
example that cannot run here (it needs a running web server, say); the
summary line counts skipped blocks and why.

The script exits with status 1 when an example does not match. Run it from
anywhere with the repository's interpreter, for example
``.venv/bin/python scripts/check_docs.py``.
"""
import contextlib
import glob
import importlib.util
import io
import os
import re
import socket
import sys
import tempfile
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


DIRECTIVE = re.compile(r"(?:<!--\s*check-docs:(.*?)-->\s*)?```[a-z]*\n(.*?)```", re.S)


def parse_directives(text):
    """{'continue': bool, 'requires': [packages], 'skip': reason or None}."""
    d = {"continue": False, "requires": [], "skip": None}
    text = (text or "").strip()
    m = re.search(r"(?:^|;)\s*skip\b", text)
    if m:                                    # the reason runs to the end and may contain ';'
        d["skip"] = text[m.end():].lstrip(": ").strip()
        if not d["skip"]:
            raise ValueError("check-docs: skip needs a reason")
        text = text[:m.start()]
    for part in text.split(";"):
        part = part.strip()
        if part == "continue":
            d["continue"] = True
        elif part.startswith("requires"):
            d["requires"] += part.split()[1:]
        elif part:
            raise ValueError(f"unknown check-docs directive: {part!r}")
    return d


def blocks(text):
    """(kind, body, directives) for every runnable fenced block: kind is 'single' or 'pair'."""
    for m in DIRECTIVE.finditer(text):
        body = m.group(2)
        first = body.lstrip()
        d = parse_directives(m.group(1))
        if first.startswith("?> ") and (d["skip"] or d["continue"] or d["requires"]
                                        or not re.search(SKIP_SINGLE, body)):
            yield "single", body, d
        elif PAIR.match(first):
            yield "pair", body, d


def importable(pkg):
    return importlib.util.find_spec(pkg) is not None


def parse(body, prefixes):
    """[[who, input, [expected lines]]]; ``who`` is None for the single form."""
    steps = []
    for line in body.splitlines():
        for who, prefix in prefixes.items():
            if line.startswith(prefix):
                steps.append([who, line[len(prefix):], []])
                break
        else:
            if steps and not steps[-1][2] and line[:3] == ".. ":
                steps[-1][1] += "\n" + line[3:]      # continuation line of a multi-line entry
            elif steps:
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


def run_step(klong, expr, buf, settle=0.0):
    """Run one input; the lines it printed, then its value (or ``error: ...``).

    ``settle`` waits that long first so that callbacks started by the input have printed."""
    buf.seek(0)
    buf.truncate()
    try:
        r = klong(expr)
        if r is not None:
            klong['it'] = r      # the REPL keeps the last printed value in `it`
        text = kg_write(r, klong._backend, display=False)
    except Exception as ex:
        text = "error: " + str(ex).split("\n")[0]
    if settle:
        time.sleep(settle)
    return (buf.getvalue() + text).split("\n")


class Session:
    """The interpreters one block (and the blocks that continue it) run in."""

    def __init__(self, kind):
        self.kind = kind
        self.buf = io.StringIO()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(contextlib.redirect_stdout(self.buf))
        self.tmp = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="check_docs_"))
        self.port = str(free_port())
        self.loops = []
        if kind == "single":
            klong, loops = create_repl()
            self.env = {None: klong}
            self.loops.append(loops)
        else:
            server, sloops = create_repl()
            client, cloops = create_repl()
            self.env = {"server": server, "client": client}
            self.loops += [sloops, cloops]

    def localize(self, text):
        return text.replace("/tmp/", self.tmp + "/").replace(DOC_PORT, self.port)

    def run(self, path, body, report):
        prefixes = {None: "?> "} if self.kind == "single" else {"server": "server?> ", "client": "client?> "}
        for who, expr, exp in parse(body, prefixes):
            expr = self.localize(expr)
            exp = [self.localize(line) for line in exp]
            got = run_step(self.env[who], expr, self.buf, SETTLE if self.kind == "pair" else 0.0)
            report(path, expr, exp, got)

    def close(self):
        if self.kind == "pair":
            for k in self.env.values():
                try:
                    k['.system']['closeEvent'].trigger()
                except Exception:
                    pass
        for loops in self.loops:
            cleanup_repl(loops)
        self.stack.close()


def check_text(path, text, report, skipped):
    """Run the blocks of one document. ``skipped`` collects (path, reason)."""
    session = None
    skip_chain = None
    try:
        for kind, body, d in blocks(text):
            if not d["continue"]:
                if session:
                    session.close()
                session, skip_chain = None, None
            reason = skip_chain or d["skip"]
            if not reason:
                missing = [p for p in d["requires"] if not importable(p)]
                if missing:
                    reason = "needs " + ", ".join(missing)
            if reason:
                skip_chain = reason
                skipped.append((path, reason))
                continue
            if session is None:
                session = Session(kind)
            elif session.kind != kind:
                raise ValueError(f"{path}: a continued block must have the same form as the one before it")
            session.run(path, body, report)
    finally:
        if session:
            session.close()


def main(argv):
    paths = argv or [os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))
    counts = {"checked": 0, "bad": 0}
    skipped = []

    def report(path, expr, exp, got):
        counts["checked"] += 1
        if got != exp:
            counts["bad"] += 1
            sys.__stdout__.write(f"MISMATCH {os.path.relpath(path, ROOT)}: {expr}\n  expected {exp}\n  got      {got}\n")

    for path in paths:
        with open(path) as f:
            text = f.read()
        check_text(path, text, report, skipped)
    line = f"{counts['checked']} examples checked, {counts['bad']} mismatches"
    if skipped:
        reasons = sorted({r for _, r in skipped})
        line += f"; {len(skipped)} blocks skipped ({'; '.join(reasons)})"
    print(line)
    return 1 if counts["bad"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
