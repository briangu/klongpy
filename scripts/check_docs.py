"""Check the transcript examples in the documentation.

    python scripts/check_docs.py                  # README.md, docs/*.md and docs/native/*.md of this repository
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

* a block with the fence tag ``console`` whose first line starts with ``$ `` is a shell
  transcript: each ``$ `` line is a command run with ``sh -c`` in a temporary directory,
  with ``> `` lines directly after it as the rest of the command (a here-document, say),
  and the lines up to the next ``$ `` are the output it must print, standard output and
  standard error together. ``kgpy`` runs this checkout's command line (``python -m
  klongpy.cli``); the text is compared exactly except for trailing blanks. A block that
  continues a shell block shares its directory, so files made earlier are still there.
  ``NO_COLOR`` is set. The exit status is not shown; ask for it with ``echo $?``.
  ``python`` runs the interpreter that runs this script, with this checkout on its path,
  as ``kgpy`` does.

In all forms ``/tmp/`` in a block is replaced by a fresh temporary directory
for the run, so examples that write files do not touch the real ``/tmp``, and
the port 8888 by a free port. A ``?> `` block that starts a server or opens a
client is run only when a directive on it asks for it (see ``requires``).

A comment on the line before a fenced block changes how it is run::

    <!-- check-docs: continue -->               run in the interpreters of the previous block
    <!-- check-docs: requires duckdb pandas --> run only if these packages import, else skip
    <!-- check-docs: requires-cmd clang -->      run only if these commands are on the PATH, else skip
    <!-- check-docs: requires-file PATH -->      run only if this path of the repository exists, else skip (a built binary, say)
    <!-- check-docs: link examples -->           (console blocks) link these paths of the repository into the directory
    <!-- check-docs: path native/target/release -->  (console blocks) put these directories of the repository on the PATH
    <!-- check-docs: strict 2 -->                run in an interpreter at strict level 2 (0, 1 or 2)
    <!-- check-docs: render -->                  show a failed input as the REPL prints it on stderr
    <!-- check-docs: skip: reason -->           do not run; the reason is required

Directives combine with ``;`` (``continue; requires duckdb``); a ``skip`` reason runs to the end of the comment. ``requires`` and
``requires-cmd`` on a block apply to the blocks that continue it. ``requires-cmd`` names programs (a C compiler, say) that
must be found on the ``PATH``; the block is skipped, with ``needs clang`` as the reason, where one is missing. ``requires-file``
names paths relative to the repository root (``native/target/release/klong``, a binary that has to be built first); the
block is skipped, with ``needs native/target/release/klong`` as the reason, where one does not exist. ``path`` puts the named
directories of the repository, in front of the ``PATH`` of a console block's commands, so that ``klong`` there is the
one that was built; it is given on the block that starts the session, is an error on any other form, and the directory has
to exist unless the block is skipped. ``link`` makes a
console block's directory contain a symbolic link to each named path of the repository (``examples`` makes
``examples/embedded/mavg.kg`` readable by the commands, so a page can use the paths that a reader's checkout has); it is given
on the block that starts the session and is an error on any other form. ``strict N`` sets the strict level of the interpreters the block creates, so a
page can show that its examples are accepted at level 2 without a ``.strict(2)`` line in the transcript; it applies to the
blocks that continue it, and a continued block may not give a different level. ``render`` (for this block only; repeat it
on a continued block) shows an error as the REPL prints it, the position line, the source line, the carets and the ``hint:``
line, instead of ``error: <first line of the message>``; the position starts at ``line:column`` as at the REPL. Use ``skip`` only for an
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
import random
import re
import shutil
import socket
import subprocess
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


DIRECTIVE = re.compile(r"(?:<!--\s*check-docs:(.*?)-->\s*)?```([a-z]*)\n(.*?)```", re.S)


def parse_directives(text):
    """{'continue': bool, 'requires': [packages], 'skip': reason or None}."""
    d = {"continue": False, "requires": [], "requires_cmd": [], "requires_file": [], "link": [], "path": [],
         "skip": None, "strict": None, "render": False}
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
        elif part.split()[:1] == ["requires-cmd"]:
            d["requires_cmd"] += part.split()[1:]
        elif part.split()[:1] == ["requires-file"]:
            d["requires_file"] += part.split()[1:]
        elif part.split()[:1] == ["path"]:
            d["path"] += part.split()[1:]
        elif part.split()[:1] == ["requires"]:
            d["requires"] += part.split()[1:]
        elif part.split()[:1] == ["link"]:
            d["link"] += part.split()[1:]
        elif part == "render":
            d["render"] = True
        elif part.split()[:1] == ["strict"]:
            words = part.split()
            if len(words) != 2 or words[1] not in ("0", "1", "2"):
                raise ValueError(f"check-docs: strict needs a level 0, 1 or 2, got {part!r}")
            d["strict"] = int(words[1])
        elif part:
            raise ValueError(f"unknown check-docs directive: {part!r}")
    return d


def blocks(text):
    """(kind, body, directives) for every runnable fenced block: kind is 'single', 'pair' or 'shell'."""
    for m in DIRECTIVE.finditer(text):
        body = m.group(3)
        first = body.lstrip()
        d = parse_directives(m.group(1))
        if m.group(2) == "console" and body.startswith("$ "):
            yield "shell", body, d
        elif first.startswith("?> ") and (d["skip"] or d["continue"] or d["requires"] or d["requires_cmd"]
                                        or d["requires_file"] or not re.search(SKIP_SINGLE, body)):
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


def parse_shell(body):
    """[[command, [expected lines]]]: ``$ `` starts a command, ``> `` lines directly after it continue it."""
    steps = []
    for line in body.splitlines():
        if line.startswith("$ "):
            steps.append([line[2:], []])
        elif steps and not steps[-1][1] and line.startswith("> "):
            steps[-1][0] += "\n" + line[2:]
        elif steps:
            steps[-1][1].append(line)
    for st in steps:
        while st[1] and st[1][-1] == "":
            st[1].pop()
    return steps


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def free_port_block(count=16, low=20000, high=30000):
    """The first of ``count`` consecutive free ports, below the ranges operating systems hand out to
    outgoing connections (macOS from 49152, Linux from 32768).

    The pages use the port 8888 and the ones above it (``8888+10``). Taken from the ephemeral range,
    as ``free_port`` does, those ports are the source ports of the sockets that the page's own clients
    and servers open: a client that connects to ``localhost:PORT`` (the IPv6 address first, where nothing
    listens) can be given PORT as its own port and connect to itself, and read its own greeting as the
    server's reply (one run in several, on a busy machine)."""
    for _ in range(200):
        base = random.randrange(low, high - count)
        socks = []
        try:
            for p in range(base, base + count):
                s = socket.socket()
                socks.append(s)
                s.bind(("127.0.0.1", p))
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
        return base
    return free_port()


def run_step(klong, expr, buf, settle=0.0, render=False):
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
        if render and hasattr(ex, "render"):
            text = ex.render().replace("<string>:", "", 1)
        else:
            text = "error: " + str(ex).split("\n")[0]
    if settle:
        time.sleep(settle)
    return (buf.getvalue() + text).split("\n")


class Session:
    """The interpreters one block (and the blocks that continue it) run in."""

    def __init__(self, kind, strict=0, link=(), path=()):
        self.kind = kind
        self.path = []
        self.strict = strict
        self.buf = io.StringIO()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(contextlib.redirect_stdout(self.buf))
        self.tmp = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="check_docs_"))
        self.port = str(free_port_block())
        self.loops = []
        self.bin = None
        if kind == "shell":
            self.bin = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="check_docs_bin_"))
            shim = os.path.join(self.bin, "kgpy")
            with open(shim, "w") as f:
                f.write(f'#!/bin/sh\nexec "{sys.executable}" -m klongpy.cli "$@"\n')
            os.chmod(shim, 0o755)
            py = os.path.join(self.bin, "python")
            with open(py, "w") as f:
                f.write(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
            os.chmod(py, 0o755)
            for name in path:
                target = os.path.join(ROOT, name)
                if not os.path.isdir(target) or os.path.isabs(name) or ".." in name.split("/"):
                    raise ValueError(f"check-docs: cannot put {name!r} on the PATH: not a directory of the repository")
                self.path.append(target)
            for name in link:
                target = os.path.join(ROOT, name)
                if not os.path.exists(target) or os.path.isabs(name) or ".." in name.split("/"):
                    raise ValueError(f"check-docs: cannot link {name!r}: not a path of the repository")
                os.makedirs(os.path.dirname(os.path.join(self.tmp, name)), exist_ok=True)
                os.symlink(target, os.path.join(self.tmp, name))
        elif kind == "single":
            klong, loops = create_repl(strict=strict)
            self.env = {None: klong}
            self.loops.append(loops)
        else:
            server, sloops = create_repl(strict=strict)
            client, cloops = create_repl(strict=strict)
            self.env = {"server": server, "client": client}
            self.loops += [sloops, cloops]

    def localize(self, text):
        return text.replace("/tmp/", self.tmp + "/").replace(DOC_PORT, self.port)

    def run_shell(self, path, body, report):
        env = dict(os.environ, NO_COLOR="1", PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   PATH=os.pathsep.join(self.path + [self.bin, os.environ.get("PATH", "")]), KGPY_HISTORY="")
        for cmd, exp in parse_shell(body):
            cmd = self.localize(cmd)
            exp = [self.localize(line).rstrip() for line in exp]
            r = subprocess.run(["sh", "-c", cmd], cwd=self.tmp, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
            got = [line.rstrip() for line in r.stdout.rstrip("\n").split("\n")] if r.stdout.strip() else []
            report(path, cmd, exp, got)

    def run(self, path, body, report, render=False):
        if self.kind == "shell":
            return self.run_shell(path, body, report)
        prefixes = {None: "?> "} if self.kind == "single" else {"server": "server?> ", "client": "client?> "}
        for who, expr, exp in parse(body, prefixes):
            expr = self.localize(expr)
            exp = [self.localize(line) for line in exp]
            got = run_step(self.env[who], expr, self.buf, SETTLE if self.kind == "pair" else 0.0, render)
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
                missing += [c for c in d["requires_cmd"] if shutil.which(c) is None]
                missing += [f for f in d["requires_file"] if not os.path.exists(os.path.join(ROOT, f))]
                if missing:
                    reason = "needs " + ", ".join(missing)
            if (d["link"] or d["path"]) and kind != "shell":
                raise ValueError(f"{path}: link and path apply to console blocks only")
            if reason:
                skip_chain = reason
                skipped.append((path, reason))
                continue
            if session is None:
                session = Session(kind, d["strict"] or 0, d["link"], d["path"])
            elif session.kind != kind:
                raise ValueError(f"{path}: a continued block must have the same form as the one before it")
            elif d["strict"] is not None and d["strict"] != session.strict:
                raise ValueError(f"{path}: a continued block cannot change the strict level")
            session.run(path, body, report, d["render"])
    finally:
        if session:
            session.close()


def main(argv):
    paths = argv or ([os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))
                     + sorted(glob.glob(os.path.join(ROOT, "docs", "native", "*.md"))))
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
