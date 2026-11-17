"""
Run a command on the build machine, or here.

    r = Remote()                          # $KLONG_HW_BUILD names the machine (user@host); unset means this machine
    res = r.run("vitis_hls -f hls_f.tcl", files={"a.c": b"...", "dir/in.bin": "/path/on/disk"},
                fetch=["data/out.bin", "proj_f/sol1/syn/report/*_csynth.xml"], timeout=1800)
    res.returncode, res.stdout, res.stderr, res.fetched (local directory), res.elapsed

Each run gets a fresh directory under the fixed root ``<root>/work/`` (the root is $KLONG_HW_ROOT, default
``~/klong-hw``, resolved on the machine; here: a temporary directory), the files are copied in, the command runs there
with the Vitis HLS settings sourced, and the files asked for are copied back. Files
travel as a tar stream over ssh (nothing but ssh and tar is needed on either side).

Rules this module keeps (docs/hardware.md):

  * nothing is created or removed outside ``<root>/work/<run directory>``; the
    only deletion is of the run directory this object created, after its name
    was checked against the fixed pattern;
  * ``vivado=True`` (a command that starts Vivado or v++) refuses to start when
    one already runs under the same user, and holds a lock file while it runs;
    Vitis HLS (C simulation, C synthesis) does not need it;
  * the command runs under ``timeout``, so nothing is left behind when the time is up
    (no process is ever signalled by name or pattern).
"""
import io
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid

__all__ = ["Remote", "Result", "RemoteError", "DEFAULT_ROOT", "root_setting", "resolve_root", "hls_settings", "clean_stderr",
           "build_target"]

DEFAULT_ROOT = "~/klong-hw"      # on the build machine: "~" is that machine's home, not ours
_RUN_DIR = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{5,80}\Z")
_TARGET = re.compile(r"[A-Za-z0-9._-]+(@[A-Za-z0-9._-]+)?\Z")
_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
_BUSY = 75              # exit status of "a Vivado or v++ run is in progress"


class RemoteError(RuntimeError):
    """The machine cannot be used (unreachable, a bad path) or a run was refused."""


class Result:
    """returncode, stdout and stderr of the command; ``fetched``: local directory with the copied-back files; ``workdir``: where it ran."""
    def __init__(self, returncode, stdout, stderr, fetched, workdir, elapsed, timed_out=False):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.fetched = fetched
        self.workdir = workdir
        self.elapsed = elapsed
        self.timed_out = timed_out

    @property
    def ok(self):
        return self.returncode == 0

    def path(self, name):
        return os.path.join(self.fetched, name)

    def read(self, name, binary=False):
        with open(self.path(name), "rb" if binary else "r") as f:
            return f.read()

    def cleanup(self):
        shutil.rmtree(self.fetched, ignore_errors=True)


def clean_stderr(text):
    """stderr of ssh without its post-quantum key exchange warning (lines that start with "** ")."""
    return "\n".join(line for line in text.splitlines() if not line.startswith("** "))


def build_target(env=None):
    """The ssh destination of the build machine from $KLONG_HW_BUILD, or None (this machine)."""
    v = (env if env is not None else os.environ).get("KLONG_HW_BUILD", "").strip()
    if not v:
        return None
    if not _TARGET.match(v):
        raise RemoteError(f"KLONG_HW_BUILD='{v}' is not a user@host destination")
    return v


def root_setting(env=None):
    """
    The fixed root on the build machine as written: $KLONG_HW_ROOT, default ``~/klong-hw``. An absolute path, or a
    path under ``~`` (the home directory of the machine it is used on; see ``resolve_root``).
    """
    v = (env if env is not None else os.environ).get("KLONG_HW_ROOT", "").strip() or DEFAULT_ROOT
    ok = (v.startswith("/") or v == "~" or v.startswith("~/")) and "\0" not in v and ".." not in v.split("/") and \
        not any(c.isspace() for c in v)
    if not ok or v.rstrip("/") in ("", "~"):
        raise RemoteError(f"KLONG_HW_ROOT='{v}' is not an absolute path (or a path under ~) without spaces or '..'")
    return v.rstrip("/")


_HOMES = {}


def resolve_root(target, spec=None):
    """The root as an absolute path on ``target`` (user@host, or None for this machine); a leading ``~`` is that machine's home."""
    spec = root_setting() if spec is None else spec
    if not spec.startswith("~"):
        return spec
    if target is None:
        home = os.path.expanduser("~")
    else:
        home = _HOMES.get(target)
        if home is None:
            try:
                p = subprocess.run(_SSH + [target, 'printf %s "$HOME"'], capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                raise RemoteError(f"{target} did not answer")
            home = p.stdout.decode(errors="replace").strip()
            if p.returncode != 0 or not home.startswith("/"):
                raise RemoteError(f"cannot find the home directory on {target} (set KLONG_HW_ROOT to an absolute path): "
                                  f"{clean_stderr(p.stderr.decode(errors='replace')).strip()}")
            _HOMES[target] = home
    return home.rstrip("/") + spec[1:]


def hls_settings(env=None):
    """The Vitis HLS settings script to source before a command ($KLONG_HW_HLS_SETTINGS), or None: the tools are on PATH."""
    return (env if env is not None else os.environ).get("KLONG_HW_HLS_SETTINGS", "").strip() or None


def _tar(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, src in sorted(files.items()):
            _check_rel(name)
            if isinstance(src, (bytes, bytearray)):
                data = bytes(src)
            elif isinstance(src, str) and not os.path.isfile(src):
                data = src.encode()          # text
            else:
                with open(src, "rb") as f:
                    data = f.read()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mode = 0o644
            ti.mtime = int(time.time())
            t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _check_rel(name):
    if not name or name.startswith("/") or ".." in name.split("/") or "\0" in name or name.startswith("-"):
        raise RemoteError(f"'{name}' is not a plain relative path")


def _untar(data, dest):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as t:
        for m in t.getmembers():
            if not m.isfile() or os.path.basename(m.name).startswith("._"):
                continue
            _check_rel(m.name)
            target = os.path.join(dest, m.name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(t.extractfile(m).read())


class Remote:
    """
    A place to run commands: ``target`` is user@host (default: $KLONG_HW_BUILD), or None for this machine.
    ``root`` is the fixed root on the machine ($KLONG_HW_ROOT, default ~/klong-hw, found on the machine when first
    needed; here, a temporary directory). ``settings`` is sourced before each command (default $KLONG_HW_HLS_SETTINGS;
    unset: Vitis HLS must be on the PATH of the machine).
    """
    def __init__(self, target="env", root=None, settings="env"):
        self.target = build_target() if target == "env" else target
        if self.target is not None and not _TARGET.match(self.target):
            raise RemoteError(f"'{self.target}' is not a user@host destination")
        if self.target is not None:
            self._spec = root_setting()
            if root not in (None, self._spec):
                raise RemoteError(f"the root on the build machine is fixed at {self._spec}")
            self._root = None
        else:
            self._root = root or os.path.join(tempfile.gettempdir(), "klong-hw")
        self.settings = hls_settings() if settings == "env" else settings

    @property
    def root(self):
        if self._root is None:
            self._root = resolve_root(self.target, self._spec)
        return self._root

    # -- helpers ----------------------------------------------------------------
    @property
    def work(self):
        return self.root + "/work"

    def _valid(self, workdir):
        return (workdir.startswith(self.work + "/") and _RUN_DIR.match(workdir[len(self.work) + 1:]) is not None)

    def _sh(self, script, stdin=b"", timeout=None):
        """Run a shell script (bash) here or over ssh; returns the CompletedProcess (bytes)."""
        data = script.encode() if isinstance(script, str) else script
        if self.target:
            return subprocess.run(_SSH + [self.target, "bash", "-s"], input=data, capture_output=True, timeout=timeout)
        # here: a process group of our own, so that a timeout stops the command and what it started (and only that)
        proc = subprocess.Popen(["bash", "-s"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
        try:
            out, err = proc.communicate(data, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            os.killpg(proc.pid, 9)      # proc.pid: the process started just above
            proc.communicate()
            raise e
        return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)

    def _sh_with(self, script, data, timeout):
        """Run `script` where the tar stream `data` is its stdin: the script is the command, data the input."""
        cmd = (_SSH + [self.target, script]) if self.target else ["bash", "-c", script]
        return subprocess.run(cmd, input=data, capture_output=True, timeout=timeout)

    def check(self):
        """Raise RemoteError unless the machine answers."""
        try:
            p = self._sh("echo ok", timeout=60)
        except subprocess.TimeoutExpired:
            raise RemoteError(f"{self.target} did not answer")
        if p.returncode != 0 or b"ok" not in p.stdout:
            raise RemoteError(f"cannot reach {self.target}: {clean_stderr(p.stderr.decode(errors='replace')).strip()}")

    def new_workdir(self):
        name = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        workdir = f"{self.work}/{name}"
        assert self._valid(workdir)
        p = self._sh(f"mkdir -p {shlex.quote(self.work)} && mkdir {shlex.quote(workdir)}", timeout=60)
        if p.returncode != 0:
            raise RemoteError(f"cannot create {workdir}: {clean_stderr(p.stderr.decode(errors='replace')).strip()}")
        return workdir

    def remove_workdir(self, workdir):
        """Remove a run directory made by new_workdir (and nothing else)."""
        if not self._valid(workdir):
            raise RemoteError(f"refusing to remove '{workdir}': not a run directory under {self.work}")
        self._sh(f"rm -rf -- {shlex.quote(workdir)}", timeout=120)

    # -- the run -------------------------------------------------------------------
    def run(self, command, files=None, fetch=(), timeout=1800, vivado=False, keep=False, env=None, settings=True):
        """
        Run ``command`` (a shell command line) in a fresh run directory after copying ``files`` there.

        files    {relative path: bytes | text | path of a local file}
        fetch    relative paths or globs of files to copy back (into a local temporary directory: Result.fetched)
        timeout  seconds for the command; when it is up the command is stopped (by timeout(1) on the machine)
        vivado   the command starts Vivado or v++: refuse if one already runs, and hold the build lock
        keep     leave the run directory (default: remove it once the files are fetched)
        env      {NAME: value} exported before the command
        """
        t0 = time.time()
        workdir = self.new_workdir()
        fetched = tempfile.mkdtemp(prefix="kghw_")
        try:
            if files:
                p = self._sh_with(f"tar xf - -C {shlex.quote(workdir)}", _tar(files), 600)
                if p.returncode != 0:
                    raise RemoteError(f"cannot copy files to {workdir}: {clean_stderr(p.stderr.decode(errors='replace')).strip()}")
            lines = ["set -u", f"cd {shlex.quote(workdir)}"]
            for k, v in (env or {}).items():
                if not re.match(r"[A-Z_][A-Z0-9_]*\Z", k):
                    raise RemoteError(f"'{k}' is not an environment variable name")
                lines.append(f"export {k}={shlex.quote(str(v))}")
            if settings and self.settings:
                lines.append(f"source {shlex.quote(self.settings)} >/dev/null 2>&1")
            kill_after = 30
            if self.target:
                inner = f"timeout --kill-after={kill_after} {int(timeout)} bash -c {shlex.quote(command)} </dev/null"
            else:
                inner = f"bash -c {shlex.quote(command)} </dev/null"      # _sh enforces the time here
            if vivado:
                lock = shlex.quote(self.root + "/build.lock")
                lines += [
                    'if pgrep -u "$(id -u)" -f \'[v]ivado|[v]\\+\\+\' >/dev/null; then',
                    '  echo "refusing to start: a Vivado or v++ run is in progress on this machine" >&2',
                    f"  exit {_BUSY}",
                    "fi",
                    f"exec 9>{lock}",
                    "if ! flock -n 9; then",
                    '  echo "refusing to start: another build of ours holds the build lock" >&2',
                    f"  exit {_BUSY}",
                    "fi",
                ]
            lines.append(inner)
            script = "\n".join(lines) + "\n"
            timed_out = False
            try:
                p = self._sh(script, timeout=(timeout + kill_after + 60) if self.target else timeout)
                rc, out, err = p.returncode, p.stdout.decode(errors="replace"), clean_stderr(p.stderr.decode(errors="replace"))
            except subprocess.TimeoutExpired as e:
                timed_out = True
                rc, out, err = 124, (e.stdout or b"").decode(errors="replace"), "timed out"
            if rc == 124:
                timed_out = True
            if fetch:
                globs = " ".join(_glob(g) for g in fetch)
                # the shell expands the globs; what matches nothing is left out (no GNU tar options: bsdtar also works here)
                p = self._sh_with(f"cd {shlex.quote(workdir)} && f=$(ls -d -- {globs} 2>/dev/null); "
                                  f'if [ -n "$f" ]; then COPYFILE_DISABLE=1 tar cf - $f; fi', b"", 600)
                if p.returncode != 0 and not p.stdout:
                    raise RemoteError(f"cannot copy files back: {clean_stderr(p.stderr.decode(errors='replace')).strip()}")
                if p.stdout:
                    _untar(p.stdout, fetched)
            return Result(rc, out, err, fetched, workdir, time.time() - t0, timed_out)
        finally:
            if not keep:
                try:
                    self.remove_workdir(workdir)
                except Exception:
                    pass


def _glob(g):
    """A path or glob made safe for the shell: everything but * ? [ ] is quoted."""
    _check_rel(g)
    out = []
    for ch in g:
        if ch in "*?[]":
            out.append(ch)
        elif re.match(r"[A-Za-z0-9_./-]", ch):
            out.append(ch)
        else:
            out.append("\\" + ch)
    return "".join(out)
