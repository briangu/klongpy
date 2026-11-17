"""
``kgpy --hw-build FILE...``: a bitstream for the Alveo U250 from static Klong programs, cached by source.

The work happens on the build machine ($KLONG_HW_BUILD, ``user@host``) under ``<root>/builds/<key>/``:

    src/<program>/   the files of ``--emit-hls`` (C, top function, test bench, tcl)
    xo/              one Vitis kernel object per entry (Vitis HLS, ``KG_EXPORT=1``)
    link.cfg         connects every kernel's m_axi ports to DDR[0]
    build.sh         the whole flow; started detached (nohup, setsid), so a dropped ssh session changes nothing
    build.log, phase, build.pid, started, exit-code     progress, read by ``--status`` and ``--wait``
    <image>.xclbin   the image (renamed ``*.timing-failed.xclbin`` when routed timing is not met)
    build.json       key, source hash, tools, clock, WNS, utilisation, wall time, SHA-256 (written by gate.py)

The key is the cache key of ticket 110 (one program) or the hash of the keys of the programs (several programs
in one image). A finished passing build is reused; ``--force`` builds again. One build runs at a time on the
machine: ``--hw-build`` refuses while another of ours runs or any Vivado or v++ process runs, and the script
checks again when it starts. A failing image is never retried at another clock: the tool prints the clock
that the slack suggests, and the next build is the caller's decision.
"""
import hashlib
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import time

from .remote import REMOTE_ROOT, Remote, RemoteError, _tar, clean_stderr

__all__ = ["Plan", "Builder", "plan_for", "find_image", "StaleImage", "hw_build_main", "suggest_clock", "format_result", "VITIS_SETTINGS", "BUSY"]

VITIS_SETTINGS = "/home/brian/tools/Xilinx/Vitis/2022.2/settings64.sh"
PLATFORM_REPO_PATHS = "/home/brian/tools/u250-platform/opt/xilinx/platforms"
XRT_DIR = "/opt/xilinx/xrt"
BUSY = 75
_KEY = re.compile(r"[0-9a-f]{64}(-hw_emu)?\Z")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_HERE = os.path.dirname(os.path.abspath(__file__))


class Plan:
    """What one ``--hw-build`` builds: the HLS outputs of the programs, the key, the image name."""
    def __init__(self, outputs, clock_mhz, target="hw", lanes=1):
        from ..static.emit_hls import PLATFORM
        self.outputs = outputs
        self.clock_mhz = clock_mhz
        self.target = target
        self.lanes = lanes
        self.platform = PLATFORM
        keys = [o.manifest["cache_key"] for o in outputs]
        self.program_keys = keys
        self.key = keys[0] if len(keys) == 1 else hashlib.sha256("\n".join(keys).encode()).hexdigest()
        self.name = outputs[0].prefix if len(outputs) == 1 else "klong_" + self.key[:12]
        self.dirname = self.key + ("-hw_emu" if target == "hw_emu" else "")
        h = hashlib.sha256()
        for o in outputs:
            for n in sorted(o.files):
                h.update(n.encode() + b"\0" + o.files[n].encode() + b"\0")
        self.sources_sha = h.hexdigest()
        prefixes = [o.prefix for o in outputs]
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("two programs have the prefix '%s': give each file a different name" % prefixes[0])
        tops = [e["top"] for o in outputs for e in o.entries]
        if len(set(tops)) != len(tops):
            raise ValueError("two kernels have the same name: " + ", ".join(sorted(t for t in tops if tops.count(t) > 1)))

    @property
    def kernels(self):
        return [(o.prefix, e) for o in self.outputs for e in o.entries]

    @property
    def image(self):
        return self.name + (".hw_emu" if self.target == "hw_emu" else "") + ".xclbin"

    def manifests(self):
        return [o.manifest for o in self.outputs]


class StaleImage(RemoteError):
    """No passing image exists for this source: the message says the command that builds it."""


def plan_for(files, int_bits=32, clock_mhz=300, target="hw", wrapped=False, lanes=None, reassociate=False):
    """The Plan (key, image name) of the programs ``files``, without building anything."""
    from ..static.compile import emit_hls_program
    return Plan([emit_hls_program(f, None, int_bits=int_bits, clock_mhz=clock_mhz, wrapped=wrapped, lanes=lanes,
                                  reassociate=reassociate) for f in files], clock_mhz, target=target, lanes=lanes or 1)


def find_image(files, host="env", int_bits=32, clock_mhz=300, lanes=None):
    """
    The image built from exactly this source (by key) on ``host`` (default $KLONG_HW_HOST; the card host is the build
    machine today): (path of the .xclbin, build.json dict). Raises StaleImage, with the build command, when there is none,
    when the build failed timing, or when it is not finished; an image of other source is never returned.
    """
    from .client import host_target
    target = host_target() if host == "env" else host
    lanes = lanes if lanes and lanes > 1 else None
    plan = plan_for(files, int_bits, clock_mhz, lanes=lanes)
    cmd = "kgpy --hw-build %s%s%s" % (" ".join(files), "" if clock_mhz == 300 else " --clock-mhz %d" % clock_mhz,
                                      " --lanes %d" % lanes if lanes else "")
    b = Builder(Remote(target=target))
    st = b.state(plan.dirname)
    meta = st.get("build") or {}
    if st["state"] == "running":
        raise StaleImage("the image for this source (key %s) is still being built; wait: kgpy --hw-build --wait" % plan.key[:16])
    if st["state"] != "passed" or not meta.get("passed"):
        why = "failed routed timing" if st["state"] == "failed" else "does not exist"
        raise StaleImage("no usable image for this source (key %s): the build %s. Build it with: %s" % (plan.key[:16], why, cmd))
    return "%s/%s" % (st["dir"], meta["xclbin"]), meta


def suggest_clock(clock_mhz, wns_ns):
    """The clock a failed image's slack points to: Fmax = 1 / (T - WNS) with a 5 percent margin, in whole 5 MHz steps."""
    period = 1000.0 / clock_mhz
    fmax = 1000.0 / (period - wns_ns) if period - wns_ns > 0 else clock_mhz
    return max(5, int(fmax * 0.95 // 5 * 5))


class Builder:
    """The build machine side. ``remote`` is a Remote (default: $KLONG_HW_BUILD); ``root`` its fixed root."""
    def __init__(self, remote=None, vitis_settings=VITIS_SETTINGS, platform_repo=PLATFORM_REPO_PATHS, xrt=XRT_DIR):
        self.remote = remote or Remote()
        self.root = self.remote.root
        self.vitis_settings = vitis_settings
        self.platform_repo = platform_repo
        self.xrt = xrt

    # -- plumbing ---------------------------------------------------------------------
    def sh(self, script, timeout=120):
        p = self.remote._sh(script, timeout=timeout)
        return p.returncode, p.stdout.decode(errors="replace"), clean_stderr(p.stderr.decode(errors="replace"))

    def dir_of(self, dirname):
        if not _KEY.match(dirname):
            raise RemoteError("'%s' is not a build directory name" % dirname)
        return "%s/builds/%s" % (self.root, dirname)

    # -- the script -------------------------------------------------------------------
    def script(self, plan):
        q = shlex.quote
        B = self.dir_of(plan.dirname)
        L = ["#!/usr/bin/env bash", "# generated by kgpy --hw-build; do not edit", "set -u",
             "ROOT=" + q(self.root), "B=" + q(B), 'cd "$B" || exit 2', "echo $$ > build.pid", "date +%s > started",
             "rm -f exit-code", "mkdir -p logs xo",
             "log() { echo \"== $(date '+%F %T') $*\"; }",
             'phase() { echo "$1" > phase; log "phase: $1"; }',
             'finish() { echo "$1" > exit-code; log "finished: exit $1 after $SECONDS s"; exit "$1"; }',
             'exec 9>"$ROOT/build.lock"',
             'if ! flock -n 9; then log "refused: another build of ours holds the build lock"; finish %d; fi' % BUSY,
             # any user's Vivado or v++ (pgrep -f with the bracket trick does not match itself)
             "if pgrep -f '[v]ivado|[v]\\+\\+|[v]itis_hls' >/dev/null; then log \"refused: a Vivado, v++ or Vitis HLS process is running\"; finish %d; fi" % BUSY]
        if self.vitis_settings:
            L += ["if [ ! -f %s ]; then log 'missing %s'; finish 2; fi" % (q(self.vitis_settings), self.vitis_settings),
                  "source %s >/dev/null 2>&1" % q(self.vitis_settings)]
        L += ["export PLATFORM_REPO_PATHS=" + q(self.platform_repo), "export XILINX_XRT=" + q(self.xrt)]
        # 1: one kernel object per entry
        for prefix, e in plan.kernels:
            c = e["c_name"]
            for ident in (prefix, c, e["top"]):
                if not _IDENT.match(ident):
                    raise ValueError("'%s' is not usable as a file name in a build" % ident)
            L += ['phase "hls %s"' % e["top"],
                  "(cd src/%s && KG_EXPORT=1 vitis_hls -f %s) > logs/hls_%s.log 2>&1 || { tail -n 25 logs/hls_%s.log; finish 10; }"
                  % (prefix, e["files"]["tcl"], e["top"], e["top"]),
                  "xo=$(find src/%s/proj_%s -name '*.xo' | head -n 1)" % (prefix, c),
                  'if [ -z "$xo" ]; then log "no .xo from %s"; finish 11; fi' % e["top"],
                  'cp "$xo" xo/%s.xo' % e["top"]]
        # 2: the link
        L += ["cat > link.cfg <<'CFG'", "[connectivity]"]
        for _prefix, e in plan.kernels:
            for port in ("in", "out", "status") + (("params",) if e.get("register_table") else ()):
                L.append("sp=%s_1.%s:DDR[0]" % (e["top"], port))
        L += ["CFG"]
        if plan.target == "hw_emu":
            L += ['phase "link (v++ -l -t hw_emu, %d MHz)"' % plan.clock_mhz,
                  "v++ -l -t hw_emu --platform %s --kernel_frequency %d --save-temps --config link.cfg -o %s xo/*.xo >> build.log 2>&1"
                  % (plan.platform, plan.clock_mhz, plan.image),
                  "rc=$?",
                  'if [ $rc -ne 0 ] || [ ! -f %s ]; then log "v++ failed (exit $rc)"; finish 20; fi' % plan.image,
                  "emconfigutil --platform %s --od . >> build.log 2>&1 || log 'emconfigutil failed'" % plan.platform]
        else:
            L += ['phase "link (v++ -l -t hw, %d MHz): synthesis, place, route"' % plan.clock_mhz,
                  "v++ -l -t hw -R2 --platform %s --kernel_frequency %d --vivado.synth.jobs 8 --vivado.impl.jobs 8 "
                  "--save-temps --config link.cfg -o %s xo/*.xo >> build.log 2>&1" % (plan.platform, plan.clock_mhz, plan.image),
                  "rc=$?",
                  'log "v++ exit $rc"']
        # 3: the gate
        L += ['phase "gate"',
              "python3 gate.py \"$B\" --xclbin %s --key %s --sources-sha %s --clock-mhz %d --wall $SECONDS "
              "--kernels kernels.json --target %s >> build.log 2>&1" % (plan.image, plan.key, plan.sources_sha, plan.clock_mhz, plan.target),
              "grc=$?", 'finish "$grc"', ""]
        return "\n".join(L)

    # -- state ------------------------------------------------------------------------
    def state(self, dirname):
        """
        {state, ...} of a build directory: none; running (pid, phase, elapsed, tail); passed / failed (build.json);
        stopped (no finish marker and no process: the machine rebooted or the script was killed).
        """
        B = self.dir_of(dirname)
        script = (
            "cd %s 2>/dev/null || { echo NONE; exit 0; }\n"
            "echo PID=$(cat build.pid 2>/dev/null)\n"
            "echo STARTED=$(cat started 2>/dev/null)\n"
            "echo NOW=$(date +%%s)\n"
            "echo PHASE=$(cat phase 2>/dev/null)\n"
            "echo EXIT=$(cat exit-code 2>/dev/null)\n"
            "p=$(cat build.pid 2>/dev/null); if [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null; then echo ALIVE=1; else echo ALIVE=0; fi\n"
            "echo ---TAIL; tail -n 4 build.log 2>/dev/null\n"
            "echo ---JSON; cat build.json 2>/dev/null; exit 0\n" % shlex.quote(B))
        rc, out, err = self.sh(script)
        if rc != 0:
            raise RemoteError("cannot read the state of %s: %s" % (dirname, err.strip()))
        if out.startswith("NONE"):
            return {"state": "none", "dir": B}
        head, _, rest = out.partition("---TAIL\n")
        tail, _, js = rest.partition("---JSON\n")
        kv = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
        st = {"dir": B, "pid": kv.get("PID") or None, "phase": kv.get("PHASE", ""), "tail": tail.rstrip("\n"),
              "exit": kv.get("EXIT") or None}
        if kv.get("STARTED"):
            st["started"] = int(kv["STARTED"])
            st["elapsed"] = int(kv["NOW"]) - st["started"]
        meta = None
        if js.strip():
            try:
                meta = json.loads(js)
            except ValueError:
                meta = None
        st["build"] = meta
        if kv.get("ALIVE") == "1":
            st["state"] = "running"
        elif kv.get("EXIT") not in (None, ""):
            ok = kv["EXIT"] == "0" and meta and meta.get("passed")
            st["state"] = "passed" if ok else ("emulation" if kv["EXIT"] == "0" and meta and meta.get("emulation") else "failed")
        elif st["pid"]:
            st["state"] = "stopped"
        else:
            st["state"] = "none"
        return st

    def running(self):
        """The directory name of a running build of ours, or '' ; and whether some other Vivado or v++ process runs."""
        script = (
            "for d in %s/builds/*/; do [ -f \"$d/build.pid\" ] || continue; p=$(cat \"$d/build.pid\"); "
            "if kill -0 \"$p\" 2>/dev/null && [ ! -f \"$d/exit-code\" ]; then basename \"$d\"; fi; done; echo ---; "
            "pgrep -a -f '[v]ivado|[v]\\+\\+|[v]itis_hls' | cut -c1-120 | head -n 3" % shlex.quote(self.root))
        rc, out, err = self.sh(script)
        ours, _, others = out.partition("---\n")
        return ours.strip(), others.strip()

    def latest(self):
        rc, out, _ = self.sh("ls -1t %s/builds 2>/dev/null | head -n 1" % shlex.quote(self.root))
        return out.strip()

    # -- start ------------------------------------------------------------------------
    def start(self, plan, force=False):
        """Upload and launch (detached). Returns the state dict after the launch. Raises RemoteError when refused."""
        ours, others = self.running()
        if ours:
            raise RemoteError("a build is running (%s); wait for it: kgpy --hw-build --wait" % ours)
        if others:
            raise RemoteError("another Vivado, v++ or Vitis HLS process is running on the build machine (not started by "
                              "this tool); wait for it to finish:\n  " + others.replace("\n", "\n  "))
        B = self.dir_of(plan.dirname)
        if force:
            self.sh("rm -rf -- %s" % shlex.quote(B))
        files = {"build.sh": self.script(plan), "kernels.json": json.dumps(plan.manifests(), indent=1)}
        with open(os.path.join(_HERE, "gate.py")) as f:
            files["gate.py"] = f.read()
        for o in plan.outputs:
            for n, text in o.files.items():
                files["src/%s/%s" % (o.prefix, n)] = text
        rc, out, err = self.sh("mkdir -p %s && cd %s && ls -A | head -n 1" % (shlex.quote(B), shlex.quote(B)))
        if out.strip():
            raise RemoteError("%s is not empty: it holds an earlier build that did not pass (use --force to build again)" % B)
        p = self.remote._sh_with("tar xf - -C %s" % shlex.quote(B), _tar(files), 600)
        if p.returncode != 0:
            raise RemoteError("cannot copy the sources: " + clean_stderr(p.stderr.decode(errors="replace")).strip())
        launch = ("cd %s || exit 1\nnohup setsid bash build.sh > build.log 2>&1 < /dev/null &\n"
                  "sleep 2; cat build.pid" % shlex.quote(B))
        rc, out, err = self.sh(launch)
        if rc != 0:
            raise RemoteError("cannot start the build: " + err.strip())
        return self.state(plan.dirname)

    def wait(self, dirname, poll=60, out=None, max_errors=30):
        """Poll until the build is no longer running; prints the phase when it changes. Returns the final state."""
        out = out or sys.stdout
        last, errors = None, 0
        while True:
            try:
                st = self.state(dirname)
                errors = 0
            except (RemoteError, subprocess.TimeoutExpired) as e:
                errors += 1
                if errors > max_errors:
                    raise RemoteError("the build machine did not answer %d times in a row: %s" % (errors, e))
                time.sleep(poll)
                continue
            if st["state"] != "running":
                return st
            if st["phase"] != last:
                print("[%s] %s (%s elapsed)" % (time.strftime("%H:%M:%S"), st["phase"], _hms(st.get("elapsed", 0))), file=out, flush=True)
                last = st["phase"]
            time.sleep(poll)


def _hms(s):
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def format_result(plan_clock, st, builder=None):
    """Lines that say what a finished build is: image path, WNS and clock, and what to do after a failure."""
    meta = st.get("build") or {}
    t = meta.get("timing") or {}
    lines = []
    d = st["dir"]
    if meta.get("target") == "hw_emu":
        lines.append("hardware emulation image: %s/%s (not for the card)" % (d, meta.get("xclbin")))
        return lines
    wns = t.get("wns_ns")
    clock = meta.get("clock_mhz_image") or meta.get("clock_mhz_requested") or plan_clock
    wall = meta.get("wall_seconds")
    sline = "timing summary: WNS %s ns, TNS %s ns, WHS %s ns, failing endpoints %s, clock %s MHz (requested %s) - %s" % (
        wns, t.get("tns_ns"), t.get("whs_ns"), t.get("failing_endpoints"), clock, meta.get("clock_mhz_requested"),
        "met" if meta.get("passed") else "NOT MET")
    lines.append(sline)
    if meta.get("passed"):
        lines.append("image: %s/%s" % (d, meta.get("xclbin")))
        lines.append("sha256 %s; build wall time %s" % (meta.get("xclbin_sha256"), _hms(int(wall)) if wall else "?"))
        if meta.get("clock_mhz_image") and meta.get("clock_mhz_requested") and abs(meta["clock_mhz_image"] - meta["clock_mhz_requested"]) > 0.5:
            lines.append("note: the tool ran the kernel clock at %s MHz, not the requested %s MHz" % (meta["clock_mhz_image"], meta["clock_mhz_requested"]))
    else:
        lines.append("the image failed routed timing and is not loadable: %s/%s" % (d, meta.get("xclbin") or "(none)"))
        lines.append("build directory kept: %s" % d)
        if isinstance(wns, (int, float)) and wns < 0 and meta.get("clock_mhz_requested"):
            c = suggest_clock(meta["clock_mhz_requested"], wns)
            lines.append("worst negative slack %s ns at %s MHz; to try again: kgpy --hw-build ... --clock-mhz %d" % (wns, meta["clock_mhz_requested"], c))
        elif not meta:
            lines.append("no build.json: the flow stopped before the gate (exit %s); see %s/build.log" % (st.get("exit"), d))
    return lines


def hw_build_main(files, int_bits=32, clock_mhz=300, lanes=None, force=False, status=False, wait=False, hw_emu=False,
                  stdout=None, stderr=None, builder=None, poll=60, wrapped=False, reassociate=False):
    """
    The body of ``kgpy --hw-build [FILE...] [--status] [--wait] [--force]``. Returns 0 when the image is built (or
    cached, or the build was started and --wait was not given), 1 when the build or the timing gate failed or a build is
    refused, 2 on a usage error.
    """
    from ..errors import KlongError
    from ..static.compile import emit_hls_program
    out, err = stdout or sys.stdout, stderr or sys.stderr
    if lanes == 1 and not wrapped:
        lanes = None            # one core, no lane dimension: the kernel without --lanes
    if lanes is not None and wrapped:
        print("--lanes needs the streamable back end: it cannot be combined with --hls-wrapped", file=err)
        return 2
    try:
        b = builder or Builder()
        if b.remote.target is None and builder is None:
            print("--hw-build needs $KLONG_HW_BUILD (user@host of the build machine)", file=err)
            return 2
        b.remote.check()
        target = "hw_emu" if hw_emu else "hw"
        if not files:
            if not (status or wait):
                print("--hw-build needs FILEs (or --status, --wait for the build in progress)", file=err)
                return 2
            ours, _ = b.running()
            dirname = ours or b.latest()
            if not dirname:
                print("no build on %s" % (b.remote.target or "this machine"), file=out)
                return 0
            return _report(b, dirname, clock_mhz, wait, out, err, poll)
        outputs = []
        for path in files:
            try:
                outputs.append(emit_hls_program(path, None, int_bits=int_bits, clock_mhz=clock_mhz, wrapped=wrapped,
                                                lanes=lanes, reassociate=reassociate))
            except (OSError, UnicodeDecodeError) as e:
                print("%s: cannot read: %s" % (path, getattr(e, "strerror", None) or e), file=err)
                return 2
            except KlongError as e:
                for d in getattr(e, "diagnostics", None) or [e]:
                    print(d.render(False), file=err)
                return 1
            except ValueError as e:
                print("%s: %s" % (path, e), file=err)
                return 1
        try:
            plan = Plan(outputs, clock_mhz, target=target, lanes=lanes or 1)
        except ValueError as e:
            print(str(e), file=err)
            return 2
        st = b.state(plan.dirname)
        print("image %s, key %s, %d kernel(s): %s; clock %d MHz" % (plan.image, plan.key[:16], len(plan.kernels),
              ", ".join(e["top"] for _p, e in plan.kernels), clock_mhz), file=out)
        if status:
            return _report(b, plan.dirname, clock_mhz, wait, out, err, poll)
        if st["state"] == "running":
            print("this build is already running (pid %s, %s, %s elapsed)" % (st["pid"], st["phase"], _hms(st.get("elapsed", 0))), file=out)
            return _report(b, plan.dirname, clock_mhz, wait, out, err, poll)
        if st["state"] in ("passed", "emulation") and not force:
            print("cached: the image for this source, clock and tools exists", file=out)
            for line in format_result(clock_mhz, st):
                print(line, file=out)
            return 0
        if st["state"] == "failed" and not force:
            print("an earlier build of this key did not pass:", file=err)
            for line in format_result(clock_mhz, st):
                print(line, file=err)
            print("(--force builds it again; a different --clock-mhz is a different build)", file=err)
            return 1
        try:
            st = b.start(plan, force=force)
        except RemoteError as e:
            if "is not empty" in str(e) and st["state"] == "stopped":
                print("%s\n  an earlier build was stopped before it finished" % e, file=err)
            else:
                print("--hw-build: %s" % e, file=err)
            return 1
        print("build started on %s: pid %s, directory %s" % (b.remote.target or "this machine", st["pid"], st["dir"]), file=out)
        print("it takes hours and keeps running if this session ends: kgpy --hw-build --status, kgpy --hw-build --wait", file=out)
        if wait:
            return _report(b, plan.dirname, clock_mhz, True, out, err, poll)
        return 0
    except RemoteError as e:
        print("--hw-build: %s" % e, file=err)
        return 1


def _report(b, dirname, clock_mhz, wait, out, err, poll):
    st = b.state(dirname)
    if st["state"] == "running" and wait:
        st = b.wait(dirname, poll=poll, out=out)
    if st["state"] == "none":
        print("no build %s" % dirname[:16], file=out)
        return 0
    print("build %s: %s" % (dirname[:16], st["state"]) +
          (", pid %s, %s, %s elapsed" % (st["pid"], st["phase"], _hms(st.get("elapsed", 0))) if st["state"] == "running" else ""), file=out)
    if st["state"] == "running":
        print("log tail:\n  " + st["tail"].replace("\n", "\n  "), file=out)
        return 0
    if st["state"] == "stopped":
        print("the build stopped before it finished (phase %s); see %s/build.log" % (st["phase"], st["dir"]), file=err)
        return 1
    for line in format_result(clock_mhz, st):
        print(line, file=(out if st["state"] in ("passed", "emulation") else err))
    return 0 if st["state"] in ("passed", "emulation") else 1
