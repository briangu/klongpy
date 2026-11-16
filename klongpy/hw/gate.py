"""
The timing gate and build.json of a bitstream build.

Standard library only and Python 3.8 syntax: this file is copied to the build machine and run there at the end of a
build (``python3 gate.py BUILD_DIR``), and the tests import it here. The rule (the same as the owner's link script):
an image is usable only when the routed timing summary says "All user specified timing constraints are met." and
its worst negative slack (WNS, setup) is not negative. Otherwise the image is renamed ``*.timing-failed.xclbin``,
build.json records the failure, and the loader (server.py) refuses both the name and any image without a passing
build.json.

    python3 gate.py BUILD_DIR  [--xclbin NAME] [--key K] [--sources-sha S] [--clock-mhz N] [--wall S] [--kernels JSON]
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

__all__ = ["parse_timing_summary", "find_timing_report", "gate", "parse_utilization", "parse_clock", "FAILED_SUFFIX",
           "passes", "image_ok"]

FAILED_SUFFIX = ".timing-failed.xclbin"
MET = "All user specified timing constraints are met."
_COLUMNS = ("wns", "tns", "tns_failing", "tns_total", "whs", "ths", "ths_failing", "ths_total",
            "wpws", "tpws", "tpws_failing", "tpws_total")


def parse_timing_summary(text):
    """
    The "Design Timing Summary" of a Vivado report_timing_summary file: {met, wns, tns, tns_failing, whs, ths,
    wpws, ...} (numbers as floats or ints; None when missing). ``met`` is True only when the tool's own sentence
    says so.
    """
    lines = text.splitlines()
    out = {k: None for k in _COLUMNS}
    out["met"] = MET in text
    start = None
    for i, line in enumerate(lines):
        if "Design Timing Summary" in line:
            start = i
            break
    if start is None:
        return out
    for i in range(start, len(lines)):
        if "WNS(ns)" not in lines[i]:
            continue
        for nxt in lines[i + 1:i + 6]:
            tokens = nxt.split()
            if not tokens or set(tokens[0]) <= set("-"):
                continue
            try:
                vals = [float(t) for t in tokens]
            except ValueError:
                continue
            for k, v in zip(_COLUMNS, vals):
                out[k] = int(v) if k.endswith(("failing", "total")) else v
            return out
        break
    return out


def find_timing_report(build_dir):
    """The routed timing summary of a v++ link: the file in _x/reports/link/imp (a list is possible: the first that exists)."""
    pats = ["_x/reports/link/imp/*timing_summary_routed.rpt",
            "_x/link/vivado/vpl/prj/prj.runs/impl_1/*timing_summary_routed.rpt"]
    for p in pats:
        found = sorted(glob.glob(os.path.join(build_dir, p)))
        if found:
            return found[0]
    return None


def parse_utilization(text):
    """The totals of a routed utilization report (the whole device image, shell included): {name: (used, available)}."""
    want = {"CLB LUTs": "lut", "CLB Registers": "ff", "DSPs": "dsp", "Block RAM Tile": "bram_tiles", "URAM": "uram"}
    out = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|\n ").split("|")]
        if len(cells) >= 5 and cells[0] in want and want[cells[0]] not in out:
            try:
                out[want[cells[0]]] = {"used": float(cells[1]), "available": float(cells[-2]), "percent": float(cells[-1])}
            except ValueError:
                pass
    return out


def parse_clock(info_text):
    """The kernel (data) clock in MHz from ``xclbinutil --info``: the DATA_CLK frequency, else None."""
    m = re.search(r"Name:\s+DATA_CLK\s+Index:\s+\d+\s+Type:\s+DATA\s+Frequency:\s+([0-9.]+) MHz", info_text)
    return float(m.group(1)) if m else None


def passes(summary):
    """True when a parsed timing summary is a pass: the tool says met and the worst setup slack is not negative."""
    return bool(summary and summary.get("met") and summary.get("wns") is not None and summary["wns"] >= 0)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_ok(xclbin):
    """
    Whether an image may be loaded: (True, build.json dict) or (False, reason). The name must not be a timing-failed
    one, build.json must sit next to it, record a pass of the routed timing, and name this very file by its SHA-256.
    """
    if xclbin.endswith(FAILED_SUFFIX) or "timing-failed" in os.path.basename(xclbin):
        return False, "the image failed routed timing (its name says so); it is never loaded"
    if not xclbin.endswith(".xclbin") or not os.path.isfile(xclbin):
        return False, "not an .xclbin file"
    meta_path = os.path.join(os.path.dirname(xclbin), "build.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False, "no readable build.json next to the image, so no recorded timing pass"
    if meta.get("target") == "hw_emu":
        if os.environ.get("XCL_EMULATION_MODE") != "hw_emu":
            return False, "a hardware emulation image: only with XCL_EMULATION_MODE=hw_emu, never on the card"
        if meta.get("xclbin") != os.path.basename(xclbin) or meta.get("xclbin_sha256") != _sha256(xclbin):
            return False, "build.json does not describe this emulation image"
        return True, meta
    elif os.environ.get("XCL_EMULATION_MODE"):
        return False, "XCL_EMULATION_MODE is set: a hardware image is not loaded in emulation"
    t = meta.get("timing") or {}
    if not (t.get("met") is True and isinstance(t.get("wns_ns"), (int, float)) and t["wns_ns"] >= 0):
        return False, "build.json does not record a pass of the routed timing (wns_ns >= 0 and met)"
    if meta.get("xclbin") != os.path.basename(xclbin):
        return False, "build.json describes %s, not this file" % meta.get("xclbin")
    if meta.get("xclbin_sha256") != _sha256(xclbin):
        return False, "the SHA-256 of the image differs from build.json (the file changed)"
    return True, meta


def gate(build_dir, xclbin_name, info=None):
    """
    Apply the gate to a finished link in build_dir. Renames a failing image to ``<stem>.timing-failed.xclbin``.
    Returns (ok, timing dict, image name). ``timing`` has met, wns_ns, tns_ns, whs_ns, failing_endpoints, report.
    """
    xclbin = os.path.join(build_dir, xclbin_name)
    rpt = find_timing_report(build_dir)
    summary = None
    if rpt:
        with open(rpt, errors="replace") as f:
            summary = parse_timing_summary(f.read())
    ok = passes(summary)
    timing = {"report": os.path.relpath(rpt, build_dir) if rpt else None, "met": bool(summary and summary["met"]),
              "wns_ns": summary and summary["wns"], "tns_ns": summary and summary["tns"],
              "whs_ns": summary and summary["whs"], "failing_endpoints": summary and summary["tns_failing"]}
    name = xclbin_name
    if not ok and os.path.exists(xclbin):
        name = xclbin_name[:-len(".xclbin")] + FAILED_SUFFIX
        os.replace(xclbin, os.path.join(build_dir, name))
    return ok, timing, name


def main(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("build_dir")
    p.add_argument("--xclbin", default=None)
    p.add_argument("--key", default="")
    p.add_argument("--sources-sha", default="")
    p.add_argument("--clock-mhz", type=float, default=0)
    p.add_argument("--wall", type=float, default=0.0, help="seconds the build took")
    p.add_argument("--kernels", default=None, help="a JSON file: the manifests of the programs the image holds")
    p.add_argument("--target", default="hw")
    a = p.parse_args(argv)
    d = os.path.abspath(a.build_dir)
    found = sorted(glob.glob(os.path.join(d, "*.xclbin")))
    name = a.xclbin or (os.path.basename(found[0]) if found else None)
    if not name or not os.path.exists(os.path.join(d, name)):
        print("gate: no image was produced", file=sys.stderr)
        meta = {"format": 1, "key": a.key, "passed": False, "reason": "no image", "wall_seconds": a.wall}
        with open(os.path.join(d, "build.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return 3
    kernels = []
    if a.kernels and os.path.exists(a.kernels):
        with open(a.kernels) as f:
            kernels = json.load(f)
    if a.target == "hw_emu":
        meta = {"format": 1, "key": a.key, "target": "hw_emu", "passed": False, "emulation": True, "xclbin": name,
                "xclbin_sha256": _sha256(os.path.join(d, name)), "timing": {"met": None, "wns_ns": None},
                "clock_mhz_requested": a.clock_mhz, "wall_seconds": a.wall, "kernels": kernels,
                "note": "a hardware emulation image: it never goes on the card"}
        with open(os.path.join(d, "build.json"), "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        print("hardware emulation image %s built (no timing gate: it is not loaded on the card)" % name, flush=True)
        return 0
    ok, timing, final = gate(d, name)
    info_text = ""
    try:
        info_text = subprocess.run(["/opt/xilinx/xrt/bin/xclbinutil", "--info", "--input", os.path.join(d, final)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
        with open(os.path.join(d, "xclbin-info.txt"), "w") as f:
            f.write(info_text)
    except OSError:
        pass
    util = {}
    for u in sorted(glob.glob(os.path.join(d, "_x/reports/link/imp/*full_util_routed.rpt"))):
        with open(u, errors="replace") as f:
            util = parse_utilization(f.read())
        break
    versions = {}
    for cmd, key in ((["v++", "--version"], "vpp"), (["vitis_hls", "-version"], "vitis_hls")):
        try:
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True).stdout
            versions[key] = out.strip().splitlines()[0] if out.strip() else ""
        except OSError:
            versions[key] = ""
    meta = {
        "format": 1, "key": a.key, "sources_sha256": a.sources_sha, "target": a.target,
        "passed": ok, "xclbin": final, "xclbin_sha256": _sha256(os.path.join(d, final)),
        "clock_mhz_requested": a.clock_mhz, "clock_mhz_image": parse_clock(info_text),
        "timing": timing, "utilization": util, "tools": versions,
        "wall_seconds": a.wall, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kernels": kernels,
    }
    with open(os.path.join(d, "build.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    t = timing
    line = "timing: WNS %s ns, TNS %s ns, WHS %s ns, met=%s, clock %s MHz -> %s" % (
        t["wns_ns"], t["tns_ns"], t["whs_ns"], t["met"], meta["clock_mhz_image"] or a.clock_mhz,
        "PASS" if ok else "FAILED (image renamed %s)" % final)
    print(line, flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
