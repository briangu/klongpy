"""
Reading what Vitis HLS writes: ``<top>_csynth.xml`` (estimates of timing, latency, interval and
resources, one file per module) and the messages of its log. Standard library only.
"""
import glob
import os
import re
import xml.etree.ElementTree as ET

__all__ = ["parse_csynth_xml", "parse_synth_dir", "warnings_of", "warning_classes", "format_table", "record_interval",
           "stream_summary", "ii_warnings"]

_WARN = re.compile(r"^(WARNING|ERROR|CRITICAL WARNING): \[([A-Z]+ [0-9]+-[0-9]+)\] (.*)$")


def _num(text):
    """An int when the text is one, else None ("undef", "?", "")."""
    try:
        return int(text.strip())
    except (AttributeError, ValueError):
        return None


def _text(node, path):
    n = node.find(path)
    return n.text.strip() if n is not None and n.text else None


def parse_csynth_xml(path):
    """
    The estimates of one module: {module, part, target_ns, estimated_ns, fmax_mhz, latency_best, latency_worst,
    interval_min, interval_max, bram, dsp, ff, lut, uram, available: {...}}. A number the tool does not
    know (a loop with a run-time bound) is None.
    """
    root = ET.parse(path).getroot()
    est = _text(root, "PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")
    est_ns = float(est) if est and re.match(r"[0-9.]+\Z", est) else None
    res = {k: _num(_text(root, f"AreaEstimates/Resources/{k}")) for k in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")}
    avail = {k: _num(_text(root, f"AreaEstimates/AvailableResources/{k}")) for k in ("BRAM_18K", "DSP", "FF", "LUT", "URAM")}
    lat = "PerformanceEstimates/SummaryOfOverallLatency/"
    loops = []
    sl = root.find("PerformanceEstimates/SummaryOfLoopLatency")
    if sl is not None:
        for node in sl:
            loops.append({"name": node.tag, "ii": _num(_text(node, "PipelineII")), "depth": _num(_text(node, "PipelineDepth")),
                          "trip": _num(_text(node, "TripCount")), "latency": _num(_text(node, "Latency"))})
    return {
        "loops": loops,
        "module": _text(root, "UserAssignments/TopModelName"),
        "part": _text(root, "UserAssignments/Part"),
        "target_ns": float(_text(root, "UserAssignments/TargetClockPeriod") or 0) or None,
        "estimated_ns": est_ns,
        "fmax_mhz": round(1000.0 / est_ns, 1) if est_ns else None,
        "latency_best": _num(_text(root, lat + "Best-caseLatency")),
        "latency_worst": _num(_text(root, lat + "Worst-caseLatency")),
        "interval_min": _num(_text(root, lat + "Interval-min")),
        "interval_max": _num(_text(root, lat + "Interval-max")),
        "bram": res["BRAM_18K"], "dsp": res["DSP"], "ff": res["FF"], "lut": res["LUT"], "uram": res["URAM"],
        "available": avail,
    }


def parse_synth_dir(directory, top):
    """The estimates of a top function and of its modules, from <dir>/proj_*/sol1/syn/report/*_csynth.xml."""
    found = {}
    for p in glob.glob(os.path.join(directory, "proj_*", "sol1", "syn", "report", "*_csynth.xml")):
        d = parse_csynth_xml(p)
        found[d["module"]] = d
    return found.get(top), {k: v for k, v in found.items() if k != top}


def warnings_of(log_text):
    """[(severity, code, message)] of the tool's messages, each distinct line once, in order of first appearance."""
    seen, out = set(), []
    for line in log_text.splitlines():
        m = _WARN.match(line.strip())
        if m and line not in seen:
            seen.add(line)
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def warning_classes(log_text):
    """{code: (severity, count, first message)} over the log."""
    out = {}
    for sev, code, msg in warnings_of(log_text):
        if code in out:
            s, n, first = out[code]
            out[code] = (s, n + 1, first)
        else:
            out[code] = (sev, 1, msg)
    return out


def record_interval(top, modules):
    """
    The interval between two records in steady state: that of the dataflow region of the per-record loop when
    the tool made one (its slowest process), else that of the top function. (None, "") when unknown.
    """
    for name in sorted(modules):
        if name.startswith("dataflow_in_loop") and modules[name]["interval_max"] is not None:
            return modules[name]["interval_max"], name
    if top and top["interval_max"] is not None:
        return top["interval_max"], top["module"]
    return None, ""


def stream_summary(modules, in_words, out_words):
    """
    The steady-state rate of a streamable kernel (kgpy fpga source, stage B), from the initiation intervals of the
    pipelined loops of its three processes: the core takes one record per ``core_ii`` cycles, the load process one
    word per ``load_ii`` (a record is in_words of them), the store one word per ``store_ii``. The kernel moves one
    record every ``interval`` cycles, the largest of the three; ``limit`` names which. None where the tool reported no
    pipelined loop (a block program).
    """
    def ii_of(prefix):
        found = [(m["module"], lp) for name, m in modules.items() if name.split("_Pipeline")[0] == prefix
                 for lp in m.get("loops", []) if lp["ii"] is not None]
        return max(found, key=lambda x: x[1]["ii"]) if found else None
    core, load, store = ii_of("core"), ii_of("load"), ii_of("store")
    parts = {"core": core[1]["ii"] if core else None, "load": load[1]["ii"] * in_words if load else None,
             "store": store[1]["ii"] * out_words if store else None}
    known = {k: v for k, v in parts.items() if v is not None}
    if not known:
        return {"interval": None, "limit": None, "core_ii": None, "load_ii": None, "store_ii": None, "parts": parts,
                "core_depth": None, "core_loop": None}
    limit = max(known, key=lambda k: known[k])
    return {"interval": known[limit], "limit": limit, "core_ii": parts["core"],
            "load_ii": load[1]["ii"] if load else None, "store_ii": store[1]["ii"] if store else None, "parts": parts,
            "core_depth": core[1]["depth"] if core else None, "core_loop": core[1]["name"] if core else None}


def ii_warnings(log_text):
    """The tool's messages about a pipeline that did not reach its target interval: [(code, text)]."""
    out, seen = [], set()
    for sev, code, msg in warnings_of(log_text):
        if code in ("HLS 200-880", "HLS 200-1470", "HLS 200-885", "HLS 200-1486") and (code, msg) not in seen:
            seen.add((code, msg))
            out.append((code, msg))
    return out


def _cell(v, unit=""):
    return "?" if v is None else f"{v}{unit}"


def format_table(entry_name, top, modules, clock_mhz, stream=None):
    """The synthesis table of one kernel as lines of text. Everything in it is an estimate of C synthesis."""
    lines = [f"kernel {entry_name}: C synthesis estimates (not timing closure; not measured on hardware)"]
    if top is None:
        lines.append("  no report was produced")
        return lines
    lines += [
        f"  part {top['part']}, target clock {clock_mhz} MHz ({top['target_ns']} ns)",
        f"  Fmax estimate      {_cell(top['fmax_mhz'], ' MHz')} (estimated period {_cell(top['estimated_ns'], ' ns')})",
        f"  latency (cycles)   {_cell(top['latency_best'])} to {_cell(top['latency_worst'])} ('?' = depends on n)",
        f"  interval (cycles)  {_cell(top['interval_min'])} to {_cell(top['interval_max'])} for a whole call of n records",
        f"  interval per record {_cell(record_interval(top, modules)[0])} cycles in steady state (module {record_interval(top, modules)[1] or '-'})",
        f"  LUT {_cell(top['lut'])}   FF {_cell(top['ff'])}   DSP {_cell(top['dsp'])}   BRAM_18K {_cell(top['bram'])}"
        f"   URAM {_cell(top['uram'])}",
    ]
    if stream is not None:
        sm = stream_summary(modules, *stream)
        if sm["interval"] is None:
            lines.append("  streaming rate     no pipelined core loop: the interval of the whole call is the one above")
        else:
            lines.append(f"  streaming rate     one record every {sm['interval']} cycle(s), limited by the {sm['limit']} process "
                         f"(core II {_cell(sm['core_ii'])}, load II {_cell(sm['load_ii'])} per word of {stream[0]}, "
                         f"store II {_cell(sm['store_ii'])} per word of {stream[1]}; core pipeline depth {_cell(sm['core_depth'])})")
    if modules:
        lines.append("  modules:")
        for name in sorted(modules):
            m = modules[name]
            lines.append(f"    {name}: latency {_cell(m['latency_best'])}..{_cell(m['latency_worst'])}, interval "
                         f"{_cell(m['interval_min'])}..{_cell(m['interval_max'])}, LUT {_cell(m['lut'])}, FF {_cell(m['ff'])}, "
                         f"DSP {_cell(m['dsp'])}, BRAM_18K {_cell(m['bram'])}")
    return lines
