"""
``kgpy --emit-hls --synth``: run C synthesis of each kernel on the build machine ($KLONG_HW_BUILD; unset
means this machine) and print the estimates.
"""
import os

from .remote import Remote, RemoteError
from .reports import ii_warnings, format_table, parse_synth_dir, stream_summary, warning_classes

__all__ = ["synthesize", "synth_main"]


def synthesize(res, remote=None, timeout=3600, csim_data=None, chunk=0, export=False, only=None, synth=True):
    """
    Run csynth (and, with csim_data = {"in.bin": bytes}, the C simulation first) for every entry of an HlsOutput
    (only: just that entry; synth=False: the C simulation alone). In the result, out.bin and status.bin are read from
    ``files`` when the test bench ran.
    Returns {entry c_name: dict(result=Result, top=estimates, modules=..., warnings=..., log=text, files=Result.fetched)}.
    """
    remote = remote or Remote()
    out = {}
    for ent in res.entries:
        if only is not None and ent["c_name"] != only:
            continue
        files = dict(res.files)
        env = {}
        if csim_data is not None:
            files.update(csim_data)           # in.bin next to the tcl: the tcl looks there unless KG_DATA says otherwise
            env["KG_CHUNK"] = str(chunk)
        if export:
            env["KG_EXPORT"] = "1"
        if not synth:
            env["KG_NO_SYNTH"] = "1"
        r = remote.run(f"vitis_hls -f {ent['files']['tcl']}", files=files, timeout=timeout,
                       fetch=["vitis_hls.log", "out.bin", "status.bin", "proj_*/sol1/syn/report/*_csynth.xml",
                              "proj_*/sol1/syn/report/csynth.rpt"],
                       env=env)
        top, modules = parse_synth_dir(r.fetched, ent["top"])
        log = ""
        if os.path.exists(r.path("vitis_hls.log")):
            log = r.read("vitis_hls.log")
        out[ent["c_name"]] = {"result": r, "top": top, "modules": modules, "log": log, "warnings": warning_classes(log),
                              "stdout": r.stdout,
                              "out": r.read("out.bin", True) if os.path.exists(r.path("out.bin")) else None,
                              "status": r.read("status.bin", True) if os.path.exists(r.path("status.bin")) else None}
    return out


def synth_main(res, where, out, err):
    """Print the synthesis table of every kernel of an HlsOutput. Returns 0 when all completed, else 1."""
    try:
        remote = Remote()
        remote.check()
    except RemoteError as e:
        print(f"--synth: {e}", file=err)
        return 1
    print(f"running Vitis HLS 2022.2 C synthesis on {remote.target or 'this machine'} (the numbers below are estimates)", file=out)
    status = 0
    results = synthesize(res, remote)
    clock = res.manifest["clock_mhz"]
    for ent in res.entries:
        r = results[ent["c_name"]]
        if r["result"].returncode != 0 or r["top"] is None:
            print(f"kernel {ent['top']}: synthesis failed (exit {r['result'].returncode}) after {r['result'].elapsed:.0f} s", file=err)
            tail = "\n".join((r["log"] or r["stdout"]).splitlines()[-15:])
            print(tail, file=err)
            status = 1
            continue
        streamed = res.manifest.get("stage") == "B"
        for line in format_table(ent["name"], r["top"], r["modules"], clock,
                                 (ent["in_words"], ent["out_words"]) if streamed else None):
            print(line, file=out)
        if r["top"]["fmax_mhz"] is not None and r["top"]["fmax_mhz"] < clock:
            print(f"  warning: the estimated Fmax {r['top']['fmax_mhz']} MHz is below the target clock {clock} MHz: the tool "
                  "chained operations to reach its interval; ask for a slower clock (--clock-mhz) or break the long path", file=out)
        if streamed:
            sm = stream_summary(r["modules"], ent["in_words"], ent["out_words"])
            if sm["core_ii"] is not None and sm["core_ii"] > 1:
                why = "; ".join(f"[{c}] {m[:200]}" for c, m in ii_warnings(r["log"])[:2]) or "see the tool log"
                print(f"  warning: the core loop reached an interval of {sm['core_ii']} cycles, not 1: {why}", file=out)
            elif sm["interval"] is not None and sm["interval"] > 1:
                print(f"  note: one record every {sm['interval']} cycles is set by the {sm['limit']} process "
                      f"({'a record is ' + str(ent['in_words']) + ' input word(s)' if sm['limit'] == 'load' else 'a result is ' + str(ent['out_words']) + ' word(s)' if sm['limit'] == 'store' else 'see the core loop'}), "
                      "not by the program's dependences", file=out)
        if r["warnings"]:
            print("  tool messages by class (code: count, first message):", file=out)
            for code, (sev, n, msg) in sorted(r["warnings"].items()):
                print(f"    {sev.lower()} [{code}] x{n}: {msg[:140]}", file=out)
        # keep the reports next to the sources
        dest = os.path.join(where, "reports", ent["c_name"])
        os.makedirs(dest, exist_ok=True)
        for root, _dirs, names in os.walk(r["result"].fetched):
            for n in names:
                if n.endswith((".xml", ".rpt", ".log")):
                    with open(os.path.join(root, n), "rb") as src, open(os.path.join(dest, n), "wb") as dst:
                        dst.write(src.read())
        r["result"].cleanup()
    return status
