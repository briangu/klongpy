"""
kgpy --emit-hls: the files Vitis HLS needs to turn each ``.entry`` of a static
program into a kernel (experimental; the "wrapped core" stage).

The C emitter's output is the core, unchanged. For each entry this module
writes

    <p>_<entry>_top.cpp   the top function: load, core and store as three
                          processes under a dataflow pragma, m_axi and
                          s_axilite interfaces; no Xilinx header is needed to
                          compile it (pragmas are ``#pragma HLS`` and the
                          channels between the processes are arrays)
    <p>_<entry>_tb.cpp    a C test bench: reads in.bin, calls the top
                          function, writes out.bin and status.bin
    hls_<entry>.tcl       project, part, clock, csim_design, csynth_design
                          (and export_design when $KG_EXPORT is set)
    manifest.json         entries, record layout, cache key, tool settings

The record layout is the ABI of the kernel: one record is the arguments of the
entry concatenated in order, each array flattened row-major; the kernel reads
one record per call and writes one result record per call (a scalar result is
a record of one element). The words are ``kg_float`` (float32) or ``kg_int``
(32 or 64 bits) when the record is uniform; a record that mixes the two is
made of 32-bit words (so ``--int64`` rejects it).
"""
import hashlib
import json
import os

from ..errors import KlongStaticError
from . import fixed as fx
from .emit_c import ctype_of
from .ir import Call, ExternCall, dump, fnspecs, walk
from .signature import c_identifier, is_fixed

__all__ = ["emit_hls", "HlsOutput", "EMITTER_VERSION", "PART", "PLATFORM", "TOOL", "DEFAULT_CLOCK_MHZ", "reachable_externs"]

EMITTER_VERSION = "klong-emit-hls/1"
PART = "xcu250-figd2104-2L-e"
PLATFORM = "xilinx_u250_gen3x16_xdma_4_1_202210_1"
TOOL = {"name": "Vitis HLS", "version": "2022.2", "flow_target": "vitis"}
DEFAULT_CLOCK_MHZ = 300
TRAP_FLAG = "kg_hls_trap"
# the loop over the calls has a run-time bound; this is only what the latency estimate of the report assumes
TRIPCOUNT = 1024
STATUS_ANY = 1          # bit 0: some trap happened; bit 1: index out of range; bit 2: integer division by zero


class HlsOutput:
    """The generated files: ``files`` {name: text}, ``manifest`` (a dict), ``entries`` (one dict per entry)."""
    def __init__(self, prefix, files, manifest):
        self.prefix = prefix
        self.files = files
        self.manifest = manifest

    @property
    def entries(self):
        return self.manifest["entries"]

    def write(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for name, text in self.files.items():
            with open(os.path.join(out_dir, name), "w") as f:
                f.write(text)


def reachable_externs(entry):
    """The names of the externs an entry can reach through the functions it calls."""
    seen, names, todo = set(), [], [entry.fn]
    while todo:
        fn = todo.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        for n in walk(fn.body):
            if isinstance(n, ExternCall):
                if n.ext.name not in names:
                    names.append(n.ext.name)
            elif isinstance(n, Call):
                todo.append(n.fn)
            todo.extend(fnspecs(n))
    return names


def _reject(program, entry, why, hint=None):
    d = entry.decl
    return KlongStaticError(f"entry {d.name} ({d.signature}) cannot be a hardware kernel: {why} [hls]",
                            span=d.span, source=d.source, hint=hint)


def _bytes_of(elem, int_bits):
    if is_fixed(elem):
        return 4 if fx.spec_of(elem).W <= 32 else 8
    return 4 if elem == "float" else int_bits // 8


def roles(program, wrapped=False):
    """
    ``(readers, pairs)``: the entries that are not kernels of their own and the kernels that read them.

    A *summary pair* is a ``void`` entry with arguments (the driver: ``step: fixed<32,16> -> void``) and the one entry
    without arguments that returns a value (the reader: ``summary: -> fixed<64,32>[3]``). The two share the program's
    state. The driver's kernel makes its n calls, writes nothing per call, then calls the reader once per lane and writes
    that one result record: "replay many ticks, read a small summary at the end". Only the streamable kernel has it.
    ``pairs`` is {id(driver entry): reader entry}; a program with a driver and no single reader, or with a reader and
    no driver, gets no pairs and the entries are rejected one by one as before.
    """
    drivers = [e for e in program.entries if e.decl.signature.params and e.decl.signature.result.elem == "void"]
    readers = [e for e in program.entries if not e.decl.signature.params and e.decl.signature.result.elem != "void"]
    if wrapped or not drivers or len(readers) != 1:
        return [], {}
    return readers, {id(d): readers[0] for d in drivers}


def _layout(program, entry, reader=None):
    """The record layout of an entry, or a KlongStaticError saying why it has none. ``reader``: see roles()."""
    sig = entry.decl.signature
    ext = reachable_externs(entry) + ([n for n in reachable_externs(reader) if n not in reachable_externs(entry)] if reader else [])
    ib = program.int_bits
    if ext:
        raise _reject(program, entry, f"it calls the extern {', '.join(ext)}; hardware has no externs in this version",
                      "keep the extern out of the entry, or compute its value on the host and pass it as an argument")
    if sig.result.elem == "void" and reader is None:
        raise _reject(program, entry, "it returns nothing, so there is no result record to read back",
                      "declare one more entry without arguments that returns the state, for example "
                      '.entry("summary";"-> float[3]"): the kernel then runs its calls and writes that result once')
    params, off = [], 0
    for i, ty in enumerate(sig.params):
        if ty.elem not in ("int", "float") and not is_fixed(ty.elem):
            raise _reject(program, entry, f"argument {i + 1} is a {ty.elem}")
        name = entry.fn.param_names[i] if i < len(entry.fn.param_names) else f"a{i}"
        params.append({"name": name, "elem": ty.elem, "shape": list(ty.shape), "offset": off, "words": ty.size})
        off += ty.size
    widths = {_bytes_of(p["elem"], ib) for p in params}
    if len(widths) > 1:
        raise _reject(program, entry, "its arguments mix words of different widths (--int64 and fixed<W,I> with W over "
                                      "32 are 8 bytes, a float is 4), which one record cannot hold",
                      "make the arguments words of one width")
    r = reader.decl.signature.result if reader else sig.result
    return params, off, {"elem": r.elem, "shape": list(r.shape), "words": r.size}


# names the generated top function uses for its own arguments and locals: a parameter of the program cannot have one
RESERVED = frozenset(["in", "out", "n", "reset", "status", "params", "inst", "started", "pw", "sin", "sout", "sst", "win", "rec",
                      "buf", "res", "obuf", "ibuf", "first", "load", "core", "store"])


def registers(program, lanes):
    """The ``.param`` registers of the program: name, type, C type, default (and lanes) in a fixed order."""
    out = []
    for g in program.globals:
        if not g.param:
            continue
        if c_identifier(g.name) in RESERVED:
            raise KlongStaticError(f"the parameter {g.name} cannot be a kernel register: its name is one that the generated kernel "
                                   "uses for its own arguments ({}) [hls]".format(", ".join(sorted(RESERVED))),
                                   hint=f"rename the parameter, for example {g.name}_p")
        out.append({"name": g.name, "elem": g.ty.elem, "ctype": ctype_of(g.ty.elem), "cid": c_identifier(g.name),
                    "default": int(g.value) if g.ty.elem != "float" else float(g.value), "lanes": lanes or 1,
                    "bytes": _bytes_of(g.ty.elem, program.int_bits)})
    return out


def register_table(regs, lanes):
    """
    How the parameters of a kernel with lanes reach it: one table in card memory (an m_axi argument ``params``), because
    the host runtime passes scalars to an s_axilite argument and cannot write an array to one. Row k holds the
    ``lanes`` values of register k, each in a word of ``word_bytes`` (the widest register; a float travels as its bits,
    an int sign-extended). None without lanes: a parameter is then an s_axilite scalar.
    """
    if not lanes or not regs:
        return None
    wb = max(r["bytes"] for r in regs)
    return {"order": [r["name"] for r in regs], "lanes": lanes, "word_bytes": wb, "words": len(regs) * lanes,
            "bytes": len(regs) * lanes * wb}


def emit_hls(program, emitted, clock_mhz=DEFAULT_CLOCK_MHZ, source_name=None, float_bits=32, *, wrapped=False, lanes=None,
             reassociate=False, mode=None):
    """
    The HLS files for a lowered program (``program``) whose C is ``emitted`` (the result of emit_c.emit; for the
    streamable back end, emitted with an HlsOptions). Raises KlongStaticError for an entry that cannot be a kernel and
    ValueError for a float that is not float32.

        wrapped      stage A: the C is the core, unchanged, called once per record (--hls-wrapped)
        lanes        N cores on one input stream (stage B); parameters become arrays of N, the result gets a leading N
        reassociate  float reductions as trees (recorded in the manifest and the cache key)
    """
    if float_bits != 32:
        raise ValueError("--emit-hls: the card has float32 only in this version (--float64 is not available)")
    if lanes is not None and (wrapped or lanes < 1):
        raise ValueError("--lanes needs the streamable back end and N >= 1" if wrapped else "--lanes N: N must be 1 or more")
    if not program.entries:
        raise KlongStaticError("the program has no .entry, so there is no kernel to make [hls]",
                               hint='declare one, for example .entry("f";"float -> float")')
    prefix = emitted.prefix
    ib = program.int_bits
    regs = registers(program, lanes)
    table = register_table(regs, lanes)
    files, entries = {}, []
    flags = [f"-DKG_TRAP_FLAG={TRAP_FLAG}"] + (["-DKG_INT64"] if ib == 64 else [])
    readers, pairs = roles(program, wrapped)
    for e in program.entries:
        if any(e is r for r in readers):
            continue                      # read by the kernel of the driver, not a kernel of its own
        reader = pairs.get(id(e))
        params, in_words, result = _layout(program, e, reader)
        if in_words == 0:
            raise _reject(program, e, "it takes no argument, so a call has no input record")
        stem = f"{prefix}_{e.c_name}"
        top = f"{stem}_kernel"
        elems = {p["elem"] for p in params}
        mixed = len(elems) > 1
        wbytes = _bytes_of(params[0]["elem"], ib)
        in_elem = params[0]["elem"]
        out_elem = result["elem"]
        lw = lanes or 1
        out_words = result["words"] * lw
        ent = {
            "name": e.decl.name, "c_name": e.c_name, "signature": str(e.decl.signature), "top": top,
            "params": params, "result": dict(result, shape=([lanes] if lanes else []) + result["shape"],
                                              words=out_words),
            "in_words": in_words, "out_words": out_words,
            "in_type": f"uint{8 * wbytes}_t" if mixed else ctype_of(in_elem),
            "out_type": ctype_of(out_elem),
            "in_bytes": sum(_bytes_of(p["elem"], ib) * p["words"] for p in params),
            "out_bytes": _bytes_of(out_elem, ib) * out_words,
            "files": {"top": f"{stem}_top.cpp", "tb": f"{stem}_tb.cpp", "tcl": f"hls_{e.c_name}.tcl"},
            "kernel_args": ["in", "out", "n", "reset"] + (["params"] if table else [r["name"] for r in regs]) + ["status"],
        }
        if table:
            ent["register_table"] = table
        if mixed:
            ent["in_type"] = f"uint{8 * wbytes}_t"
        if reader is not None:
            # the result of the kernel is the one record of the reader, written after the n calls
            ent["output"] = "summary"
            ent["summary"] = {"entry": reader.decl.name, "c_name": reader.c_name, "signature": str(reader.decl.signature)}
        if wrapped:
            files[ent["files"]["top"]] = _top(prefix, e, ent, params, mixed, regs)
        elif reader is not None and lanes and lanes > 1 and not mixed:
            # one dataflow process per lane: see _top_lanes
            ent["structure"] = "lane-processes"
            ent["lane_files"] = [f"lane_{l}.c" for l in range(lanes)]
            for l in range(lanes):
                files[f"lane_{l}.c"] = _lane_c(prefix, l, emitted.h, regs, program)
            files[ent["files"]["top"]] = _top_lanes(program, prefix, e, ent, params, regs, lanes, reader, emitted.h)
        else:
            files[ent["files"]["top"]] = _top_b(program, prefix, e, ent, params, mixed, regs, lanes, mode, reader)
        files[ent["files"]["tb"]] = _tb(prefix, ent, regs, lanes, table)
        files[ent["files"]["tcl"]] = _tcl(prefix, ent, clock_mhz, flags, source_name, stage_b=not wrapped)
        entries.append(ent)
    if not wrapped:
        files["kg_hls_stub.h"] = STREAM_STUB
    key = cache_key(program, clock_mhz, lanes, wrapped, reassociate)
    manifest = {
        "format": 1,
        "emitter": EMITTER_VERSION,
        "program": prefix,
        "source": source_name or prefix + ".kg",
        "int_bits": ib,
        "float_bits": 32,
        "lanes": lanes or 1,
        "clock_mhz": clock_mhz,
        "cache_key": key,
        "cflags": flags,
        "tool": dict(TOOL, part=PART, platform=PLATFORM),
        "c_files": [prefix + ".c", prefix + ".h", "klong_rt.h"],
        "entries": entries,
        "status_bits": {"any_trap": 0, "bounds": 1, "div_zero": 2},
        "kernel": "void <top>(const in_type *in, out_type *out, unsigned n, unsigned reset, unsigned *status)",
    }
    if regs:
        manifest["registers"] = regs
        if table:
            manifest["register_table"] = table
            manifest["kernel"] = ("void <top>(const in_type *in, out_type *out, unsigned n, unsigned reset, "
                                  f"const uint{8 * table['word_bytes']}_t *params, unsigned *status)")
        else:
            manifest["kernel"] = ("void <top>(const in_type *in, out_type *out, unsigned n, unsigned reset, "
                                  + ", ".join(f"{r['ctype']} {r['name']}" for r in regs) + ", unsigned *status)")
    if any(e.get("lane_files") for e in entries):
        manifest["c_files"] += [f for e in entries for f in e.get("lane_files", [])]
    if not wrapped:
        manifest["stage"] = "B"
        manifest["lane_dimension"] = bool(lanes)
        manifest["reassociate"] = bool(reassociate)
        manifest["c_files"].append("kg_hls_stub.h")
    files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    return HlsOutput(prefix, files, manifest)


def cache_key(program, clock_mhz, lanes=None, wrapped=True, reassociate=False):
    """
    Design section 8: SHA-256 of the IR dump, the emitter version, the lanes, the clock, the platform and the tool version
    (and, for the streamable back end, the stage and the reassociation, which change the hardware). With the defaults
    of ticket 110 (stage A, no lanes) the key is the one that ticket computed.
    """
    h = hashlib.sha256()
    parts = [dump(program), EMITTER_VERSION, f"lanes={lanes or 1}", f"clock_mhz={clock_mhz}", f"platform={PLATFORM}",
             f"tool={TOOL['name']} {TOOL['version']}"]
    if not wrapped:
        parts += ["stage=B", f"lane_dimension={bool(lanes)}", f"reassociate={bool(reassociate)}"]
    for part in parts:
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The top function
# ---------------------------------------------------------------------------

def _reg_decl(regs, lanes=None):
    return "".join(f", {r['ctype']} {r['cid']}" + (f"[{lanes}]" if lanes else "") for r in regs)


def _top(prefix, e, ent, params, mixed, regs=()):
    stem = f"{prefix}_{e.c_name}"
    IN, OUT = ent["in_words"], ent["out_words"]
    tin, tout = ent["in_type"], ent["out_type"]
    L = []
    w = L.append
    w(f"/* {ent['files']['top']} - generated by KlongPy (kgpy --emit-hls) from {prefix}. Do not edit.")
    w(f" * Kernel {ent['top']} for entry {ent['name']}: {ent['signature']}.")
    w(f" * One input record is {IN} word(s) of {tin}: the arguments in order, arrays flattened row-major.")
    w(f" * One output record is {OUT} word(s) of {tout}. Each call of the kernel makes n calls of the entry.")
    w(" * status bit 0: some trap happened; bit 1: an index out of range; bit 2: an integer division by zero.")
    w(" * Build with -DKG_TRAP_FLAG=" + TRAP_FLAG + " (and -DKG_INT64 for a 64-bit kg_int; see hls_*.tcl). */")
    w("#include <stddef.h>")
    w("#include <stdint.h>")
    w("#include <string.h>")
    w("")
    w("#ifndef KG_TRAP_FLAG")
    w('#error "compile with -DKG_TRAP_FLAG=' + TRAP_FLAG + ' (the generated C records traps in that counter)"')
    w("#endif")
    w("")
    w('extern "C" {')
    w(f'#include "{prefix}.h"')
    w("extern unsigned KG_TRAP_FLAG;")
    w("}")
    w("")
    w(f"#define IN_WORDS {IN}")
    w(f"#define OUT_WORDS {OUT}")
    w(f"typedef {tin} in_t;")
    w(f"typedef {tout} out_t;")
    w("")
    if mixed:
        w("template <class T, class W> static inline T from_word(W v) { T t; memcpy(&t, &v, sizeof t); return t; }")
        w("")
    w("/* load: one record from memory to a buffer */")
    w("static void load(const in_t *in, in_t buf[IN_WORDS]) {")
    w("    for (unsigned j = 0; j < IN_WORDS; j++) {")
    w("#pragma HLS pipeline II=1")
    w("        buf[j] = in[j];")
    w("    }")
    w("}")
    w("")
    w("/* core: the C of the program, unchanged, on one record */")
    w(f"static void core(const in_t buf[IN_WORDS], out_t res[OUT_WORDS], bool first, bool reset{_reg_decl(regs)}) {{")
    w(f"    static {prefix}_t inst;")
    w("    static bool started = false;")
    w("    if (first) {")
    w("        KG_TRAP_FLAG = 0;")
    w("    }")
    w("    if (!started || (first && reset)) {")
    w(f"        {prefix}_init(&inst);")
    w("        started = true;")
    w("    }")
    for r in regs:
        w(f"    {prefix}_set_{r['cid']}(&inst, {r['cid']});")
    call_args = []
    for i, p in enumerate(params):
        o = p["offset"]
        if not mixed:
            call_args.append(f"&buf[{o}]" if p["shape"] else f"buf[{o}]")
            continue
        ctype = ctype_of(p["elem"])
        conv = f"from_word<{ctype}>"
        if p["shape"]:
            w(f"    {ctype} a{i}[{p['words']}];")
            w(f"    for (unsigned j = 0; j < {p['words']}; j++) {{")
            w(f"        a{i}[j] = {conv}(buf[{o} + j]);")
            w("    }")
            call_args.append(f"a{i}")
        else:
            call_args.append(f"{conv}(buf[{o}])")
    fn = f"{stem}"
    if e.decl.signature.result.shape:
        w(f"    {fn}(&inst{''.join(', ' + a for a in call_args)}, res);")
    else:
        w(f"    res[0] = {fn}(&inst{''.join(', ' + a for a in call_args)});")
    w("}")
    w("")
    w("/* store: one result record from a buffer to memory */")
    w("static void store(const out_t res[OUT_WORDS], out_t *out) {")
    w("    for (unsigned j = 0; j < OUT_WORDS; j++) {")
    w("#pragma HLS pipeline II=1")
    w("        out[j] = res[j];")
    w("    }")
    w("}")
    w("")
    w(f'extern "C" void {ent["top"]}(const in_t *in, out_t *out, unsigned n, unsigned reset{_reg_decl(regs)}, unsigned *status) {{')
    w("#pragma HLS interface m_axi port=in offset=slave bundle=gmem0")
    w("#pragma HLS interface m_axi port=out offset=slave bundle=gmem1")
    w("#pragma HLS interface m_axi port=status offset=slave bundle=gmem2")
    for port in ("in", "out", "n", "reset", *[r["cid"] for r in regs], "status"):
        w(f"#pragma HLS interface s_axilite port={port} bundle=control")
    w("#pragma HLS interface s_axilite port=return bundle=control")
    w("    for (unsigned i = 0; i < n; i++) {")
    w(f"#pragma HLS loop_tripcount min=1 max={TRIPCOUNT}")
    w("        in_t ibuf[IN_WORDS];")
    w("        out_t obuf[OUT_WORDS];")
    w("        const in_t *ip = in + (size_t)i * IN_WORDS;")
    w("        out_t *op = out + (size_t)i * OUT_WORDS;")
    w("        const bool first = (i == 0);")
    w("        const bool rst = (reset != 0);")
    w("#pragma HLS dataflow")
    w("        load(ip, ibuf);")
    w(f"        core(ibuf, obuf, first, rst{''.join(', ' + r['cid'] for r in regs)});")
    w("        store(obuf, op);")
    w("    }")
    w("    unsigned t = (n != 0) ? KG_TRAP_FLAG : 0u;")
    w("    *status = t ? (t | 1u) : 0u;")
    w("}")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# The streamable top function (stage B)
# ---------------------------------------------------------------------------

STREAM_STUB = """/* kg_hls_stub.h - generated by KlongPy (kgpy --emit-hls). A stand-in for hls::stream, used only when the
 * Xilinx header <hls_stream.h> is not on the include path (a test with an ordinary C++ compiler). Do not edit. */
#ifndef KG_HLS_STUB_H
#define KG_HLS_STUB_H
#include <deque>
namespace hls {
template <class T, int DEPTH = 0>
class stream {
public:
    stream() {}
    explicit stream(const char *) {}
    void write(const T &v) { q_.push_back(v); }
    T read() { T v = q_.front(); q_.pop_front(); return v; }
    bool empty() const { return q_.empty(); }
private:
    std::deque<T> q_;
};
}
#endif
"""

STREAM_DEPTH = 32
WIDE_BITS = 2048        # a record narrower than this travels between the processes as one stream element


def _top_b(program, prefix, e, ent, params, mixed, regs, lanes, mode, reader=None):
    """
    The streamable kernel: load, core and store as three processes joined by streams under one dataflow region.

    load   reads the input words in one pipelined loop (so the memory port sees one long burst, not one request per
           record) and passes them on, as whole records when a record is small
    core   one pipelined loop over the records; each iteration calls the entry once per lane. A sliding-window entry
           (emit_c.match_window) is instead one loop over the words: a shift register, one output per word
    store  writes the output words in one pipelined loop and the status word last
    """
    from .emit_c import match_window
    stem = f"{prefix}_{e.c_name}"
    IN, OUT = ent["in_words"], ent["out_words"]
    L = lanes or 1
    tin, tout = ent["in_type"], ent["out_type"]
    res_words = ent["result"]["words"] // L
    sig = e.decl.signature
    summary = reader is not None
    tot = "OUT_WORDS" if summary else "n * OUT_WORDS"      # the output words of one launch
    win = match_window(program, e) if not lanes and not regs and not summary else None
    pipelined = mode != "block"
    in_bits = IN * _bytes_of(params[0]["elem"], program.int_bits) * 8
    out_bits = OUT * _bytes_of(sig.result.elem, program.int_bits) * 8
    wide_in = in_bits <= WIDE_BITS and not win
    wide_out = out_bits <= WIDE_BITS and not win
    if win:
        ent["window"] = {"taps": win["taps"], "outputs": win["outputs"], "offset": win["offset"], "inputs": win["inputs"],
                         "op": win["op"]}
    ent["streams"] = {"in": "record" if wide_in else "word", "out": "record" if wide_out else "word"}
    Ls = []
    w = Ls.append
    w(f"/* {ent['files']['top']} - generated by KlongPy (kgpy --emit-hls) from {prefix}. Do not edit.")
    w(f" * Streamable kernel {ent['top']} for entry {ent['name']}: {ent['signature']}"
      + (f", {lanes} lane(s)." if lanes else "."))
    w(f" * One input record is {IN} word(s) of {tin}: the arguments in order, arrays flattened row-major.")
    w(f" * One output record is {OUT} word(s) of {tout}" + (f": {lanes} lanes of {res_words} word(s), lane first." if lanes else "."))
    if win:
        w(f" * The entry is a sliding window of {win['taps']} samples: the core is a shift register that makes one output per"
          " input word once the window is full.")
    elif summary:
        w(f" * Each call of the kernel makes n calls of the entry, which return nothing, and then one call of {reader.decl.name}"
          f" ({reader.decl.signature}) per lane: that is the one output record.")
    else:
        w(" * Each call of the kernel makes n calls of the entry. The core is "
          + ("one pipelined region (II=1 when the program allows it)." if pipelined else "a block program: its loops are pipelined, not the call."))
    w(" * status bit 0: some trap happened; bit 1: an index out of range; bit 2: an integer division by zero.")
    w(" * Build with -DKG_TRAP_FLAG=" + TRAP_FLAG + " (and -DKG_INT64 for a 64-bit kg_int; see hls_*.tcl). */")
    w("#include <stddef.h>")
    w("#include <stdint.h>")
    w("#include <string.h>")
    w("#if defined(__has_include)")
    w("#if __has_include(<hls_stream.h>)")
    w("#include <hls_stream.h>")
    w("#else")
    w('#include "kg_hls_stub.h"')
    w("#endif")
    w("#else")
    w("#include <hls_stream.h>")
    w("#endif")
    w("")
    w("#ifndef KG_TRAP_FLAG")
    w('#error "compile with -DKG_TRAP_FLAG=' + TRAP_FLAG + ' (the generated C records traps in that counter)"')
    w("#endif")
    w("")
    w('extern "C" {')
    w(f'#include "{prefix}.h"')
    w("extern unsigned KG_TRAP_FLAG;")
    w("}")
    w("")
    w(f"#define IN_WORDS {IN}")
    w(f"#define RES_WORDS {res_words}")
    w(f"#define LANES {L}")
    w("#define OUT_WORDS (LANES * RES_WORDS)")
    w(f"typedef {tin} in_t;")
    w(f"typedef {tout} out_t;")
    if ent.get("register_table"):
        rt = ent["register_table"]
        w(f"typedef uint{8 * rt['word_bytes']}_t reg_t;")
        w(f"#define REG_WORDS {rt['words']}")
        w("template <class T, class W> static inline T from_bits(W v) { T t; memcpy(&t, &v, sizeof t); return t; }")
    if wide_in:
        w("typedef struct { in_t w[IN_WORDS]; } in_rec_t;")
    if wide_out:
        w("typedef struct { out_t w[OUT_WORDS]; } out_rec_t;")
    sin_t = "in_rec_t" if wide_in else "in_t"
    sout_t = "out_rec_t" if wide_out else "out_t"
    w("")
    if mixed:
        w("template <class T, class W> static inline T from_word(W v) { T t; memcpy(&t, &v, sizeof t); return t; }")
        w("")
    # ---- load
    w("/* load: the input words, one per cycle */")
    w(f"static void load(const in_t *in, hls::stream<{sin_t}> &s, uint64_t n) {{")
    if wide_in:
        w("    in_rec_t rec;")
        w("    unsigned j = 0;")
        w("    for (uint64_t i = 0; i < n * IN_WORDS; i++) {")
        w("#pragma HLS pipeline II=1")
        w("        rec.w[j] = in[i];")
        w("        if (++j == IN_WORDS) {")
        w("            s.write(rec);")
        w("            j = 0;")
        w("        }")
        w("    }")
    else:
        w("    for (uint64_t i = 0; i < n * IN_WORDS; i++) {")
        w("#pragma HLS pipeline II=1")
        w("        s.write(in[i]);")
        w("    }")
    w("}")
    w("")
    # ---- core
    table = ent.get("register_table")
    rdecl = ", const reg_t *params" if table else _reg_decl(regs, lanes)
    if win:
        A, B, c, N = win["taps"], win["outputs"], win["offset"], win["inputs"]
        w("/* core: a shift register over the input words; the window function is the reduction of the program */")
        w(f"static void core(hls::stream<{sin_t}> &sin, hls::stream<{sout_t}> &sout, hls::stream<unsigned> &sst, uint64_t n, "
          "unsigned reset) {")
        w("    (void)reset;")
        w(f"    in_t win[{A}];")
        w("#pragma HLS array_partition variable=win complete")
        w("    KG_TRAP_FLAG = 0;")
        w("    unsigned t = 0;")
        w("    for (uint64_t i = 0; i < n * IN_WORDS; i++) {")
        w("#pragma HLS pipeline II=1")
        w("        const in_t x = sin.read();")
        w(f"        for (unsigned k = 0; k + 1 < {A}; k++) {{")
        w("#pragma HLS unroll")
        w("            win[k] = win[k + 1];")
        w("        }")
        w(f"        win[{A - 1}] = x;")
        w(f"        if (t >= {c + A - 1} && t < {c + A - 1 + B}) {{")
        w(f"            sout.write({stem}_window(win));")
        w("        }")
        w(f"        t = (t + 1 == {N}) ? 0 : t + 1;")
        w("    }")
        w("    sst.write(n != 0 ? KG_TRAP_FLAG : 0u);")
        w("}")
    else:
        w("/* core: the C of the program, one call of the entry per record and lane */")
        w(f"static void core(hls::stream<{sin_t}> &sin, hls::stream<{sout_t}> &sout, hls::stream<unsigned> &sst, uint64_t n, "
          f"unsigned reset{rdecl}) {{")
        w(f"    static {prefix}_t inst[LANES];")
        w("    static bool started = false;")
        w("    if (!started || reset) {")
        w("        for (unsigned l = 0; l < LANES; l++) {")
        w("#pragma HLS unroll")
        w(f"            {prefix}_init(&inst[l]);")
        w("        }")
        w("        started = true;")
        w("    }")
        if table:
            w("    reg_t pw[REG_WORDS];")
            w("    for (unsigned i = 0; i < REG_WORDS; i++) {")
            w("#pragma HLS pipeline II=1")
            w("        pw[i] = params[i];")
            w("    }")
            w("    for (unsigned l = 0; l < LANES; l++) {")
            w("#pragma HLS unroll")
            for k, r in enumerate(regs):
                val = f"from_bits<{r['ctype']}>(pw[{k} * LANES + l])" if r["elem"] == "float" else f"({r['ctype']})pw[{k} * LANES + l]"
                w(f"        {prefix}_set_{r['cid']}(&inst[l], {val});")
            w("    }")
        elif regs:
            w("    for (unsigned l = 0; l < LANES; l++) {")
            w("#pragma HLS unroll")
            for r in regs:
                w(f"        {prefix}_set_{r['cid']}(&inst[l], {r['cid']}{'[l]' if lanes else ''});")
            w("    }")
        w("    KG_TRAP_FLAG = 0;")
        w("    for (uint64_t r = 0; r < n; r++) {")
        if pipelined:
            w("#pragma HLS pipeline II=1")
        if wide_in:
            w("        const in_rec_t rec = sin.read();")
            rec = "rec.w"
        else:
            w("        in_t rec[IN_WORDS];")
            w("        for (unsigned j = 0; j < IN_WORDS; j++) {")
            w("#pragma HLS pipeline II=1")
            w("            rec[j] = sin.read();")
            w("        }")
            rec = "rec"
        if not summary:
            if wide_out:
                w("        out_rec_t o;")
                oref = "o.w"
            else:
                w("        out_t o[OUT_WORDS];")
                oref = "o"
        w("        for (unsigned l = 0; l < LANES; l++) {")
        w("#pragma HLS unroll")
        call_args = []
        for i, p in enumerate(params):
            off = p["offset"]
            if not mixed:
                call_args.append(f"&{rec}[{off}]" if p["shape"] else f"{rec}[{off}]")
                continue
            ctype = ctype_of(p["elem"])
            if p["shape"]:
                w(f"            {ctype} a{i}[{p['words']}];")
                w(f"            for (unsigned j = 0; j < {p['words']}; j++) {{")
                w(f"                a{i}[j] = from_word<{ctype}>({rec}[{off} + j]);")
                w("            }")
                call_args.append(f"a{i}")
            else:
                call_args.append(f"from_word<{ctype}>({rec}[{off}])")
        args = "".join(", " + a for a in call_args)
        if summary:
            w(f"            {stem}(&inst[l]{args});")
        elif sig.result.shape:
            w(f"            {stem}(&inst[l]{args}, &{oref}[l * RES_WORDS]);")
        else:
            w(f"            {oref}[l * RES_WORDS] = {stem}(&inst[l]{args});")
        w("        }")
        if summary:
            pass
        elif wide_out:
            w("        sout.write(o);")
        else:
            w("        for (unsigned j = 0; j < OUT_WORDS; j++) {")
            w("#pragma HLS pipeline II=1")
            w("            sout.write(o[j]);")
            w("        }")
        w("    }")
        if summary:
            # the reader: one call per lane after the n calls of the driver, one record on the output stream
            rstem = f"{prefix}_{reader.c_name}"
            if wide_out:
                w("    out_rec_t o;")
                oref = "o.w"
            else:
                w("    out_t o[OUT_WORDS];")
                oref = "o"
            w("    for (unsigned l = 0; l < LANES; l++) {")
            w("#pragma HLS unroll")
            if reader.decl.signature.result.shape:
                w(f"        {rstem}(&inst[l], &{oref}[l * RES_WORDS]);")
            else:
                w(f"        {oref}[l * RES_WORDS] = {rstem}(&inst[l]);")
            w("    }")
            if wide_out:
                w("    sout.write(o);")
            else:
                w("    for (unsigned j = 0; j < OUT_WORDS; j++) {")
                w("#pragma HLS pipeline II=1")
                w("        sout.write(o[j]);")
                w("    }")
        w("    sst.write(n != 0 ? KG_TRAP_FLAG : 0u);")
        w("}")
    w("")
    # ---- store
    w("/* store: the output words, one per cycle, then the status word */")
    w(f"static void store(hls::stream<{sout_t}> &s, out_t *out, hls::stream<unsigned> &sst, unsigned *status, uint64_t n) {{")
    if summary:
        w("    (void)n;")
    if wide_out:
        w("    out_rec_t rec;")
        w("    unsigned k = 0;")
        w(f"    for (uint64_t i = 0; i < {tot}; i++) {{")
        w("#pragma HLS pipeline II=1")
        w("        if (k == 0) {")
        w("            rec = s.read();")
        w("        }")
        w("        out[i] = rec.w[k];")
        w("        k = (k + 1 == OUT_WORDS) ? 0 : k + 1;")
        w("    }")
    else:
        w(f"    for (uint64_t i = 0; i < {tot}; i++) {{")
        w("#pragma HLS pipeline II=1")
        w("        out[i] = s.read();")
        w("    }")
    w("    const unsigned t = sst.read();")
    w("    *status = t ? (t | 1u) : 0u;")
    w("}")
    w("")
    w(f'extern "C" void {ent["top"]}(const in_t *in, out_t *out, unsigned n, unsigned reset{rdecl}, unsigned *status) {{')
    w("#pragma HLS interface m_axi port=in offset=slave bundle=gmem0")
    w("#pragma HLS interface m_axi port=out offset=slave bundle=gmem1")
    w("#pragma HLS interface m_axi port=status offset=slave bundle=gmem2")
    if table:
        w("#pragma HLS interface m_axi port=params offset=slave bundle=gmem3")
    for port in ("in", "out", "n", "reset", *(["params"] if table else [r["cid"] for r in regs]), "status"):
        w(f"#pragma HLS interface s_axilite port={port} bundle=control")
    w("#pragma HLS interface s_axilite port=return bundle=control")
    w("#pragma HLS dataflow")
    w(f"    hls::stream<{sin_t}> sin(\"sin\");")
    w(f"    hls::stream<{sout_t}> sout(\"sout\");")
    w("    hls::stream<unsigned> sst(\"sst\");")
    w(f"#pragma HLS stream variable=sin depth={STREAM_DEPTH}")
    w(f"#pragma HLS stream variable=sout depth={STREAM_DEPTH}")
    w("#pragma HLS stream variable=sst depth=2")
    w("    load(in, sin, n);")
    w(f"    core(sin, sout, sst, n, reset{'' if win else (', params' if table else ''.join(', ' + r['cid'] for r in regs))});")
    w("    store(sout, out, sst, status, n);")
    w("}")
    w("")
    return "\n".join(Ls)


def _lane_names(prefix, header):
    """The public functions of the emitted C (from its header): <prefix>_init, _size, _set_*, one per entry."""
    import re
    return sorted(set(re.findall(r"\b(%s_[A-Za-z0-9_]+)\s*\(" % re.escape(prefix), header)))


def _lane_c(prefix, lane, header, regs, program):
    """lane_<l>.c: the program's C under other names, so that lane l of the kernel owns an instance, its functions and a trap flag."""
    L = [f"/* lane_{lane}.c - generated by KlongPy (kgpy --emit-hls). Do not edit.",
         f" * The C of {prefix}.c under other names (a suffix _{lane}): lane {lane} of the kernel is one dataflow process, and a",
         " * process may not share a variable with another, so it has its own functions and its own trap flag. */",
         "#undef KG_TRAP_FLAG",
         f"#define KG_TRAP_FLAG {TRAP_FLAG}_{lane}",
         f"#define kg_trap kg_trap_{lane}"]
    for name in _lane_names(prefix, header):
        L.append(f"#define {name} {name}_{lane}")
    L.append(f'#include "{prefix}.c"')
    return "\n".join(L) + "\n"


def _lane_decls(prefix, lane, header):
    """The prototypes of lane l's functions: the lines of the header that declare the program's functions, renamed."""
    import re
    names = _lane_names(prefix, header)
    out = []
    for line in header.splitlines():
        if any(re.search(r"\b%s\s*\(" % re.escape(n), line) for n in names) and line.rstrip().endswith(("*/", ";")):
            text = line
            for n in names:
                text = re.sub(r"\b%s\b" % re.escape(n), f"{n}_{lane}", text)
            out.append(text)
    return out


def _top_lanes(program, prefix, e, ent, params, regs, lanes, reader, header):
    """
    The kernel of a summary pair with lanes: one dataflow process per lane.

    A single pipelined loop that holds all the lanes stalls on one condition (the input FIFO is empty), and that one net
    drives the clock enable of every register of every lane: tens of thousands of loads on a net that is not driven by a
    register, which routes at about 10 ns however few operations there are between registers. So each lane is a process
    of its own with its own loop (a few thousand loads per stall net), joined to the others by streams:

        load   the input words, copied to every lane's stream
        regs   the register table (an m_axi argument), one set of values to each lane's stream
        lane_l the lane's instance of the program (its C is lane_<l>.c), n calls of the entry, then the reader's record
        store  the records of the lanes in order, then the OR of their trap flags
    """
    stem = f"{prefix}_{e.c_name}"
    rstem = f"{prefix}_{reader.c_name}"
    IN = ent["in_words"]
    L = lanes
    tin, tout = ent["in_type"], ent["out_type"]
    res_words = ent["result"]["words"] // L
    table = ent.get("register_table")
    in_bits = IN * _bytes_of(params[0]["elem"], program.int_bits) * 8
    wide_in = in_bits <= WIDE_BITS
    sin_t = "in_rec_t" if wide_in else "in_t"
    Ls = []
    w = Ls.append
    w(f"/* {ent['files']['top']} - generated by KlongPy (kgpy --emit-hls) from {prefix}. Do not edit.")
    w(f" * Streamable kernel {ent['top']} for entry {ent['name']}: {ent['signature']}, {lanes} lane(s), one process per lane.")
    w(f" * One input record is {IN} word(s) of {tin}: the arguments in order, arrays flattened row-major.")
    w(f" * One output record is {ent['out_words']} word(s) of {tout}: {lanes} lanes of {res_words} word(s), lane first.")
    w(f" * Each call of the kernel makes n calls of the entry, which return nothing, and then one call of {reader.decl.name}"
      f" ({reader.decl.signature}) per lane: that is the one output record.")
    w(" * status bit 0: some trap happened; bit 1: an index out of range; bit 2: an integer division by zero.")
    w(" * Build with -DKG_TRAP_FLAG=" + TRAP_FLAG + " (and -DKG_INT64 for a 64-bit kg_int; see hls_*.tcl). */")
    w("#include <stddef.h>")
    w("#include <stdint.h>")
    w("#include <string.h>")
    w("#if defined(__has_include)")
    w("#if __has_include(<hls_stream.h>)")
    w("#include <hls_stream.h>")
    w("#else")
    w('#include "kg_hls_stub.h"')
    w("#endif")
    w("#else")
    w("#include <hls_stream.h>")
    w("#endif")
    w("")
    w("#ifndef KG_TRAP_FLAG")
    w('#error "compile with -DKG_TRAP_FLAG=' + TRAP_FLAG + ' (the generated C records traps in that counter)"')
    w("#endif")
    w("")
    w('extern "C" {')
    w(f'#include "{prefix}.h"')
    for l in range(L):
        w(f"/* lane {l} */")
        for d in _lane_decls(prefix, l, header):
            w(d)
        w(f"extern unsigned {TRAP_FLAG}_{l};")
    w("}")
    w("")
    w(f"#define IN_WORDS {IN}")
    w(f"#define RES_WORDS {res_words}")
    w(f"#define LANES {L}")
    w(f"typedef {tin} in_t;")
    w(f"typedef {tout} out_t;")
    if wide_in:
        w("typedef struct { in_t w[IN_WORDS]; } in_rec_t;")
    if table:
        w(f"typedef uint{8 * table['word_bytes']}_t reg_t;")
        w(f"#define REG_WORDS {table['words']}")
        w(f"#define REG_COUNT {len(regs)}")
        w("template <class T, class W> static inline T from_bits(W v) { T t; memcpy(&t, &v, sizeof t); return t; }")
    w("")
    w("/* load: the input words, one per cycle, to every lane */")
    w(f"static void load(const in_t *in, hls::stream<{sin_t}> sin[LANES], uint64_t n) {{")
    if wide_in:
        w("    in_rec_t rec;")
        w("    unsigned j = 0;")
        w("    for (uint64_t i = 0; i < n * IN_WORDS; i++) {")
        w("#pragma HLS pipeline II=1")
        w("        rec.w[j] = in[i];")
        w("        if (++j == IN_WORDS) {")
        w("            for (unsigned l = 0; l < LANES; l++) {")
        w("#pragma HLS unroll")
        w("                sin[l].write(rec);")
        w("            }")
        w("            j = 0;")
        w("        }")
        w("    }")
    else:
        w("    for (uint64_t i = 0; i < n * IN_WORDS; i++) {")
        w("#pragma HLS pipeline II=1")
        w("        const in_t v = in[i];")
        w("        for (unsigned l = 0; l < LANES; l++) {")
        w("#pragma HLS unroll")
        w("            sin[l].write(v);")
        w("        }")
        w("    }")
    w("}")
    w("")
    if table:
        w("/* regs: the register table, read once, one set of values to each lane */")
        w("static void regs(const reg_t *params, hls::stream<reg_t> sreg[LANES]) {")
        w("    reg_t pw[REG_WORDS];")
        w("    for (unsigned i = 0; i < REG_WORDS; i++) {")
        w("#pragma HLS pipeline II=1")
        w("        pw[i] = params[i];")
        w("    }")
        w("    for (unsigned k = 0; k < REG_COUNT; k++) {")
        w("        for (unsigned l = 0; l < LANES; l++) {")
        w("#pragma HLS unroll")
        w("            sreg[l].write(pw[k * LANES + l]);")
        w("        }")
        w("    }")
        w("}")
        w("")
    for l in range(L):
        w(f"/* lane {l} */")
        regarg = f"hls::stream<reg_t> &sreg, " if table else ""
        w(f"static void lane_{l}(hls::stream<{sin_t}> &sin, {regarg}hls::stream<out_t> &sres, hls::stream<unsigned> &sst, "
          "uint64_t n, unsigned reset) {")
        w(f"    static {prefix}_t inst;")
        w("    static bool started = false;")
        w("    if (!started || reset) {")
        w(f"        {prefix}_init_{l}(&inst);")
        w("        started = true;")
        w("    }")
        for r in regs:
            if table:
                val = f"from_bits<{r['ctype']}>(sreg.read())" if r["elem"] == "float" else f"({r['ctype']})sreg.read()"
                w(f"    {prefix}_set_{r['cid']}_{l}(&inst, {val});")
        w(f"    {TRAP_FLAG}_{l} = 0;")
        w("    for (uint64_t r = 0; r < n; r++) {")
        w("#pragma HLS pipeline II=1")
        if wide_in:
            w(f"        const in_rec_t rec = sin.read();")
            rec = "rec.w"
        else:
            w("        in_t rec[IN_WORDS];")
            w("        for (unsigned j = 0; j < IN_WORDS; j++) {")
            w("#pragma HLS pipeline II=1")
            w("            rec[j] = sin.read();")
            w("        }")
            rec = "rec"
        args = "".join(", " + (f"&{rec}[{p['offset']}]" if p["shape"] else f"{rec}[{p['offset']}]") for p in params)
        w(f"        {stem}_{l}(&inst{args});")
        w("    }")
        w("    out_t o[RES_WORDS];")
        if reader.decl.signature.result.shape:
            w(f"    {rstem}_{l}(&inst, o);")
        else:
            w(f"    o[0] = {rstem}_{l}(&inst);")
        w("    for (unsigned j = 0; j < RES_WORDS; j++) {")
        w("#pragma HLS pipeline II=1")
        w("        sres.write(o[j]);")
        w("    }")
        w(f"    sst.write(n != 0 ? {TRAP_FLAG}_{l} : 0u);")
        w("}")
        w("")
    w("/* store: the records of the lanes in order, then the status word */")
    w("static void store(hls::stream<out_t> sres[LANES], hls::stream<unsigned> sst[LANES], out_t *out, unsigned *status) {")
    w("    out_t rec[LANES * RES_WORDS];")
    w("    for (unsigned l = 0; l < LANES; l++) {")
    w("#pragma HLS unroll")
    w("        for (unsigned j = 0; j < RES_WORDS; j++) {")
    w("#pragma HLS unroll")
    w("            rec[l * RES_WORDS + j] = sres[l].read();")
    w("        }")
    w("    }")
    w("    for (unsigned i = 0; i < LANES * RES_WORDS; i++) {")
    w("#pragma HLS pipeline II=1")
    w("        out[i] = rec[i];")
    w("    }")
    w("    unsigned t = 0;")
    w("    for (unsigned l = 0; l < LANES; l++) {")
    w("#pragma HLS unroll")
    w("        t |= sst[l].read();")
    w("    }")
    w("    *status = t ? (t | 1u) : 0u;")
    w("}")
    w("")
    rdecl = ", const reg_t *params" if table else ""
    w(f'extern "C" void {ent["top"]}(const in_t *in, out_t *out, unsigned n, unsigned reset{rdecl}, unsigned *status) {{')
    w("#pragma HLS interface m_axi port=in offset=slave bundle=gmem0")
    w("#pragma HLS interface m_axi port=out offset=slave bundle=gmem1")
    w("#pragma HLS interface m_axi port=status offset=slave bundle=gmem2")
    if table:
        w("#pragma HLS interface m_axi port=params offset=slave bundle=gmem3")
    for port in ("in", "out", "n", "reset", *(["params"] if table else []), "status"):
        w(f"#pragma HLS interface s_axilite port={port} bundle=control")
    w("#pragma HLS interface s_axilite port=return bundle=control")
    w("#pragma HLS dataflow")
    w(f"    hls::stream<{sin_t}> sin[LANES];")
    if table:
        w("    hls::stream<reg_t> sreg[LANES];")
    w("    hls::stream<out_t> sres[LANES];")
    w("    hls::stream<unsigned> sst[LANES];")
    w(f"#pragma HLS stream variable=sin depth={STREAM_DEPTH}")
    if table:
        w(f"#pragma HLS stream variable=sreg depth={len(regs) + 1}")
    w(f"#pragma HLS stream variable=sres depth={res_words + 1}")
    w("#pragma HLS stream variable=sst depth=2")
    w("    load(in, sin, n);")
    if table:
        w("    regs(params, sreg);")
    for l in range(L):
        w(f"    lane_{l}(sin[{l}], {'sreg[%d], ' % l if table else ''}sres[{l}], sst[{l}], n, reset);")
    w("    store(sres, sst, out, status);")
    w("}")
    w("")
    return "\n".join(Ls)


# ---------------------------------------------------------------------------
# The test bench
# ---------------------------------------------------------------------------

def _tb(prefix, ent, regs=(), lanes=None, table=None):
    L = []
    w = L.append
    w(f"/* {ent['files']['tb']} - generated by KlongPy (kgpy --emit-hls). Do not edit.")
    if regs:
        w(" * usage: tb [DIR [RECORDS_PER_CALL]]  reads DIR/in.bin and DIR/regs.bin (the values of each parameter, one per lane,")
        w(" * in manifest order; absent: the defaults), writes DIR/out.bin and DIR/status.bin (one 32-bit word,")
    else:
        w(" * usage: tb [DIR [RECORDS_PER_CALL]]  reads DIR/in.bin, writes DIR/out.bin and DIR/status.bin (one 32-bit word,")
    if ent.get("output") == "summary":
        w(" * the OR of the status of every call; out.bin is the one summary record, as it is after the last call)."
          " RECORDS_PER_CALL 0 (the default) is one call for the whole file;")
    else:
        w(" * the OR of the status of every call). RECORDS_PER_CALL 0 (the default) is one call for the whole file;")
    w(" * with a number, the first call has reset set and the rest continue the state. */")
    w("#include <stdio.h>")
    w("#include <stdlib.h>")
    w("#include <stdint.h>")
    if table:
        w("#include <string.h>")
    w("#include <vector>")
    w("")
    w('extern "C" {')
    w(f'#include "{prefix}.h"')
    w("}")
    w(f"typedef {ent['in_type']} in_t;")
    w(f"typedef {ent['out_type']} out_t;")
    w(f"#define IN_WORDS {ent['in_words']}")
    w(f"#define OUT_WORDS {ent['out_words']}")
    if table:
        w(f"typedef uint{8 * table['word_bytes']}_t reg_t;")
        w(f'extern "C" void {ent["top"]}(const in_t *in, out_t *out, unsigned n, unsigned reset, const reg_t *params, unsigned *status);')
    else:
        w(f'extern "C" void {ent["top"]}(const in_t *in, out_t *out, unsigned n, unsigned reset{_reg_decl(regs, lanes)}, unsigned *status);')
    w("")
    w("int main(int argc, char **argv) {")
    w('    const char *dir = argc > 1 ? argv[1] : ".";')
    w("    unsigned per = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 10) : 0;")
    w("    char path[4096];")
    for r in regs:
        n = lanes or 1
        w(f"    {r['ctype']} {r['cid']}[{n}];")
        w(f"    for (unsigned l = 0; l < {n}; l++) {{")
        w(f"        {r['cid']}[l] = {r['default']!r};")
        w("    }")
    if regs:
        w('    snprintf(path, sizeof path, "%s/regs.bin", dir);')
        w('    FILE *rf = fopen(path, "rb");')
        w("    if (rf) {")
        for r in regs:
            w(f"        if (fread({r['cid']}, sizeof({r['ctype']}), {lanes or 1}, rf) != {lanes or 1}) {{")
            w('            fprintf(stderr, "tb: regs.bin is too short\\n");')
            w("            return 2;")
            w("        }")
        w("        fclose(rf);")
        w("    }")
    if table:
        w(f"    std::vector<reg_t> params({table['words']});")
        for k, r in enumerate(regs):
            w(f"    for (unsigned l = 0; l < {lanes}; l++) {{")
            if r["elem"] == "float":
                w(f"        reg_t bits = 0;")
                w(f"        memcpy(&bits, &{r['cid']}[l], sizeof(float));")
                w(f"        params[{k * lanes} + l] = bits;")
            else:
                w(f"        params[{k * lanes} + l] = (reg_t)(int64_t){r['cid']}[l];")
            w("    }")
    w('    snprintf(path, sizeof path, "%s/in.bin", dir);')
    w('    FILE *f = fopen(path, "rb");')
    w("    if (!f) {")
    w('        fprintf(stderr, "tb: cannot read %s\\n", path);')
    w("        return 2;")
    w("    }")
    w("    std::vector<in_t> in;")
    w("    in_t word;")
    w("    while (fread(&word, sizeof word, 1, f) == 1) {")
    w("        in.push_back(word);")
    w("    }")
    w("    fclose(f);")
    w("    if (in.size() % IN_WORDS != 0) {")
    w('        fprintf(stderr, "tb: in.bin is not a whole number of records (%zu words of %d)\\n", in.size(), IN_WORDS);')
    w("        return 2;")
    w("    }")
    w("    unsigned n = (unsigned)(in.size() / IN_WORDS);")
    summary = ent.get("output") == "summary"
    w("    std::vector<out_t> out(" + ("(size_t)OUT_WORDS + 1" if summary else "(size_t)n * OUT_WORDS + 1") + ");")
    w("    unsigned status = 0, all = 0;")
    w("    if (per == 0 || per > n) {")
    w("        per = n;")
    w("    }")
    w("    unsigned done = 0;")
    w("    do {")
    w("        unsigned m = n - done < per ? n - done : per;")
    w(f"        {ent['top']}(in.data() + (size_t)done * IN_WORDS, out.data() + {'0' if summary else '(size_t)done * OUT_WORDS'}, m, done == 0"
      f"{', params.data()' if table else ''.join(', ' + r['cid'] + ('' if lanes else '[0]') for r in regs)}, &status);")
    w("        all |= status;")
    w("        done += m;")
    w("    } while (done < n);")
    w('    snprintf(path, sizeof path, "%s/out.bin", dir);')
    w('    f = fopen(path, "wb");')
    w("    if (!f) {")
    w('        fprintf(stderr, "tb: cannot write %s\\n", path);')
    w("        return 2;")
    w("    }")
    w("    fwrite(out.data(), sizeof(out_t), " + ("(size_t)OUT_WORDS" if summary else "(size_t)n * OUT_WORDS") + ", f);")
    w("    fclose(f);")
    w('    snprintf(path, sizeof path, "%s/status.bin", dir);')
    w('    f = fopen(path, "wb");')
    w("    if (!f) {")
    w('        fprintf(stderr, "tb: cannot write %s\\n", path);')
    w("        return 2;")
    w("    }")
    w("    fwrite(&all, sizeof all, 1, f);")
    w("    fclose(f);")
    w('    printf("records=%u status=%u\\n", n, all);')
    w("    return 0;")
    w("}")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# The tcl script
# ---------------------------------------------------------------------------

def _tcl(prefix, ent, clock_mhz, flags, source_name, stage_b=False):
    period = 1000.0 / clock_mhz
    cf = " ".join(flags)
    L = []
    w = L.append
    w(f"# {ent['files']['tcl']} - generated by KlongPy (kgpy --emit-hls). Do not edit.")
    w("# Run:  vitis_hls -f " + ent["files"]["tcl"])
    w("# Environment:  KG_DATA  directory holding in.bin for the C simulation (default: this directory; out.bin and")
    w("#                        status.bin are written there)")
    w("#               KG_CHUNK records per kernel call in the C simulation (default 0: one call)")
    w("#               KG_NO_SYNTH  set to stop after the C simulation")
    w("#               KG_EXPORT  set to also run export_design -format xo (a Vitis kernel object)")
    w("set here [file dirname [file normalize [info script]]]")
    w("set data $here")
    w("if {[info exists ::env(KG_DATA)]} { set data $::env(KG_DATA) }")
    w("set chunk 0")
    w("if {[info exists ::env(KG_CHUNK)]} { set chunk $::env(KG_CHUNK) }")
    w(f'set cflags "-I$here {cf}"')
    w(f"open_project -reset proj_{ent['c_name']}")
    w(f"set_top {ent['top']}")
    w(f"add_files $here/{prefix}.c -cflags $cflags")
    for lf in ent.get("lane_files", []):
        w(f"add_files $here/{lf} -cflags $cflags")
    w(f"add_files $here/{ent['files']['top']} -cflags $cflags")
    w(f"add_files -tb $here/{ent['files']['tb']} -cflags $cflags")
    w("open_solution -reset sol1 -flow_target vitis")
    w(f"set_part {{{PART}}}")
    w(f"create_clock -period {period:.3f} -name default")
    w('if {[file exists $data/in.bin]} { csim_design -argv "$data $chunk" }')
    w("if {[info exists ::env(KG_NO_SYNTH)]} { exit }")
    w("csynth_design")
    w("if {[info exists ::env(KG_EXPORT)]} { export_design -format xo }")
    w("exit")
    w("")
    return "\n".join(L)
