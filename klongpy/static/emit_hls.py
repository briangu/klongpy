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
from .ir import Call, ExternCall, dump, fnspecs, walk

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


def _layout(program, entry):
    """The record layout of an entry, or a KlongStaticError saying why it has none."""
    sig = entry.decl.signature
    ext = reachable_externs(entry)
    if ext:
        raise _reject(program, entry, f"it calls the extern {', '.join(ext)}; hardware has no externs in this version",
                      "keep the extern out of the entry, or compute its value on the host and pass it as an argument")
    if sig.result.elem == "void":
        raise _reject(program, entry, "it returns nothing, so there is no result record to read back")
    params, off = [], 0
    for i, ty in enumerate(sig.params):
        if ty.elem not in ("int", "float"):
            raise _reject(program, entry, f"argument {i + 1} is a {ty.elem}")
        name = entry.fn.param_names[i] if i < len(entry.fn.param_names) else f"a{i}"
        params.append({"name": name, "elem": ty.elem, "shape": list(ty.shape), "offset": off, "words": ty.size})
        off += ty.size
    elems = {p["elem"] for p in params}
    if len(elems) > 1 and program.int_bits != 32:
        raise _reject(program, entry, "its arguments mix int and float and --int64 makes the words of the two differ in width",
                      "compile without --int64, or pass one kind of argument")
    r = sig.result
    return params, off, {"elem": r.elem, "shape": list(r.shape), "words": r.size}


def _bytes_of(elem, int_bits):
    return 4 if elem == "float" else int_bits // 8


def emit_hls(program, emitted, clock_mhz=DEFAULT_CLOCK_MHZ, source_name=None, float_bits=32):
    """
    The HLS files for a lowered program (``program``) whose C is ``emitted``
    (the result of emit_c.emit). Raises KlongStaticError for an entry that
    cannot be a kernel and ValueError for a float that is not float32.
    """
    if float_bits != 32:
        raise ValueError("--emit-hls: the card has float32 only in this version (--float64 is not available)")
    if not program.entries:
        raise KlongStaticError("the program has no .entry, so there is no kernel to make [hls]",
                               hint='declare one, for example .entry("f";"float -> float")')
    prefix = emitted.prefix
    ib = program.int_bits
    files, entries = {}, []
    flags = [f"-DKG_TRAP_FLAG={TRAP_FLAG}"] + (["-DKG_INT64"] if ib == 64 else [])
    for e in program.entries:
        params, in_words, result = _layout(program, e)
        if in_words == 0:
            raise _reject(program, e, "it takes no argument, so a call has no input record")
        stem = f"{prefix}_{e.c_name}"
        top = f"{stem}_kernel"
        mixed = len({p["elem"] for p in params}) > 1
        in_elem = "mixed" if mixed else (params[0]["elem"] if params else "int")
        out_elem = result["elem"]
        ent = {
            "name": e.decl.name, "c_name": e.c_name, "signature": str(e.decl.signature), "top": top,
            "params": params, "result": result,
            "in_words": in_words, "out_words": result["words"],
            "in_type": "uint32_t" if mixed else ("kg_float" if in_elem == "float" else "kg_int"),
            "out_type": "kg_float" if out_elem == "float" else "kg_int",
            "in_bytes": sum(_bytes_of(p["elem"], ib) * p["words"] for p in params) if not mixed else 4 * in_words,
            "out_bytes": _bytes_of(out_elem, ib) * result["words"],
            "files": {"top": f"{stem}_top.cpp", "tb": f"{stem}_tb.cpp", "tcl": f"hls_{e.c_name}.tcl"},
            "kernel_args": ["in", "out", "n", "reset", "status"],
        }
        files[ent["files"]["top"]] = _top(prefix, e, ent, params, mixed)
        files[ent["files"]["tb"]] = _tb(prefix, ent)
        files[ent["files"]["tcl"]] = _tcl(prefix, ent, clock_mhz, flags, source_name)
        entries.append(ent)
    key = cache_key(program, clock_mhz)
    manifest = {
        "format": 1,
        "emitter": EMITTER_VERSION,
        "program": prefix,
        "source": source_name or prefix + ".kg",
        "int_bits": ib,
        "float_bits": 32,
        "lanes": 1,
        "clock_mhz": clock_mhz,
        "cache_key": key,
        "cflags": flags,
        "tool": dict(TOOL, part=PART, platform=PLATFORM),
        "c_files": [prefix + ".c", prefix + ".h", "klong_rt.h"],
        "entries": entries,
        "status_bits": {"any_trap": 0, "bounds": 1, "div_zero": 2},
        "kernel": "void <top>(const in_type *in, out_type *out, unsigned n, unsigned reset, unsigned *status)",
    }
    files["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    return HlsOutput(prefix, files, manifest)


def cache_key(program, clock_mhz, lanes=1):
    """Design section 8: SHA-256 of the IR dump, the emitter version, the lanes, the clock, the platform and the tool version."""
    h = hashlib.sha256()
    for part in (dump(program), EMITTER_VERSION, f"lanes={lanes}", f"clock_mhz={clock_mhz}", f"platform={PLATFORM}",
                 f"tool={TOOL['name']} {TOOL['version']}"):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The top function
# ---------------------------------------------------------------------------

def _top(prefix, e, ent, params, mixed):
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
        w("static inline kg_float word_float(uint32_t v) { kg_float f; memcpy(&f, &v, sizeof f); return f; }")
        w("static inline kg_int word_int(uint32_t v) { kg_int i; memcpy(&i, &v, sizeof i); return i; }")
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
    w("static void core(const in_t buf[IN_WORDS], out_t res[OUT_WORDS], bool first, bool reset) {")
    w(f"    static {prefix}_t inst;")
    w("    static bool started = false;")
    w("    if (first) {")
    w("        KG_TRAP_FLAG = 0;")
    w("    }")
    w("    if (!started || (first && reset)) {")
    w(f"        {prefix}_init(&inst);")
    w("        started = true;")
    w("    }")
    call_args = []
    for i, p in enumerate(params):
        o = p["offset"]
        if not mixed:
            call_args.append(f"&buf[{o}]" if p["shape"] else f"buf[{o}]")
            continue
        ctype = "kg_float" if p["elem"] == "float" else "kg_int"
        conv = "word_float" if p["elem"] == "float" else "word_int"
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
    w(f'extern "C" void {ent["top"]}(const in_t *in, out_t *out, unsigned n, unsigned reset, unsigned *status) {{')
    w("#pragma HLS interface m_axi port=in offset=slave bundle=gmem0")
    w("#pragma HLS interface m_axi port=out offset=slave bundle=gmem1")
    w("#pragma HLS interface m_axi port=status offset=slave bundle=gmem2")
    for port in ("in", "out", "n", "reset", "status"):
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
    w("        core(ibuf, obuf, first, rst);")
    w("        store(obuf, op);")
    w("    }")
    w("    unsigned t = (n != 0) ? KG_TRAP_FLAG : 0u;")
    w("    *status = t ? (t | 1u) : 0u;")
    w("}")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# The test bench
# ---------------------------------------------------------------------------

def _tb(prefix, ent):
    L = []
    w = L.append
    w(f"/* {ent['files']['tb']} - generated by KlongPy (kgpy --emit-hls). Do not edit.")
    w(" * usage: tb [DIR [RECORDS_PER_CALL]]  reads DIR/in.bin, writes DIR/out.bin and DIR/status.bin (one 32-bit word,")
    w(" * the OR of the status of every call). RECORDS_PER_CALL 0 (the default) is one call for the whole file;")
    w(" * with a number, the first call has reset set and the rest continue the state. */")
    w("#include <stdio.h>")
    w("#include <stdlib.h>")
    w("#include <stdint.h>")
    w("#include <vector>")
    w("")
    w('extern "C" {')
    w(f'#include "{prefix}.h"')
    w("}")
    w(f"typedef {ent['in_type']} in_t;")
    w(f"typedef {ent['out_type']} out_t;")
    w(f"#define IN_WORDS {ent['in_words']}")
    w(f"#define OUT_WORDS {ent['out_words']}")
    w(f'extern "C" void {ent["top"]}(const in_t *in, out_t *out, unsigned n, unsigned reset, unsigned *status);')
    w("")
    w("int main(int argc, char **argv) {")
    w('    const char *dir = argc > 1 ? argv[1] : ".";')
    w("    unsigned per = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 10) : 0;")
    w("    char path[4096];")
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
    w("    std::vector<out_t> out((size_t)n * OUT_WORDS + 1);")
    w("    unsigned status = 0, all = 0;")
    w("    if (per == 0 || per > n) {")
    w("        per = n;")
    w("    }")
    w("    unsigned done = 0;")
    w("    do {")
    w("        unsigned m = n - done < per ? n - done : per;")
    w(f"        {ent['top']}(in.data() + (size_t)done * IN_WORDS, out.data() + (size_t)done * OUT_WORDS, m, done == 0, &status);")
    w("        all |= status;")
    w("        done += m;")
    w("    } while (done < n);")
    w('    snprintf(path, sizeof path, "%s/out.bin", dir);')
    w('    f = fopen(path, "wb");')
    w("    if (!f) {")
    w('        fprintf(stderr, "tb: cannot write %s\\n", path);')
    w("        return 2;")
    w("    }")
    w("    fwrite(out.data(), sizeof(out_t), (size_t)n * OUT_WORDS, f);")
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

def _tcl(prefix, ent, clock_mhz, flags, source_name):
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
