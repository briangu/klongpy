//! `kgpy fpga source` (`klongpy/static/emit_hls.py`, `emit_hls_program` and `emit_hls_main` of compile.py): the files Vitis HLS needs to
//! turn each `.entry` of a static program into a kernel: the top function, the test bench, the tcl script and the manifest. The C of the
//! program (`emit_c`) is the core, unchanged.
use crate::cli::render;
use crate::check::{Diag, Report};
use crate::compile::{write_files, EmitFailure, RUNTIME_HEADER};
use crate::emit_c::{ctype_of, hls_work, match_window, Emitted, HlsOptions, HLS_WORK_LIMIT};
use crate::fixed as fx;
use crate::ir::{dump, Entry, Program, K, V};
use crate::sig::{c_identifier, is_fixed};
use crate::util::py_float;
use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::rc::Rc;

pub const EMITTER_VERSION: &str = "klong-emit-hls/1";
pub const PART: &str = "xcu250-figd2104-2L-e";
pub const PLATFORM: &str = "xilinx_u250_gen3x16_xdma_4_1_202210_1";
pub const TOOL_NAME: &str = "Vitis HLS";
pub const TOOL_VERSION: &str = "2022.2";
pub const TOOL_FLOW_TARGET: &str = "vitis";
pub const DEFAULT_CLOCK_MHZ: u32 = 300;
const TRAP_FLAG: &str = "kg_hls_trap";
const TRIPCOUNT: usize = 1024;
const STREAM_DEPTH: usize = 32;
const WIDE_BITS: usize = 2048;

// ---------------------------------------------------------------------------
// JSON as `json.dumps(obj, indent=2)` writes it
// ---------------------------------------------------------------------------

pub mod json {
    /// A JSON value; objects keep the order of their keys.
    #[derive(Clone, Debug)]
    pub enum J {
        Null,
        Bool(bool),
        Int(i128),
        Flt(f64),
        Str(String),
        Arr(Vec<J>),
        Obj(Vec<(String, J)>),
    }

    pub fn s(t: &str) -> J {
        J::Str(t.to_string())
    }

    fn string(t: &str, out: &mut String) {
        out.push('"');
        for c in t.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 || (c as u32) > 0x7e && (c as u32) != 0x7f => {
                    let mut buf = [0u16; 2];
                    for u in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{:04x}", u));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }

    fn flt(x: f64) -> String {
        if x.is_nan() {
            "NaN".to_string()
        } else if x.is_infinite() {
            if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
        } else {
            crate::util::py_float(x)
        }
    }

    fn write(v: &J, level: usize, out: &mut String) {
        match v {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Int(i) => out.push_str(&i.to_string()),
            J::Flt(f) => out.push_str(&flt(*f)),
            J::Str(t) => string(t, out),
            J::Arr(a) => {
                if a.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    out.push_str(&" ".repeat(2 * (level + 1)));
                    write(x, level + 1, out);
                }
                out.push('\n');
                out.push_str(&" ".repeat(2 * level));
                out.push(']');
            }
            J::Obj(o) => {
                if o.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                for (i, (k, x)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    out.push_str(&" ".repeat(2 * (level + 1)));
                    string(k, out);
                    out.push_str(": ");
                    write(x, level + 1, out);
                }
                out.push('\n');
                out.push_str(&" ".repeat(2 * level));
                out.push('}');
            }
        }
    }

    /// `json.dumps(v, indent=2)`
    pub fn dumps(v: &J) -> String {
        let mut out = String::new();
        write(v, 0, &mut out);
        out
    }
}

use json::{s as js, J};

fn obj(items: Vec<(&str, J)>) -> J {
    J::Obj(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn int(n: usize) -> J {
    J::Int(n as i128)
}

fn ints(v: &[usize]) -> J {
    J::Arr(v.iter().map(|x| int(*x)).collect())
}

// ---------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|x| format!("{:08x}", x)).collect()
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A `KlongStaticError` of emit_hls.py (rule `hls`; the position is the declaration of the entry, or none) or a `ValueError`.
#[derive(Debug)]
pub enum HlsError {
    Static { message: String, hint: Option<String>, span: Option<(usize, usize)>, src: usize },
    Value(String),
}

fn reject(e: &Entry, why: &str, hint: Option<&str>) -> HlsError {
    let d = &e.decl;
    HlsError::Static {
        message: format!("entry {} ({}) cannot be a hardware kernel: {}", d.name, d.signature, why),
        hint: hint.map(String::from),
        span: Some(d.span),
        src: d.src,
    }
}

impl HlsError {
    /// The text `KlongError.render` gives, or `path: message` for a ValueError; and the exit status.
    pub fn render(&self, path: &str, r: &Report) -> String {
        match self {
            HlsError::Value(m) => format!("{}: {}", path, m),
            HlsError::Static { message, hint, span: Some(sp), src } => {
                let d = Diag { rule: "hls", span: *sp, src: *src, message: message.clone(), hint: hint.clone(), function: None };
                render(&d, r)
            }
            HlsError::Static { message, hint, span: None, .. } => {
                let mut out = format!("static error: {} [hls]", message);
                if let Some(h) = hint {
                    if !h.is_empty() {
                        out.push_str(&format!("\n  hint: {}", h));
                    }
                }
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The layout of the records
// ---------------------------------------------------------------------------

pub struct HlsOutput {
    pub prefix: String,
    /// {name: text}, in sorted order
    pub files: BTreeMap<String, String>,
    pub manifest: J,
    pub entries: Vec<Ent>,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: String,
    pub elem: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub words: usize,
}

#[derive(Clone, Debug)]
pub struct Reg {
    pub name: String,
    pub elem: String,
    pub ctype: &'static str,
    pub cid: String,
    pub default: Default,
    pub lanes: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Default {
    I(i64),
    F(f64),
}

impl Default {
    fn repr(&self) -> String {
        match self {
            Default::I(i) => i.to_string(),
            Default::F(f) => py_float(*f),
        }
    }
    fn json(&self) -> J {
        match self {
            Default::I(i) => J::Int(*i as i128),
            Default::F(f) => J::Flt(*f),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Table {
    pub order: Vec<String>,
    pub lanes: usize,
    pub word_bytes: usize,
    pub words: usize,
    pub bytes: usize,
}

#[derive(Clone, Debug)]
pub struct WindowInfo {
    pub taps: usize,
    pub outputs: usize,
    pub offset: usize,
    pub inputs: usize,
    pub op: String,
}

/// One entry of the manifest.
#[derive(Clone, Debug)]
pub struct Ent {
    pub name: String,
    pub c_name: String,
    pub signature: String,
    pub top: String,
    pub params: Vec<Param>,
    pub res_elem: String,
    pub res_shape: Vec<usize>,
    pub in_words: usize,
    pub out_words: usize,
    pub in_type: String,
    pub out_type: String,
    pub in_bytes: usize,
    pub out_bytes: usize,
    pub f_top: String,
    pub f_tb: String,
    pub f_tcl: String,
    pub kernel_args: Vec<String>,
    pub table: Option<Table>,
    /// the reader of a summary pair: (name, c_name, signature)
    pub summary: Option<(String, String, String)>,
    pub lane_files: Vec<String>,
    pub window: Option<WindowInfo>,
    pub streams: Option<(&'static str, &'static str)>,
}

impl Ent {
    fn json(&self) -> J {
        let params: Vec<J> = self.params.iter().map(|p| obj(vec![("name", js(&p.name)), ("elem", js(&p.elem)), ("shape", ints(&p.shape)), ("offset", int(p.offset)), ("words", int(p.words))])).collect();
        let mut o: Vec<(&str, J)> = vec![
            ("name", js(&self.name)),
            ("c_name", js(&self.c_name)),
            ("signature", js(&self.signature)),
            ("top", js(&self.top)),
            ("params", J::Arr(params)),
            ("result", obj(vec![("elem", js(&self.res_elem)), ("shape", ints(&self.res_shape)), ("words", int(self.out_words))])),
            ("in_words", int(self.in_words)),
            ("out_words", int(self.out_words)),
            ("in_type", js(&self.in_type)),
            ("out_type", js(&self.out_type)),
            ("in_bytes", int(self.in_bytes)),
            ("out_bytes", int(self.out_bytes)),
            ("files", obj(vec![("top", js(&self.f_top)), ("tb", js(&self.f_tb)), ("tcl", js(&self.f_tcl))])),
            ("kernel_args", J::Arr(self.kernel_args.iter().map(|a| js(a)).collect())),
        ];
        if let Some(t) = &self.table {
            o.push(("register_table", table_json(t)));
        }
        if let Some((n, c, sg)) = &self.summary {
            o.push(("output", js("summary")));
            o.push(("summary", obj(vec![("entry", js(n)), ("c_name", js(c)), ("signature", js(sg))])));
        }
        if !self.lane_files.is_empty() {
            o.push(("structure", js("lane-processes")));
            o.push(("lane_files", J::Arr(self.lane_files.iter().map(|f| js(f)).collect())));
        }
        if let Some(w) = &self.window {
            o.push(("window", obj(vec![("taps", int(w.taps)), ("outputs", int(w.outputs)), ("offset", int(w.offset)), ("inputs", int(w.inputs)), ("op", js(&w.op))])));
        }
        if let Some((i, out)) = &self.streams {
            o.push(("streams", obj(vec![("in", js(i)), ("out", js(out))])));
        }
        obj(o)
    }
}

fn table_json(t: &Table) -> J {
    obj(vec![
        ("order", J::Arr(t.order.iter().map(|n| js(n)).collect())),
        ("lanes", int(t.lanes)),
        ("word_bytes", int(t.word_bytes)),
        ("words", int(t.words)),
        ("bytes", int(t.bytes)),
    ])
}

fn reg_json(r: &Reg) -> J {
    obj(vec![("name", js(&r.name)), ("elem", js(&r.elem)), ("ctype", js(r.ctype)), ("cid", js(&r.cid)), ("default", r.default.json()), ("lanes", int(r.lanes)), ("bytes", int(r.bytes))])
}

/// The names of the externs an entry can reach through the functions it calls.
pub fn reachable_externs(entry: &Entry) -> Vec<String> {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut names: Vec<String> = Vec::new();
    let mut todo = vec![entry.f.clone()];
    while let Some(f) = todo.pop() {
        if !seen.insert(Rc::as_ptr(&f) as usize) {
            continue;
        }
        for n in f.body.walked() {
            match &n.k {
                K::ExternCall { ext, .. } => {
                    if !names.contains(&ext.name) {
                        names.push(ext.name.clone());
                    }
                }
                K::Call { f, .. } => todo.push(f.clone()),
                _ => {}
            }
            for g in n.fnspecs() {
                todo.push(g.clone());
            }
        }
    }
    names
}

fn bytes_of(elem: &str, int_bits: u32) -> usize {
    if is_fixed(elem) {
        if fx::spec_of(elem).w <= 32 {
            4
        } else {
            8
        }
    } else if elem == "float" {
        4
    } else {
        (int_bits / 8) as usize
    }
}

/// `(readers, pairs)`: the entries (indices) that are not kernels of their own and, for each driver, the reader it calls at the end.
pub fn roles(program: &Program, wrapped: bool) -> (Vec<usize>, Vec<(usize, usize)>) {
    let drivers: Vec<usize> = (0..program.entries.len()).filter(|i| {
        let s = &program.entries[*i].decl.signature;
        !s.params.is_empty() && s.result.elem == "void"
    }).collect();
    let readers: Vec<usize> = (0..program.entries.len()).filter(|i| {
        let s = &program.entries[*i].decl.signature;
        s.params.is_empty() && s.result.elem != "void"
    }).collect();
    if wrapped || drivers.is_empty() || readers.len() != 1 {
        return (Vec::new(), Vec::new());
    }
    let r = readers[0];
    (readers, drivers.into_iter().map(|d| (d, r)).collect())
}

struct Layout {
    params: Vec<Param>,
    in_words: usize,
    res_elem: String,
    res_shape: Vec<usize>,
    res_words: usize,
}

fn layout(program: &Program, entry: &Entry, reader: Option<&Entry>) -> Result<Layout, HlsError> {
    let sig = &entry.decl.signature;
    let mut ext = reachable_externs(entry);
    if let Some(r) = reader {
        for n in reachable_externs(r) {
            if !reachable_externs(entry).contains(&n) {
                ext.push(n);
            }
        }
    }
    let ib = program.int_bits;
    if !ext.is_empty() {
        return Err(reject(
            entry,
            &format!("it calls the extern {}; hardware has no externs in this version", ext.join(", ")),
            Some("keep the extern out of the entry, or compute its value on the host and pass it as an argument"),
        ));
    }
    if sig.result.elem == "void" && reader.is_none() {
        return Err(reject(
            entry,
            "it returns nothing, so there is no result record to read back",
            Some("declare one more entry without arguments that returns the state, for example .entry(\"summary\";\"-> float[3]\"): the kernel then runs its calls and writes that result once"),
        ));
    }
    let mut params = Vec::new();
    let mut off = 0usize;
    for (i, ty) in sig.params.iter().enumerate() {
        if ty.elem != "int" && ty.elem != "float" && !is_fixed(&ty.elem) {
            return Err(reject(entry, &format!("argument {} is a {}", i + 1, ty.elem), None));
        }
        let name = entry.f.param_names.get(i).cloned().unwrap_or_else(|| format!("a{}", i));
        let size: usize = ty.shape.iter().product();
        params.push(Param { name, elem: ty.elem.clone(), shape: ty.shape.clone(), offset: off, words: size });
        off += size;
    }
    let widths: HashSet<usize> = params.iter().map(|p| bytes_of(&p.elem, ib)).collect();
    if widths.len() > 1 {
        return Err(reject(
            entry,
            "its arguments mix words of different widths (--int64 and fixed<W,I> with W over 32 are 8 bytes, a float is 4), which one record cannot hold",
            Some("make the arguments words of one width"),
        ));
    }
    let r = reader.map(|r| &r.decl.signature.result).unwrap_or(&sig.result);
    Ok(Layout { params, in_words: off, res_elem: r.elem.clone(), res_shape: r.shape.clone(), res_words: r.shape.iter().product() })
}

const RESERVED: &[&str] = &["in", "out", "n", "reset", "status", "params", "inst", "started", "pw", "sin", "sout", "sst", "win", "rec", "buf", "res", "obuf", "ibuf", "first", "load", "core", "store"];

/// The `.param` registers of the program.
pub fn registers(program: &Program, lanes: Option<usize>) -> Result<Vec<Reg>, HlsError> {
    let mut out = Vec::new();
    for g in program.globals.iter().filter(|g| g.param) {
        if RESERVED.contains(&c_identifier(&g.name).as_str()) {
            let mut names: Vec<&str> = RESERVED.to_vec();
            names.sort();
            return Err(HlsError::Static {
                message: format!("the parameter {} cannot be a kernel register: its name is one that the generated kernel uses for its own arguments ({})", g.name, names.join(", ")),
                hint: Some(format!("rename the parameter, for example {}_p", g.name)),
                span: None,
                src: 0,
            });
        }
        let first_i = |v: &V| match v {
            V::I(i) => *i,
            V::F(f) => *f as i64,
            V::AI(a) => a.first().copied().unwrap_or(0),
            V::AF(a) => a.first().map(|x| *x as i64).unwrap_or(0),
            V::S(_) => 0,
        };
        let first_f = |v: &V| match v {
            V::I(i) => *i as f64,
            V::F(f) => *f,
            V::AI(a) => a.first().map(|x| *x as f64).unwrap_or(0.0),
            V::AF(a) => a.first().copied().unwrap_or(0.0),
            V::S(_) => 0.0,
        };
        let default = if g.ty.elem != "float" { Default::I(first_i(&g.value)) } else { Default::F(first_f(&g.value)) };
        out.push(Reg {
            name: g.name.clone(),
            elem: g.ty.elem.clone(),
            ctype: ctype_of(&g.ty.elem),
            cid: c_identifier(&g.name),
            default,
            lanes: lanes.unwrap_or(1).max(1),
            bytes: bytes_of(&g.ty.elem, program.int_bits),
        });
    }
    Ok(out)
}

/// How the parameters of a kernel with lanes reach it: one table in card memory. None without lanes.
pub fn register_table(regs: &[Reg], lanes: Option<usize>) -> Option<Table> {
    let lanes = lanes.filter(|l| *l > 0)?;
    if regs.is_empty() {
        return None;
    }
    let wb = regs.iter().map(|r| r.bytes).max().unwrap();
    Some(Table { order: regs.iter().map(|r| r.name.clone()).collect(), lanes, word_bytes: wb, words: regs.len() * lanes, bytes: regs.len() * lanes * wb })
}

// ---------------------------------------------------------------------------
// Emitting
// ---------------------------------------------------------------------------

struct Lines(Vec<String>);

impl Lines {
    fn w(&mut self, s: impl Into<String>) {
        self.0.push(s.into());
    }
    fn text(self) -> String {
        self.0.join("\n")
    }
}

/// The HLS files for a lowered program whose C is `emitted`. `lanes`: None or N >= 1 (the caller checks).
pub fn emit_hls(program: &Program, emitted: &Emitted, clock_mhz: u32, source_name: Option<&str>, wrapped: bool, lanes: Option<usize>, reassociate: bool, mode: Option<&str>) -> Result<HlsOutput, HlsError> {
    if program.entries.is_empty() {
        return Err(HlsError::Static {
            message: "the program has no .entry, so there is no kernel to make".to_string(),
            hint: Some("declare one, for example .entry(\"f\";\"float -> float\")".to_string()),
            span: None,
            src: 0,
        });
    }
    let lanes = lanes.filter(|l| *l > 0);
    let prefix = emitted.prefix.clone();
    let ib = program.int_bits;
    let regs = registers(program, lanes)?;
    let table = register_table(&regs, lanes);
    let mut files: BTreeMap<String, String> = BTreeMap::new();
    let mut entries: Vec<Ent> = Vec::new();
    let mut flags = vec![format!("-DKG_TRAP_FLAG={}", TRAP_FLAG)];
    if ib == 64 {
        flags.push("-DKG_INT64".to_string());
    }
    let (readers, pairs) = roles(program, wrapped);
    for (idx, e) in program.entries.iter().enumerate() {
        if readers.contains(&idx) {
            continue;
        }
        let reader: Option<&Entry> = pairs.iter().find(|(d, _)| *d == idx).map(|(_, r)| &program.entries[*r]);
        let lay = layout(program, e, reader)?;
        if lay.in_words == 0 {
            return Err(reject(e, "it takes no argument, so a call has no input record", None));
        }
        let params = lay.params.clone();
        let stem = format!("{}_{}", prefix, e.c_name);
        let top = format!("{}_kernel", stem);
        let elems: HashSet<&str> = params.iter().map(|p| p.elem.as_str()).collect();
        let mixed = elems.len() > 1;
        let wbytes = bytes_of(&params[0].elem, ib);
        let lw = lanes.unwrap_or(1);
        let out_words = lay.res_words * lw;
        let mut res_shape = lay.res_shape.clone();
        if let Some(l) = lanes {
            res_shape.insert(0, l);
        }
        let mut ent = Ent {
            name: e.decl.name.clone(),
            c_name: e.c_name.clone(),
            signature: e.decl.signature.to_string(),
            top: top.clone(),
            params: params.clone(),
            res_elem: lay.res_elem.clone(),
            res_shape,
            in_words: lay.in_words,
            out_words,
            in_type: if mixed { format!("uint{}_t", 8 * wbytes) } else { ctype_of(&params[0].elem).to_string() },
            out_type: ctype_of(&lay.res_elem).to_string(),
            in_bytes: params.iter().map(|p| bytes_of(&p.elem, ib) * p.words).sum(),
            out_bytes: bytes_of(&lay.res_elem, ib) * out_words,
            f_top: format!("{}_top.cpp", stem),
            f_tb: format!("{}_tb.cpp", stem),
            f_tcl: format!("hls_{}.tcl", e.c_name),
            kernel_args: Vec::new(),
            table: table.clone(),
            summary: reader.map(|r| (r.decl.name.clone(), r.c_name.clone(), r.decl.signature.to_string())),
            lane_files: Vec::new(),
            window: None,
            streams: None,
        };
        let mut ka: Vec<String> = ["in", "out", "n", "reset"].iter().map(|s| s.to_string()).collect();
        if table.is_some() {
            ka.push("params".to_string());
        } else {
            ka.extend(regs.iter().map(|r| r.name.clone()));
        }
        ka.push("status".to_string());
        ent.kernel_args = ka;
        if wrapped {
            let t = top_a(&prefix, e, &ent, &params, mixed, &regs);
            files.insert(ent.f_top.clone(), t);
        } else if reader.is_some() && lanes.map(|l| l > 1).unwrap_or(false) && !mixed {
            let l = lanes.unwrap();
            ent.lane_files = (0..l).map(|i| format!("lane_{}.c", i)).collect();
            for i in 0..l {
                files.insert(format!("lane_{}.c", i), lane_c(&prefix, i, &emitted.h));
            }
            let t = top_lanes(program, &prefix, e, &ent, &params, &regs, l, reader.unwrap(), &emitted.h);
            files.insert(ent.f_top.clone(), t);
        } else {
            let t = top_b(program, &prefix, e, &mut ent, &params, mixed, &regs, lanes, mode, reader);
            files.insert(ent.f_top.clone(), t);
        }
        files.insert(ent.f_tb.clone(), tb(&prefix, &ent, &regs, lanes, table.as_ref()));
        files.insert(ent.f_tcl.clone(), tcl(&prefix, &ent, clock_mhz, &flags, !wrapped));
        entries.push(ent);
    }
    if !wrapped {
        files.insert("kg_hls_stub.h".to_string(), STREAM_STUB.to_string());
    }
    let key = cache_key(program, clock_mhz, lanes, wrapped, reassociate);
    let mut c_files: Vec<String> = vec![format!("{}.c", prefix), format!("{}.h", prefix), "klong_rt.h".to_string()];
    let mut kernel = "void <top>(const in_type *in, out_type *out, unsigned n, unsigned reset, unsigned *status)".to_string();
    if !regs.is_empty() {
        if let Some(t) = &table {
            kernel = format!("void <top>(const in_type *in, out_type *out, unsigned n, unsigned reset, const uint{}_t *params, unsigned *status)", 8 * t.word_bytes);
        } else {
            let decl: Vec<String> = regs.iter().map(|r| format!("{} {}", r.ctype, r.name)).collect();
            kernel = format!("void <top>(const in_type *in, out_type *out, unsigned n, unsigned reset, {}, unsigned *status)", decl.join(", "));
        }
    }
    for e in &entries {
        c_files.extend(e.lane_files.iter().cloned());
    }
    if !wrapped {
        c_files.push("kg_hls_stub.h".to_string());
    }
    let source = source_name.map(String::from).unwrap_or_else(|| format!("{}.kg", prefix));
    let mut m: Vec<(&str, J)> = vec![
        ("format", int(1)),
        ("emitter", js(EMITTER_VERSION)),
        ("program", js(&prefix)),
        ("source", js(&source)),
        ("int_bits", J::Int(ib as i128)),
        ("float_bits", int(32)),
        ("lanes", int(lanes.unwrap_or(1))),
        ("clock_mhz", J::Int(clock_mhz as i128)),
        ("cache_key", js(&key)),
        ("cflags", J::Arr(flags.iter().map(|f| js(f)).collect())),
        ("tool", obj(vec![("name", js(TOOL_NAME)), ("version", js(TOOL_VERSION)), ("flow_target", js(TOOL_FLOW_TARGET)), ("part", js(PART)), ("platform", js(PLATFORM))])),
        ("c_files", J::Arr(c_files.iter().map(|f| js(f)).collect())),
        ("entries", J::Arr(entries.iter().map(|e| e.json()).collect())),
        ("status_bits", obj(vec![("any_trap", int(0)), ("bounds", int(1)), ("div_zero", int(2))])),
        ("kernel", js(&kernel)),
    ];
    if !regs.is_empty() {
        m.push(("registers", J::Arr(regs.iter().map(reg_json).collect())));
        if let Some(t) = &table {
            m.push(("register_table", table_json(t)));
        }
    }
    if !wrapped {
        m.push(("stage", js("B")));
        m.push(("lane_dimension", J::Bool(lanes.is_some())));
        m.push(("reassociate", J::Bool(reassociate)));
    }
    let manifest = obj(m);
    files.insert("manifest.json".to_string(), json::dumps(&manifest) + "\n");
    Ok(HlsOutput { prefix, files, manifest, entries })
}

/// SHA-256 of the IR dump, the emitter version, the lanes, the clock, the platform and the tool version (and, for the streamable back
/// end, the stage and the reassociation).
pub fn cache_key(program: &Program, clock_mhz: u32, lanes: Option<usize>, wrapped: bool, reassociate: bool) -> String {
    let b = |x: bool| if x { "True" } else { "False" };
    let mut parts = vec![dump(program), EMITTER_VERSION.to_string(), format!("lanes={}", lanes.filter(|l| *l > 0).unwrap_or(1)), format!("clock_mhz={}", clock_mhz), format!("platform={}", PLATFORM), format!("tool={} {}", TOOL_NAME, TOOL_VERSION)];
    if !wrapped {
        parts.push("stage=B".to_string());
        parts.push(format!("lane_dimension={}", b(lanes.map(|l| l > 0).unwrap_or(false))));
        parts.push(format!("reassociate={}", b(reassociate)));
    }
    let mut data: Vec<u8> = Vec::new();
    for p in parts {
        data.extend_from_slice(p.as_bytes());
        data.push(0);
    }
    sha256_hex(&data)
}

fn reg_decl(regs: &[Reg], lanes: Option<usize>) -> String {
    regs.iter().map(|r| format!(", {} {}{}", r.ctype, r.cid, lanes.map(|l| format!("[{}]", l)).unwrap_or_default())).collect()
}

fn header_comment_tail(l: &mut Lines) {
    l.w(" * status bit 0: some trap happened; bit 1: an index out of range; bit 2: an integer division by zero.");
    l.w(format!(" * Build with -DKG_TRAP_FLAG={} (and -DKG_INT64 for a 64-bit kg_int; see hls_*.tcl). */", TRAP_FLAG));
}

fn trap_guard(l: &mut Lines) {
    l.w("#ifndef KG_TRAP_FLAG");
    l.w(format!("#error \"compile with -DKG_TRAP_FLAG={} (the generated C records traps in that counter)\"", TRAP_FLAG));
    l.w("#endif");
    l.w("");
}

fn stream_includes(l: &mut Lines) {
    l.w("#include <stddef.h>");
    l.w("#include <stdint.h>");
    l.w("#include <string.h>");
    l.w("#if defined(__has_include)");
    l.w("#if __has_include(<hls_stream.h>)");
    l.w("#include <hls_stream.h>");
    l.w("#else");
    l.w("#include \"kg_hls_stub.h\"");
    l.w("#endif");
    l.w("#else");
    l.w("#include <hls_stream.h>");
    l.w("#endif");
    l.w("");
}

fn call_args_a(params: &[Param], mixed: bool, buf: &str, l: &mut Lines, indent: &str, word: &str) -> Vec<String> {
    let mut args = Vec::new();
    for (i, p) in params.iter().enumerate() {
        let o = p.offset;
        if !mixed {
            args.push(if p.shape.is_empty() { format!("{}[{}]", buf, o) } else { format!("&{}[{}]", buf, o) });
            continue;
        }
        let ctype = ctype_of(&p.elem);
        if !p.shape.is_empty() {
            l.w(format!("{}{} a{}[{}];", indent, ctype, i, p.words));
            l.w(format!("{}for (unsigned j = 0; j < {}; j++) {{", indent, p.words));
            l.w(format!("{}    a{}[j] = {}<{}>({}[{} + j]);", indent, i, word, ctype, buf, o));
            l.w(format!("{}}}", indent));
            args.push(format!("a{}", i));
        } else {
            args.push(format!("{}<{}>({}[{}])", word, ctype, buf, o));
        }
    }
    args
}

fn top_a(prefix: &str, e: &Entry, ent: &Ent, params: &[Param], mixed: bool, regs: &[Reg]) -> String {
    let stem = format!("{}_{}", prefix, e.c_name);
    let (inw, outw) = (ent.in_words, ent.out_words);
    let (tin, tout) = (&ent.in_type, &ent.out_type);
    let mut l = Lines(Vec::new());
    l.w(format!("/* {} - generated by KlongPy (kgpy fpga source) from {}. Do not edit.", ent.f_top, prefix));
    l.w(format!(" * Kernel {} for entry {}: {}.", ent.top, ent.name, ent.signature));
    l.w(format!(" * One input record is {} word(s) of {}: the arguments in order, arrays flattened row-major.", inw, tin));
    l.w(format!(" * One output record is {} word(s) of {}. Each call of the kernel makes n calls of the entry.", outw, tout));
    header_comment_tail(&mut l);
    l.w("#include <stddef.h>");
    l.w("#include <stdint.h>");
    l.w("#include <string.h>");
    l.w("");
    trap_guard(&mut l);
    l.w("extern \"C\" {");
    l.w(format!("#include \"{}.h\"", prefix));
    l.w("extern unsigned KG_TRAP_FLAG;");
    l.w("}");
    l.w("");
    l.w(format!("#define IN_WORDS {}", inw));
    l.w(format!("#define OUT_WORDS {}", outw));
    l.w(format!("typedef {} in_t;", tin));
    l.w(format!("typedef {} out_t;", tout));
    l.w("");
    if mixed {
        l.w("template <class T, class W> static inline T from_word(W v) { T t; memcpy(&t, &v, sizeof t); return t; }");
        l.w("");
    }
    l.w("/* load: one record from memory to a buffer */");
    l.w("static void load(const in_t *in, in_t buf[IN_WORDS]) {");
    l.w("    for (unsigned j = 0; j < IN_WORDS; j++) {");
    l.w("#pragma HLS pipeline II=1");
    l.w("        buf[j] = in[j];");
    l.w("    }");
    l.w("}");
    l.w("");
    l.w("/* core: the C of the program, unchanged, on one record */");
    l.w(format!("static void core(const in_t buf[IN_WORDS], out_t res[OUT_WORDS], bool first, bool reset{}) {{", reg_decl(regs, None)));
    l.w(format!("    static {}_t inst;", prefix));
    l.w("    static bool started = false;");
    l.w("    if (first) {");
    l.w("        KG_TRAP_FLAG = 0;");
    l.w("    }");
    l.w("    if (!started || (first && reset)) {");
    l.w(format!("        {}_init(&inst);", prefix));
    l.w("        started = true;");
    l.w("    }");
    for r in regs {
        l.w(format!("    {}_set_{}(&inst, {});", prefix, r.cid, r.cid));
    }
    let args = call_args_a(params, mixed, "buf", &mut l, "    ", "from_word");
    let a: String = args.iter().map(|x| format!(", {}", x)).collect();
    if !e.decl.signature.result.shape.is_empty() {
        l.w(format!("    {}(&inst{}, res);", stem, a));
    } else {
        l.w(format!("    res[0] = {}(&inst{});", stem, a));
    }
    l.w("}");
    l.w("");
    l.w("/* store: one result record from a buffer to memory */");
    l.w("static void store(const out_t res[OUT_WORDS], out_t *out) {");
    l.w("    for (unsigned j = 0; j < OUT_WORDS; j++) {");
    l.w("#pragma HLS pipeline II=1");
    l.w("        out[j] = res[j];");
    l.w("    }");
    l.w("}");
    l.w("");
    l.w(format!("extern \"C\" void {}(const in_t *in, out_t *out, unsigned n, unsigned reset{}, unsigned *status) {{", ent.top, reg_decl(regs, None)));
    l.w("#pragma HLS interface m_axi port=in offset=slave bundle=gmem0");
    l.w("#pragma HLS interface m_axi port=out offset=slave bundle=gmem1");
    l.w("#pragma HLS interface m_axi port=status offset=slave bundle=gmem2");
    let mut ports: Vec<String> = ["in", "out", "n", "reset"].iter().map(|s| s.to_string()).collect();
    ports.extend(regs.iter().map(|r| r.cid.clone()));
    ports.push("status".to_string());
    for port in ports {
        l.w(format!("#pragma HLS interface s_axilite port={} bundle=control", port));
    }
    l.w("#pragma HLS interface s_axilite port=return bundle=control");
    l.w("    for (unsigned i = 0; i < n; i++) {");
    l.w(format!("#pragma HLS loop_tripcount min=1 max={}", TRIPCOUNT));
    l.w("        in_t ibuf[IN_WORDS];");
    l.w("        out_t obuf[OUT_WORDS];");
    l.w("        const in_t *ip = in + (size_t)i * IN_WORDS;");
    l.w("        out_t *op = out + (size_t)i * OUT_WORDS;");
    l.w("        const bool first = (i == 0);");
    l.w("        const bool rst = (reset != 0);");
    l.w("#pragma HLS dataflow");
    l.w("        load(ip, ibuf);");
    let rc: String = regs.iter().map(|r| format!(", {}", r.cid)).collect();
    l.w(format!("        core(ibuf, obuf, first, rst{});", rc));
    l.w("        store(obuf, op);");
    l.w("    }");
    l.w("    unsigned t = (n != 0) ? KG_TRAP_FLAG : 0u;");
    l.w("    *status = t ? (t | 1u) : 0u;");
    l.w("}");
    l.w("");
    l.text()
}

pub const STREAM_STUB: &str = "/* kg_hls_stub.h - generated by KlongPy (kgpy fpga source). A stand-in for hls::stream, used only when the
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
";

#[allow(clippy::too_many_arguments)]
fn top_b(program: &Program, prefix: &str, e: &Entry, ent: &mut Ent, params: &[Param], mixed: bool, regs: &[Reg], lanes: Option<usize>, mode: Option<&str>, reader: Option<&Entry>) -> String {
    let stem = format!("{}_{}", prefix, e.c_name);
    let (inw, outw) = (ent.in_words, ent.out_words);
    let lcount = lanes.unwrap_or(1);
    let (tin, tout) = (ent.in_type.clone(), ent.out_type.clone());
    let res_words = ent.out_words / lcount;
    let sig = &e.decl.signature;
    let summary = reader.is_some();
    let tot = if summary { "OUT_WORDS" } else { "n * OUT_WORDS" };
    let win = if lanes.is_none() && regs.is_empty() && !summary { match_window(program, e) } else { None };
    let pipelined = mode != Some("block");
    let in_bits = inw * bytes_of(&params[0].elem, program.int_bits) * 8;
    let out_bits = outw * bytes_of(&sig.result.elem, program.int_bits) * 8;
    let wide_in = in_bits <= WIDE_BITS && win.is_none();
    let wide_out = out_bits <= WIDE_BITS && win.is_none();
    if let Some(w) = &win {
        ent.window = Some(WindowInfo { taps: w.taps, outputs: w.outputs, offset: w.offset, inputs: w.inputs, op: w.op.clone() });
    }
    ent.streams = Some((if wide_in { "record" } else { "word" }, if wide_out { "record" } else { "word" }));
    let mut l = Lines(Vec::new());
    l.w(format!("/* {} - generated by KlongPy (kgpy fpga source) from {}. Do not edit.", ent.f_top, prefix));
    l.w(format!(" * Streamable kernel {} for entry {}: {}{}", ent.top, ent.name, ent.signature, lanes.map(|n| format!(", {} lane(s).", n)).unwrap_or_else(|| ".".to_string())));
    l.w(format!(" * One input record is {} word(s) of {}: the arguments in order, arrays flattened row-major.", inw, tin));
    l.w(format!(" * One output record is {} word(s) of {}{}", outw, tout, if lanes.is_some() { format!(": {} lanes of {} word(s), lane first.", lcount, res_words) } else { ".".to_string() }));
    if let Some(w) = &win {
        l.w(format!(" * The entry is a sliding window of {} samples: the core is a shift register that makes one output per input word once the window is full.", w.taps));
    } else if let Some(r) = reader {
        l.w(format!(" * Each call of the kernel makes n calls of the entry, which return nothing, and then one call of {} ({}) per lane: that is the one output record.", r.decl.name, r.decl.signature));
    } else {
        l.w(format!(
            " * Each call of the kernel makes n calls of the entry. The core is {}",
            if pipelined { "one pipelined region (II=1 when the program allows it)." } else { "a block program: its loops are pipelined, not the call." }
        ));
    }
    header_comment_tail(&mut l);
    stream_includes(&mut l);
    trap_guard(&mut l);
    l.w("extern \"C\" {");
    l.w(format!("#include \"{}.h\"", prefix));
    l.w("extern unsigned KG_TRAP_FLAG;");
    l.w("}");
    l.w("");
    l.w(format!("#define IN_WORDS {}", inw));
    l.w(format!("#define RES_WORDS {}", res_words));
    l.w(format!("#define LANES {}", lcount));
    l.w("#define OUT_WORDS (LANES * RES_WORDS)");
    l.w(format!("typedef {} in_t;", tin));
    l.w(format!("typedef {} out_t;", tout));
    if let Some(rt) = &ent.table {
        l.w(format!("typedef uint{}_t reg_t;", 8 * rt.word_bytes));
        l.w(format!("#define REG_WORDS {}", rt.words));
        l.w("template <class T, class W> static inline T from_bits(W v) { T t; memcpy(&t, &v, sizeof t); return t; }");
    }
    if wide_in {
        l.w("typedef struct { in_t w[IN_WORDS]; } in_rec_t;");
    }
    if wide_out {
        l.w("typedef struct { out_t w[OUT_WORDS]; } out_rec_t;");
    }
    let sin_t = if wide_in { "in_rec_t" } else { "in_t" };
    let sout_t = if wide_out { "out_rec_t" } else { "out_t" };
    l.w("");
    if mixed {
        l.w("template <class T, class W> static inline T from_word(W v) { T t; memcpy(&t, &v, sizeof t); return t; }");
        l.w("");
    }
    // ---- load
    l.w("/* load: the input words, one per cycle */");
    l.w(format!("static void load(const in_t *in, hls::stream<{}> &s, uint64_t n) {{", sin_t));
    if wide_in {
        l.w("    in_rec_t rec;");
        l.w("    unsigned j = 0;");
        l.w("    for (uint64_t i = 0; i < n * IN_WORDS; i++) {");
        l.w("#pragma HLS pipeline II=1");
        l.w("        rec.w[j] = in[i];");
        l.w("        if (++j == IN_WORDS) {");
        l.w("            s.write(rec);");
        l.w("            j = 0;");
        l.w("        }");
        l.w("    }");
    } else {
        l.w("    for (uint64_t i = 0; i < n * IN_WORDS; i++) {");
        l.w("#pragma HLS pipeline II=1");
        l.w("        s.write(in[i]);");
        l.w("    }");
    }
    l.w("}");
    l.w("");
    // ---- core
    let table = ent.table.clone();
    let rdecl = if table.is_some() { ", const reg_t *params".to_string() } else { reg_decl(regs, lanes) };
    if let Some(wn) = &win {
        let (a, b, c, n) = (wn.taps, wn.outputs, wn.offset, wn.inputs);
        l.w("/* core: a shift register over the input words; the window function is the reduction of the program */");
        l.w(format!("static void core(hls::stream<{}> &sin, hls::stream<{}> &sout, hls::stream<unsigned> &sst, uint64_t n, unsigned reset) {{", sin_t, sout_t));
        l.w("    (void)reset;");
        l.w(format!("    in_t win[{}];", a));
        l.w("#pragma HLS array_partition variable=win complete");
        l.w("    KG_TRAP_FLAG = 0;");
        l.w("    unsigned t = 0;");
        l.w("    for (uint64_t i = 0; i < n * IN_WORDS; i++) {");
        l.w("#pragma HLS pipeline II=1");
        l.w("        const in_t x = sin.read();");
        l.w(format!("        for (unsigned k = 0; k + 1 < {}; k++) {{", a));
        l.w("#pragma HLS unroll");
        l.w("            win[k] = win[k + 1];");
        l.w("        }");
        l.w(format!("        win[{}] = x;", a - 1));
        l.w(format!("        if (t >= {} && t < {}) {{", c + a - 1, c + a - 1 + b));
        l.w(format!("            sout.write({}_window(win));", stem));
        l.w("        }");
        l.w(format!("        t = (t + 1 == {}) ? 0 : t + 1;", n));
        l.w("    }");
        l.w("    sst.write(n != 0 ? KG_TRAP_FLAG : 0u);");
        l.w("}");
    } else {
        l.w("/* core: the C of the program, one call of the entry per record and lane */");
        l.w(format!("static void core(hls::stream<{}> &sin, hls::stream<{}> &sout, hls::stream<unsigned> &sst, uint64_t n, unsigned reset{}) {{", sin_t, sout_t, rdecl));
        l.w(format!("    static {}_t inst[LANES];", prefix));
        l.w("    static bool started = false;");
        l.w("    if (!started || reset) {");
        l.w("        for (unsigned l = 0; l < LANES; l++) {");
        l.w("#pragma HLS unroll");
        l.w(format!("            {}_init(&inst[l]);", prefix));
        l.w("        }");
        l.w("        started = true;");
        l.w("    }");
        if table.is_some() {
            l.w("    reg_t pw[REG_WORDS];");
            l.w("    for (unsigned i = 0; i < REG_WORDS; i++) {");
            l.w("#pragma HLS pipeline II=1");
            l.w("        pw[i] = params[i];");
            l.w("    }");
            l.w("    for (unsigned l = 0; l < LANES; l++) {");
            l.w("#pragma HLS unroll");
            for (k, r) in regs.iter().enumerate() {
                let val = if r.elem == "float" { format!("from_bits<{}>(pw[{} * LANES + l])", r.ctype, k) } else { format!("({})pw[{} * LANES + l]", r.ctype, k) };
                l.w(format!("        {}_set_{}(&inst[l], {});", prefix, r.cid, val));
            }
            l.w("    }");
        } else if !regs.is_empty() {
            l.w("    for (unsigned l = 0; l < LANES; l++) {");
            l.w("#pragma HLS unroll");
            for r in regs {
                l.w(format!("        {}_set_{}(&inst[l], {}{});", prefix, r.cid, r.cid, if lanes.is_some() { "[l]" } else { "" }));
            }
            l.w("    }");
        }
        l.w("    KG_TRAP_FLAG = 0;");
        l.w("    for (uint64_t r = 0; r < n; r++) {");
        if pipelined {
            l.w("#pragma HLS pipeline II=1");
        }
        let rec;
        if wide_in {
            l.w("        const in_rec_t rec = sin.read();");
            rec = "rec.w";
        } else {
            l.w("        in_t rec[IN_WORDS];");
            l.w("        for (unsigned j = 0; j < IN_WORDS; j++) {");
            l.w("#pragma HLS pipeline II=1");
            l.w("            rec[j] = sin.read();");
            l.w("        }");
            rec = "rec";
        }
        let mut oref = "o.w";
        if !summary {
            if wide_out {
                l.w("        out_rec_t o;");
                oref = "o.w";
            } else {
                l.w("        out_t o[OUT_WORDS];");
                oref = "o";
            }
        }
        l.w("        for (unsigned l = 0; l < LANES; l++) {");
        l.w("#pragma HLS unroll");
        let cargs = call_args_a(params, mixed, rec, &mut l, "            ", "from_word");
        let args: String = cargs.iter().map(|x| format!(", {}", x)).collect();
        if summary {
            l.w(format!("            {}(&inst[l]{});", stem, args));
        } else if !sig.result.shape.is_empty() {
            l.w(format!("            {}(&inst[l]{}, &{}[l * RES_WORDS]);", stem, args, oref));
        } else {
            l.w(format!("            {}[l * RES_WORDS] = {}(&inst[l]{});", oref, stem, args));
        }
        l.w("        }");
        if summary {
        } else if wide_out {
            l.w("        sout.write(o);");
        } else {
            l.w("        for (unsigned j = 0; j < OUT_WORDS; j++) {");
            l.w("#pragma HLS pipeline II=1");
            l.w("            sout.write(o[j]);");
            l.w("        }");
        }
        l.w("    }");
        if let Some(rd) = reader {
            let rstem = format!("{}_{}", prefix, rd.c_name);
            let oref;
            if wide_out {
                l.w("    out_rec_t o;");
                oref = "o.w";
            } else {
                l.w("    out_t o[OUT_WORDS];");
                oref = "o";
            }
            l.w("    for (unsigned l = 0; l < LANES; l++) {");
            l.w("#pragma HLS unroll");
            if !rd.decl.signature.result.shape.is_empty() {
                l.w(format!("        {}(&inst[l], &{}[l * RES_WORDS]);", rstem, oref));
            } else {
                l.w(format!("        {}[l * RES_WORDS] = {}(&inst[l]);", oref, rstem));
            }
            l.w("    }");
            if wide_out {
                l.w("    sout.write(o);");
            } else {
                l.w("    for (unsigned j = 0; j < OUT_WORDS; j++) {");
                l.w("#pragma HLS pipeline II=1");
                l.w("        sout.write(o[j]);");
                l.w("    }");
            }
        }
        l.w("    sst.write(n != 0 ? KG_TRAP_FLAG : 0u);");
        l.w("}");
    }
    l.w("");
    // ---- store
    l.w("/* store: the output words, one per cycle, then the status word */");
    l.w(format!("static void store(hls::stream<{}> &s, out_t *out, hls::stream<unsigned> &sst, unsigned *status, uint64_t n) {{", sout_t));
    if summary {
        l.w("    (void)n;");
    }
    if wide_out {
        l.w("    out_rec_t rec;");
        l.w("    unsigned k = 0;");
        l.w(format!("    for (uint64_t i = 0; i < {}; i++) {{", tot));
        l.w("#pragma HLS pipeline II=1");
        l.w("        if (k == 0) {");
        l.w("            rec = s.read();");
        l.w("        }");
        l.w("        out[i] = rec.w[k];");
        l.w("        k = (k + 1 == OUT_WORDS) ? 0 : k + 1;");
        l.w("    }");
    } else {
        l.w(format!("    for (uint64_t i = 0; i < {}; i++) {{", tot));
        l.w("#pragma HLS pipeline II=1");
        l.w("        out[i] = s.read();");
        l.w("    }");
    }
    l.w("    const unsigned t = sst.read();");
    l.w("    *status = t ? (t | 1u) : 0u;");
    l.w("}");
    l.w("");
    l.w(format!("extern \"C\" void {}(const in_t *in, out_t *out, unsigned n, unsigned reset{}, unsigned *status) {{", ent.top, rdecl));
    l.w("#pragma HLS interface m_axi port=in offset=slave bundle=gmem0");
    l.w("#pragma HLS interface m_axi port=out offset=slave bundle=gmem1");
    l.w("#pragma HLS interface m_axi port=status offset=slave bundle=gmem2");
    if table.is_some() {
        l.w("#pragma HLS interface m_axi port=params offset=slave bundle=gmem3");
    }
    let mut ports: Vec<String> = ["in", "out", "n", "reset"].iter().map(|s| s.to_string()).collect();
    if table.is_some() {
        ports.push("params".to_string());
    } else {
        ports.extend(regs.iter().map(|r| r.cid.clone()));
    }
    ports.push("status".to_string());
    for port in ports {
        l.w(format!("#pragma HLS interface s_axilite port={} bundle=control", port));
    }
    l.w("#pragma HLS interface s_axilite port=return bundle=control");
    l.w("#pragma HLS dataflow");
    l.w(format!("    hls::stream<{}> sin(\"sin\");", sin_t));
    l.w(format!("    hls::stream<{}> sout(\"sout\");", sout_t));
    l.w("    hls::stream<unsigned> sst(\"sst\");");
    l.w(format!("#pragma HLS stream variable=sin depth={}", STREAM_DEPTH));
    l.w(format!("#pragma HLS stream variable=sout depth={}", STREAM_DEPTH));
    l.w("#pragma HLS stream variable=sst depth=2");
    l.w("    load(in, sin, n);");
    let extra = if win.is_some() {
        String::new()
    } else if table.is_some() {
        ", params".to_string()
    } else {
        regs.iter().map(|r| format!(", {}", r.cid)).collect()
    };
    l.w(format!("    core(sin, sout, sst, n, reset{});", extra));
    l.w("    store(sout, out, sst, status, n);");
    l.w("}");
    l.w("");
    l.text()
}

// ---- lanes

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The public functions of the emitted C (from its header): `<prefix>_init`, `_size`, `_set_*`, one per entry: the matches of
/// `\b(<prefix>_[A-Za-z0-9_]+)\s*\(`, sorted and unique.
fn lane_names(prefix: &str, header: &str) -> Vec<String> {
    let chars: Vec<char> = header.chars().collect();
    let pat: Vec<char> = format!("{}_", prefix).chars().collect();
    let mut found: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut i = 0;
    while i < chars.len() {
        let at = chars[i..].starts_with(&pat) && (i == 0 || !is_word(chars[i - 1]));
        if at {
            let mut j = i + pat.len();
            let s = j;
            while j < chars.len() && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                j += 1;
            }
            if j > s {
                let mut k = j;
                while k < chars.len() && chars[k].is_whitespace() {
                    k += 1;
                }
                if k < chars.len() && chars[k] == '(' {
                    found.insert(chars[i..j].iter().collect());
                    i = k + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    found.into_iter().collect()
}

fn lane_c(prefix: &str, lane: usize, header: &str) -> String {
    let mut l = vec![
        format!("/* lane_{}.c - generated by KlongPy (kgpy fpga source). Do not edit.", lane),
        format!(" * The C of {}.c under other names (a suffix _{}): lane {} of the kernel is one dataflow process, and a", prefix, lane, lane),
        " * process may not share a variable with another, so it has its own functions and its own trap flag. */".to_string(),
        "#undef KG_TRAP_FLAG".to_string(),
        format!("#define KG_TRAP_FLAG {}_{}", TRAP_FLAG, lane),
        format!("#define kg_trap kg_trap_{}", lane),
    ];
    for name in lane_names(prefix, header) {
        l.push(format!("#define {} {}_{}", name, name, lane));
    }
    l.push(format!("#include \"{}.c\"", prefix));
    l.join("\n") + "\n"
}

/// `re.search(r"\bN\s*\(", line)`
fn mentions_call(line: &[char], n: &[char]) -> bool {
    let mut i = 0;
    while i + n.len() <= line.len() {
        if line[i..].starts_with(n) && (i == 0 || !is_word(line[i - 1])) {
            let mut k = i + n.len();
            while k < line.len() && line[k].is_whitespace() {
                k += 1;
            }
            if k < line.len() && line[k] == '(' {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// `re.sub(r"\bN\b", repl, line)`
fn replace_word(line: &[char], n: &[char], repl: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < line.len() {
        if line[i..].starts_with(n) && (i == 0 || !is_word(line[i - 1])) && (i + n.len() == line.len() || !is_word(line[i + n.len()])) {
            out.push_str(repl);
            i += n.len();
        } else {
            out.push(line[i]);
            i += 1;
        }
    }
    out
}

/// `str.splitlines()`
fn split_lines(t: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut it = t.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '\r' => {
                if it.peek() == Some(&'\n') {
                    it.next();
                }
                out.push(std::mem::take(&mut cur));
            }
            '\n' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}' => out.push(std::mem::take(&mut cur)),
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The prototypes of lane l's functions: the lines of the header that declare the program's functions, renamed.
fn lane_decls(prefix: &str, lane: usize, header: &str) -> Vec<String> {
    let names = lane_names(prefix, header);
    let cn: Vec<Vec<char>> = names.iter().map(|n| n.chars().collect()).collect();
    let mut out = Vec::new();
    for line in split_lines(header) {
        let chars: Vec<char> = line.chars().collect();
        let t = line.trim_end();
        if cn.iter().any(|n| mentions_call(&chars, n)) && (t.ends_with("*/") || t.ends_with(';')) {
            let mut text = line.clone();
            for (n, c) in names.iter().zip(&cn) {
                let tc: Vec<char> = text.chars().collect();
                text = replace_word(&tc, c, &format!("{}_{}", n, lane));
            }
            out.push(text);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn top_lanes(program: &Program, prefix: &str, e: &Entry, ent: &Ent, params: &[Param], regs: &[Reg], lanes: usize, reader: &Entry, header: &str) -> String {
    let stem = format!("{}_{}", prefix, e.c_name);
    let rstem = format!("{}_{}", prefix, reader.c_name);
    let inw = ent.in_words;
    let (tin, tout) = (&ent.in_type, &ent.out_type);
    let res_words = ent.out_words / lanes;
    let table = ent.table.clone();
    let in_bits = inw * bytes_of(&params[0].elem, program.int_bits) * 8;
    let wide_in = in_bits <= WIDE_BITS;
    let sin_t = if wide_in { "in_rec_t" } else { "in_t" };
    let mut l = Lines(Vec::new());
    l.w(format!("/* {} - generated by KlongPy (kgpy fpga source) from {}. Do not edit.", ent.f_top, prefix));
    l.w(format!(" * Streamable kernel {} for entry {}: {}, {} lane(s), one process per lane.", ent.top, ent.name, ent.signature, lanes));
    l.w(format!(" * One input record is {} word(s) of {}: the arguments in order, arrays flattened row-major.", inw, tin));
    l.w(format!(" * One output record is {} word(s) of {}: {} lanes of {} word(s), lane first.", ent.out_words, tout, lanes, res_words));
    l.w(format!(" * Each call of the kernel makes n calls of the entry, which return nothing, and then one call of {} ({}) per lane: that is the one output record.", reader.decl.name, reader.decl.signature));
    header_comment_tail(&mut l);
    stream_includes(&mut l);
    trap_guard(&mut l);
    l.w("extern \"C\" {");
    l.w(format!("#include \"{}.h\"", prefix));
    for i in 0..lanes {
        l.w(format!("/* lane {} */", i));
        for d in lane_decls(prefix, i, header) {
            l.w(d);
        }
        l.w(format!("extern unsigned {}_{};", TRAP_FLAG, i));
    }
    l.w("}");
    l.w("");
    l.w(format!("#define IN_WORDS {}", inw));
    l.w(format!("#define RES_WORDS {}", res_words));
    l.w(format!("#define LANES {}", lanes));
    l.w(format!("typedef {} in_t;", tin));
    l.w(format!("typedef {} out_t;", tout));
    if wide_in {
        l.w("typedef struct { in_t w[IN_WORDS]; } in_rec_t;");
    }
    if let Some(t) = &table {
        l.w(format!("typedef uint{}_t reg_t;", 8 * t.word_bytes));
        l.w(format!("#define REG_WORDS {}", t.words));
        l.w(format!("#define REG_COUNT {}", regs.len()));
        l.w("template <class T, class W> static inline T from_bits(W v) { T t; memcpy(&t, &v, sizeof t); return t; }");
    }
    l.w("");
    l.w("/* load: the input words, one per cycle, to every lane */");
    l.w(format!("static void load(const in_t *in, hls::stream<{}> sin[LANES], uint64_t n) {{", sin_t));
    if wide_in {
        l.w("    in_rec_t rec;");
        l.w("    unsigned j = 0;");
        l.w("    for (uint64_t i = 0; i < n * IN_WORDS; i++) {");
        l.w("#pragma HLS pipeline II=1");
        l.w("        rec.w[j] = in[i];");
        l.w("        if (++j == IN_WORDS) {");
        l.w("            for (unsigned l = 0; l < LANES; l++) {");
        l.w("#pragma HLS unroll");
        l.w("                sin[l].write(rec);");
        l.w("            }");
        l.w("            j = 0;");
        l.w("        }");
        l.w("    }");
    } else {
        l.w("    for (uint64_t i = 0; i < n * IN_WORDS; i++) {");
        l.w("#pragma HLS pipeline II=1");
        l.w("        const in_t v = in[i];");
        l.w("        for (unsigned l = 0; l < LANES; l++) {");
        l.w("#pragma HLS unroll");
        l.w("            sin[l].write(v);");
        l.w("        }");
        l.w("    }");
    }
    l.w("}");
    l.w("");
    if table.is_some() {
        l.w("/* regs: the register table, read once, one set of values to each lane */");
        l.w("static void regs(const reg_t *params, hls::stream<reg_t> sreg[LANES]) {");
        l.w("    reg_t pw[REG_WORDS];");
        l.w("    for (unsigned i = 0; i < REG_WORDS; i++) {");
        l.w("#pragma HLS pipeline II=1");
        l.w("        pw[i] = params[i];");
        l.w("    }");
        l.w("    for (unsigned k = 0; k < REG_COUNT; k++) {");
        l.w("        for (unsigned l = 0; l < LANES; l++) {");
        l.w("#pragma HLS unroll");
        l.w("            sreg[l].write(pw[k * LANES + l]);");
        l.w("        }");
        l.w("    }");
        l.w("}");
        l.w("");
    }
    for i in 0..lanes {
        l.w(format!("/* lane {} */", i));
        let regarg = if table.is_some() { "hls::stream<reg_t> &sreg, " } else { "" };
        l.w(format!("static void lane_{}(hls::stream<{}> &sin, {}hls::stream<out_t> &sres, hls::stream<unsigned> &sst, uint64_t n, unsigned reset) {{", i, sin_t, regarg));
        l.w(format!("    static {}_t inst;", prefix));
        l.w("    static bool started = false;");
        l.w("    if (!started || reset) {");
        l.w(format!("        {}_init_{}(&inst);", prefix, i));
        l.w("        started = true;");
        l.w("    }");
        for r in regs {
            if table.is_some() {
                let val = if r.elem == "float" { format!("from_bits<{}>(sreg.read())", r.ctype) } else { format!("({})sreg.read()", r.ctype) };
                l.w(format!("    {}_set_{}_{}(&inst, {});", prefix, r.cid, i, val));
            }
        }
        l.w(format!("    {}_{} = 0;", TRAP_FLAG, i));
        l.w("    for (uint64_t r = 0; r < n; r++) {");
        l.w("#pragma HLS pipeline II=1");
        let rec;
        if wide_in {
            l.w("        const in_rec_t rec = sin.read();");
            rec = "rec.w";
        } else {
            l.w("        in_t rec[IN_WORDS];");
            l.w("        for (unsigned j = 0; j < IN_WORDS; j++) {");
            l.w("#pragma HLS pipeline II=1");
            l.w("            rec[j] = sin.read();");
            l.w("        }");
            rec = "rec";
        }
        let args: String = params.iter().map(|p| if p.shape.is_empty() { format!(", {}[{}]", rec, p.offset) } else { format!(", &{}[{}]", rec, p.offset) }).collect();
        l.w(format!("        {}_{}(&inst{});", stem, i, args));
        l.w("    }");
        l.w("    out_t o[RES_WORDS];");
        if !reader.decl.signature.result.shape.is_empty() {
            l.w(format!("    {}_{}(&inst, o);", rstem, i));
        } else {
            l.w(format!("    o[0] = {}_{}(&inst);", rstem, i));
        }
        l.w("    for (unsigned j = 0; j < RES_WORDS; j++) {");
        l.w("#pragma HLS pipeline II=1");
        l.w("        sres.write(o[j]);");
        l.w("    }");
        l.w(format!("    sst.write(n != 0 ? {}_{} : 0u);", TRAP_FLAG, i));
        l.w("}");
        l.w("");
    }
    l.w("/* store: the records of the lanes in order, then the status word */");
    l.w("static void store(hls::stream<out_t> sres[LANES], hls::stream<unsigned> sst[LANES], out_t *out, unsigned *status) {");
    l.w("    out_t rec[LANES * RES_WORDS];");
    l.w("    for (unsigned l = 0; l < LANES; l++) {");
    l.w("#pragma HLS unroll");
    l.w("        for (unsigned j = 0; j < RES_WORDS; j++) {");
    l.w("#pragma HLS unroll");
    l.w("            rec[l * RES_WORDS + j] = sres[l].read();");
    l.w("        }");
    l.w("    }");
    l.w("    for (unsigned i = 0; i < LANES * RES_WORDS; i++) {");
    l.w("#pragma HLS pipeline II=1");
    l.w("        out[i] = rec[i];");
    l.w("    }");
    l.w("    unsigned t = 0;");
    l.w("    for (unsigned l = 0; l < LANES; l++) {");
    l.w("#pragma HLS unroll");
    l.w("        t |= sst[l].read();");
    l.w("    }");
    l.w("    *status = t ? (t | 1u) : 0u;");
    l.w("}");
    l.w("");
    let rdecl = if table.is_some() { ", const reg_t *params" } else { "" };
    l.w(format!("extern \"C\" void {}(const in_t *in, out_t *out, unsigned n, unsigned reset{}, unsigned *status) {{", ent.top, rdecl));
    l.w("#pragma HLS interface m_axi port=in offset=slave bundle=gmem0");
    l.w("#pragma HLS interface m_axi port=out offset=slave bundle=gmem1");
    l.w("#pragma HLS interface m_axi port=status offset=slave bundle=gmem2");
    if table.is_some() {
        l.w("#pragma HLS interface m_axi port=params offset=slave bundle=gmem3");
    }
    let mut ports: Vec<&str> = vec!["in", "out", "n", "reset"];
    if table.is_some() {
        ports.push("params");
    }
    ports.push("status");
    for port in ports {
        l.w(format!("#pragma HLS interface s_axilite port={} bundle=control", port));
    }
    l.w("#pragma HLS interface s_axilite port=return bundle=control");
    l.w("#pragma HLS dataflow");
    l.w(format!("    hls::stream<{}> sin[LANES];", sin_t));
    if table.is_some() {
        l.w("    hls::stream<reg_t> sreg[LANES];");
    }
    l.w("    hls::stream<out_t> sres[LANES];");
    l.w("    hls::stream<unsigned> sst[LANES];");
    l.w(format!("#pragma HLS stream variable=sin depth={}", STREAM_DEPTH));
    if table.is_some() {
        l.w(format!("#pragma HLS stream variable=sreg depth={}", regs.len() + 1));
    }
    l.w(format!("#pragma HLS stream variable=sres depth={}", res_words + 1));
    l.w("#pragma HLS stream variable=sst depth=2");
    l.w("    load(in, sin, n);");
    if table.is_some() {
        l.w("    regs(params, sreg);");
    }
    for i in 0..lanes {
        l.w(format!("    lane_{}(sin[{}], {}sres[{}], sst[{}], n, reset);", i, i, if table.is_some() { format!("sreg[{}], ", i) } else { String::new() }, i, i));
    }
    l.w("    store(sres, sst, out, status);");
    l.w("}");
    l.w("");
    l.text()
}

// ---- test bench

fn tb(prefix: &str, ent: &Ent, regs: &[Reg], lanes: Option<usize>, table: Option<&Table>) -> String {
    let mut l = Lines(Vec::new());
    l.w(format!("/* {} - generated by KlongPy (kgpy fpga source). Do not edit.", ent.f_tb));
    if !regs.is_empty() {
        l.w(" * usage: tb [DIR [RECORDS_PER_CALL]]  reads DIR/in.bin and DIR/regs.bin (the values of each parameter, one per lane,");
        l.w(" * in manifest order; absent: the defaults), writes DIR/out.bin and DIR/status.bin (one 32-bit word,");
    } else {
        l.w(" * usage: tb [DIR [RECORDS_PER_CALL]]  reads DIR/in.bin, writes DIR/out.bin and DIR/status.bin (one 32-bit word,");
    }
    let summary = ent.summary.is_some();
    if summary {
        l.w(" * the OR of the status of every call; out.bin is the one summary record, as it is after the last call). RECORDS_PER_CALL 0 (the default) is one call for the whole file;");
    } else {
        l.w(" * the OR of the status of every call). RECORDS_PER_CALL 0 (the default) is one call for the whole file;");
    }
    l.w(" * with a number, the first call has reset set and the rest continue the state. */");
    l.w("#include <stdio.h>");
    l.w("#include <stdlib.h>");
    l.w("#include <stdint.h>");
    if table.is_some() {
        l.w("#include <string.h>");
    }
    l.w("#include <vector>");
    l.w("");
    l.w("extern \"C\" {");
    l.w(format!("#include \"{}.h\"", prefix));
    l.w("}");
    l.w(format!("typedef {} in_t;", ent.in_type));
    l.w(format!("typedef {} out_t;", ent.out_type));
    l.w(format!("#define IN_WORDS {}", ent.in_words));
    l.w(format!("#define OUT_WORDS {}", ent.out_words));
    if let Some(t) = table {
        l.w(format!("typedef uint{}_t reg_t;", 8 * t.word_bytes));
        l.w(format!("extern \"C\" void {}(const in_t *in, out_t *out, unsigned n, unsigned reset, const reg_t *params, unsigned *status);", ent.top));
    } else {
        l.w(format!("extern \"C\" void {}(const in_t *in, out_t *out, unsigned n, unsigned reset{}, unsigned *status);", ent.top, reg_decl(regs, lanes)));
    }
    l.w("");
    l.w("int main(int argc, char **argv) {");
    l.w("    const char *dir = argc > 1 ? argv[1] : \".\";");
    l.w("    unsigned per = argc > 2 ? (unsigned)strtoul(argv[2], NULL, 10) : 0;");
    l.w("    char path[4096];");
    let nl = lanes.unwrap_or(1);
    for r in regs {
        l.w(format!("    {} {}[{}];", r.ctype, r.cid, nl));
        l.w(format!("    for (unsigned l = 0; l < {}; l++) {{", nl));
        l.w(format!("        {}[l] = {};", r.cid, r.default.repr()));
        l.w("    }");
    }
    if !regs.is_empty() {
        l.w("    snprintf(path, sizeof path, \"%s/regs.bin\", dir);");
        l.w("    FILE *rf = fopen(path, \"rb\");");
        l.w("    if (rf) {");
        for r in regs {
            l.w(format!("        if (fread({}, sizeof({}), {}, rf) != {}) {{", r.cid, r.ctype, nl, nl));
            l.w("            fprintf(stderr, \"tb: regs.bin is too short\\n\");");
            l.w("            return 2;");
            l.w("        }");
        }
        l.w("        fclose(rf);");
        l.w("    }");
    }
    if let Some(t) = table {
        l.w(format!("    std::vector<reg_t> params({});", t.words));
        let lanes = lanes.unwrap_or(0);
        for (k, r) in regs.iter().enumerate() {
            l.w(format!("    for (unsigned l = 0; l < {}; l++) {{", lanes));
            if r.elem == "float" {
                l.w("        reg_t bits = 0;");
                l.w(format!("        memcpy(&bits, &{}[l], sizeof(float));", r.cid));
                l.w(format!("        params[{} + l] = bits;", k * lanes));
            } else {
                l.w(format!("        params[{} + l] = (reg_t)(int64_t){}[l];", k * lanes, r.cid));
            }
            l.w("    }");
        }
    }
    l.w("    snprintf(path, sizeof path, \"%s/in.bin\", dir);");
    l.w("    FILE *f = fopen(path, \"rb\");");
    l.w("    if (!f) {");
    l.w("        fprintf(stderr, \"tb: cannot read %s\\n\", path);");
    l.w("        return 2;");
    l.w("    }");
    l.w("    std::vector<in_t> in;");
    l.w("    in_t word;");
    l.w("    while (fread(&word, sizeof word, 1, f) == 1) {");
    l.w("        in.push_back(word);");
    l.w("    }");
    l.w("    fclose(f);");
    l.w("    if (in.size() % IN_WORDS != 0) {");
    l.w("        fprintf(stderr, \"tb: in.bin is not a whole number of records (%zu words of %d)\\n\", in.size(), IN_WORDS);");
    l.w("        return 2;");
    l.w("    }");
    l.w("    unsigned n = (unsigned)(in.size() / IN_WORDS);");
    l.w(format!("    std::vector<out_t> out({});", if summary { "(size_t)OUT_WORDS + 1" } else { "(size_t)n * OUT_WORDS + 1" }));
    l.w("    unsigned status = 0, all = 0;");
    l.w("    if (per == 0 || per > n) {");
    l.w("        per = n;");
    l.w("    }");
    l.w("    unsigned done = 0;");
    l.w("    do {");
    l.w("        unsigned m = n - done < per ? n - done : per;");
    let rargs: String = if table.is_some() { ", params.data()".to_string() } else { regs.iter().map(|r| format!(", {}{}", r.cid, if lanes.is_some() { "" } else { "[0]" })).collect() };
    l.w(format!("        {}(in.data() + (size_t)done * IN_WORDS, out.data() + {}, m, done == 0{}, &status);", ent.top, if summary { "0" } else { "(size_t)done * OUT_WORDS" }, rargs));
    l.w("        all |= status;");
    l.w("        done += m;");
    l.w("    } while (done < n);");
    l.w("    snprintf(path, sizeof path, \"%s/out.bin\", dir);");
    l.w("    f = fopen(path, \"wb\");");
    l.w("    if (!f) {");
    l.w("        fprintf(stderr, \"tb: cannot write %s\\n\", path);");
    l.w("        return 2;");
    l.w("    }");
    l.w(format!("    fwrite(out.data(), sizeof(out_t), {}, f);", if summary { "(size_t)OUT_WORDS" } else { "(size_t)n * OUT_WORDS" }));
    l.w("    fclose(f);");
    l.w("    snprintf(path, sizeof path, \"%s/status.bin\", dir);");
    l.w("    f = fopen(path, \"wb\");");
    l.w("    if (!f) {");
    l.w("        fprintf(stderr, \"tb: cannot write %s\\n\", path);");
    l.w("        return 2;");
    l.w("    }");
    l.w("    fwrite(&all, sizeof all, 1, f);");
    l.w("    fclose(f);");
    l.w("    printf(\"records=%u status=%u\\n\", n, all);");
    l.w("    return 0;");
    l.w("}");
    l.w("");
    l.text()
}

// ---- tcl

fn tcl(prefix: &str, ent: &Ent, clock_mhz: u32, flags: &[String], _stage_b: bool) -> String {
    let period = 1000.0 / clock_mhz as f64;
    let cf = flags.join(" ");
    let mut l = Lines(Vec::new());
    l.w(format!("# {} - generated by KlongPy (kgpy fpga source). Do not edit.", ent.f_tcl));
    l.w(format!("# Run:  vitis_hls -f {}", ent.f_tcl));
    l.w("# Environment:  KG_DATA  directory holding in.bin for the C simulation (default: this directory; out.bin and");
    l.w("#                        status.bin are written there)");
    l.w("#               KG_CHUNK records per kernel call in the C simulation (default 0: one call)");
    l.w("#               KG_NO_SYNTH  set to stop after the C simulation");
    l.w("#               KG_EXPORT  set to also run export_design -format xo (a Vitis kernel object)");
    l.w("set here [file dirname [file normalize [info script]]]");
    l.w("set data $here");
    l.w("if {[info exists ::env(KG_DATA)]} { set data $::env(KG_DATA) }");
    l.w("set chunk 0");
    l.w("if {[info exists ::env(KG_CHUNK)]} { set chunk $::env(KG_CHUNK) }");
    l.w(format!("set cflags \"-I$here {}\"", cf));
    l.w(format!("open_project -reset proj_{}", ent.c_name));
    l.w(format!("set_top {}", ent.top));
    l.w(format!("add_files $here/{}.c -cflags $cflags", prefix));
    for lf in &ent.lane_files {
        l.w(format!("add_files $here/{} -cflags $cflags", lf));
    }
    l.w(format!("add_files $here/{} -cflags $cflags", ent.f_top));
    l.w(format!("add_files -tb $here/{} -cflags $cflags", ent.f_tb));
    l.w("open_solution -reset sol1 -flow_target vitis");
    l.w(format!("set_part {{{}}}", PART));
    l.w(format!("create_clock -period {:.3} -name default", period));
    l.w("if {[file exists $data/in.bin]} { csim_design -argv \"$data $chunk\" }");
    l.w("if {[info exists ::env(KG_NO_SYNTH)]} { exit }");
    l.w("csynth_design");
    l.w("if {[info exists ::env(KG_EXPORT)]} { export_design -format xo }");
    l.w("exit");
    l.w("");
    l.text()
}

// ---------------------------------------------------------------------------
// The driver (`emit_hls_program`, `emit_hls_main` of compile.py) and the command line
// ---------------------------------------------------------------------------

pub enum HlsFailure {
    Emit(EmitFailure),
    /// an `HlsError` with the report whose sources it points into
    Hls(HlsError, Report),
}

impl HlsFailure {
    /// The error text and the exit status.
    pub fn render(&self, path: &str) -> (Vec<String>, i32) {
        match self {
            HlsFailure::Emit(f) => f.render(path),
            HlsFailure::Hls(e, r) => (vec![e.render(path, r)], if matches!(e, HlsError::Value(_)) { 1 } else { 1 }),
        }
    }
}

pub struct HlsRun {
    pub out: HlsOutput,
    pub emitted: Emitted,
}

/// Compile a program to C and the Vitis HLS files around it. With `out_dir`, writes them there.
#[allow(clippy::too_many_arguments)]
pub fn emit_hls_program(path: &str, out_dir: Option<&str>, text: Option<&str>, int_bits: u32, prefix: Option<&str>, clock_mhz: u32, wrapped: bool, lanes: Option<usize>, reassociate: bool) -> Result<HlsRun, HlsFailure> {
    emit_hls_program_in(path, out_dir, text, int_bits, prefix, clock_mhz, wrapped, lanes, reassociate, &[])
}

/// `emit_hls_program` with library directories (see `compile::compile_program_in`).
#[allow(clippy::too_many_arguments)]
pub fn emit_hls_program_in(path: &str, out_dir: Option<&str>, text: Option<&str>, int_bits: u32, prefix: Option<&str>, clock_mhz: u32, wrapped: bool, lanes: Option<usize>, reassociate: bool, lib_dirs: &[String]) -> Result<HlsRun, HlsFailure> {
    let prefix = prefix.filter(|p| !p.is_empty());
    let compiled = crate::compile::compile_program_in(path, text, int_bits, prefix, std::collections::HashMap::new(), lib_dirs).map_err(|f| HlsFailure::Emit(EmitFailure::Compile(f)))?;
    let (options, mode): (Option<HlsOptions>, Option<String>) = if wrapped {
        (None, None)
    } else {
        let work = compiled.program.entries.iter().map(|e| hls_work(&compiled.program, e)).max().unwrap_or(0);
        let mode = if work <= HLS_WORK_LIMIT { "pipeline" } else { "block" };
        (Some(HlsOptions { reassociate, mode: mode.to_string(), ..HlsOptions::default() }), Some(mode.to_string()))
    };
    let emitted = match crate::emit_c::emit(&compiled.program, &compiled.report, prefix, 32, false, options) {
        Ok(e) => e,
        Err(e) => return Err(HlsFailure::Emit(EmitFailure::Emit(e, compiled.report))),
    };
    let source = path.rsplit('/').next().unwrap_or(path).to_string();
    let mut out = match emit_hls(&compiled.program, &emitted, clock_mhz, Some(&source), wrapped, lanes, reassociate, mode.as_deref()) {
        Ok(o) => o,
        Err(e) => return Err(HlsFailure::Hls(e, compiled.report)),
    };
    out.files.insert(format!("{}.c", emitted.prefix), emitted.c.clone());
    out.files.insert(format!("{}.h", emitted.prefix), emitted.h.clone());
    out.files.insert("klong_rt.h".to_string(), RUNTIME_HEADER.to_string());
    if let Some(d) = out_dir {
        let list: Vec<(String, String)> = out.files.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        write_files(d, &list).map_err(|m| HlsFailure::Emit(EmitFailure::Io(m)))?;
    }
    Ok(HlsRun { out, emitted })
}

/// The body of `kgpy fpga source FILE... -o DIR`: 0, 1 (a program outside the profile or that cannot be a kernel) or 2 (a file cannot
/// be read).
#[allow(clippy::too_many_arguments)]
pub fn emit_hls_main(files: &[String], out_dir: Option<&str>, int_bits: u32, prefix: Option<&str>, clock_mhz: u32, wrapped: bool, lanes: Option<usize>, reassociate: bool, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let mut status = 0;
    let prefix = prefix.filter(|p| !p.is_empty());
    if prefix.is_some() && files.len() > 1 {
        let _ = writeln!(err, "--prefix names one program: give one FILE");
        return 2;
    }
    let where_ = out_dir.filter(|d| !d.is_empty()).unwrap_or(".");
    for path in files {
        let res = match emit_hls_program(path, Some(where_), None, int_bits, prefix, clock_mhz, wrapped, lanes, reassociate) {
            Ok(r) => r,
            Err(f) => {
                let (lines, st) = f.render(path);
                for l in lines {
                    let _ = writeln!(err, "{}", l);
                }
                status = if st == 2 { status.max(2) } else { 1 };
                continue;
            }
        };
        let names: Vec<String> = res.out.files.keys().map(|n| klong_lint::recover::py_join(where_, n)).collect();
        let _ = writeln!(out, "{}: wrote {}", path, names.join(", "));
        for ent in &res.out.entries {
            let _ = writeln!(out, "  kernel {}: entry {} ({}); record in {} bytes, out {} bytes", ent.top, ent.name, ent.signature, ent.in_bytes, ent.out_bytes);
        }
    }
    status
}

const FPGA_SOURCE_HELP: &str = "\
(experimental) Write what Vitis HLS needs to make a kernel of each .entry of
FILEs of the static profile (C, <name>_<entry>_top.cpp, test bench, tcl,
manifest.json) in the directory given by -o. There is no --float64: the card
has float32 only.

positional arguments:
  FILE                  the programs to turn into kernels

options:
  -h, --help            show this help message and exit
  -o DIR, --output DIR  the directory to write to (default: .)
  --int64               kg_int is 64 bits wide (the default is 32)
  --prefix NAME         the C prefix of the public names (default: the file
                        name)
  --clock-mhz N         the kernel clock in MHz (default: 300)
  --lanes N             N cores on one input stream; every .param becomes an
                        array of N values and the result gains a leading
                        dimension N
  --wrapped             the stage A kernel (the C of compile c called once per
                        record) instead of the streamable one
  --reassociate         sum float reductions as a balanced tree (changes the
                        rounding)
  --synth               run C synthesis on $KLONG_FPGA_BUILD (user@host;
                        unset: here) and print the estimates
";

pub fn fpga_source_spec(prog: &str) -> crate::argp::Spec {
    use crate::argp::{Opt, Spec};
    let mut s = Spec {
        prog: prog.to_string(),
        opts: vec![
            Opt { short: Some("-o"), long: "--output", metavar: Some("DIR"), int: false },
            Opt { short: None, long: "--int64", metavar: None, int: false },
            Opt { short: None, long: "--prefix", metavar: Some("NAME"), int: false },
            Opt { short: None, long: "--clock-mhz", metavar: Some("N"), int: true },
            Opt { short: None, long: "--lanes", metavar: Some("N"), int: true },
            Opt { short: None, long: "--wrapped", metavar: None, int: false },
            Opt { short: None, long: "--reassociate", metavar: None, int: false },
            Opt { short: None, long: "--synth", metavar: None, int: false },
        ],
        positional: "FILE [FILE ...]",
        help: String::new(),
    };
    s.help = format!("{}\n{}", crate::argp::usage(&s), FPGA_SOURCE_HELP);
    s
}

/// `klong fpga source ...`: `args` follow the words `fpga source`.
pub fn fpga_source_cli(prog: &str, args: &[String], out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let spec = fpga_source_spec(prog);
    let p = match crate::argp::parse(&spec, args, out, err) {
        crate::argp::Outcome::Run(p) => p,
        crate::argp::Outcome::Exit(c) => return c,
    };
    let num = |k: &str| p.get(k).map(|s| s.trim().replace('_', "").parse::<i64>().unwrap_or(0));
    let clock = num("--clock-mhz").unwrap_or(300);
    let lanes = num("--lanes");
    if clock < 1 {
        return crate::argp::error(&spec, err, "--clock-mhz must be a positive number of MHz");
    }
    if matches!(lanes, Some(l) if l < 1) {
        return crate::argp::error(&spec, err, "--lanes must be 1 or more");
    }
    if p.flag("--wrapped") && (lanes.is_some() || p.flag("--reassociate")) {
        return crate::argp::error(&spec, err, "--lanes and --reassociate belong to the streamable kernel, not to --wrapped");
    }
    if p.flag("--synth") {
        let _ = writeln!(err, "{}: --synth runs the vendor tools through Python and is not part of this tool; use `kgpy fpga source --synth`", prog);
        return 2;
    }
    emit_hls_main(&p.files, p.get("--output").map(|s| s.as_str()), if p.flag("--int64") { 64 } else { 32 }, p.get("--prefix").map(|s| s.as_str()), clock as u32, p.flag("--wrapped"), lanes.map(|l| l as usize), p.flag("--reassociate"), out, err)
}
