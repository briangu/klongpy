//! The C emitter (`klongpy/static/emit_c.py`, docs/static-profile.md).
//!
//! Every expression is compiled to a *value*: a scalar (a C expression) or an array view (a function from index expressions to the C
//! expression of one element). Atomic operators, conversions and every index-remapping operator compose views and emit nothing, so a
//! tree of them becomes one loop nest when it is finally stored: that is fusion. Storage is only touched by assignments to array
//! variables, results of functions, and the nodes that need a place (an evaluated list, amend, an array conditional, a call that returns
//! an array, each, scan, an array accumulator): they get a *slot*, a field of the instance struct.
//!
//! Evaluation order is Klong's. A value computed before a later operand that has a side effect is first *pinned* (copied to a C local or
//! a slot) if it reads anything that can change. Nothing here names a board, a chip or a vendor.
use crate::fixed as fx;
use crate::ir::*;
use crate::sig::{c_identifier, is_fixed, Ty, C_KEYWORDS};
use crate::util::py_float;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

/// What the hardware flavour of the C changes (`kgpy fpga source`, stage B); the plain C of `compile c` is unchanged.
#[derive(Clone, Debug)]
pub struct HlsOptions {
    pub reassociate: bool,
    /// "pipeline": the whole call of an entry is one pipelined region; "block": each innermost loop is pipelined instead
    pub mode: String,
    pub unroll_max: usize,
    pub tree_max: usize,
}

impl Default for HlsOptions {
    fn default() -> HlsOptions {
        HlsOptions { reassociate: false, mode: "pipeline".to_string(), unroll_max: 16, tree_max: 512 }
    }
}

/// The iterations of array loops above which a call is not unrolled into one pipelined region.
pub const HLS_WORK_LIMIT: usize = 4096;

/// A rough count of the element operations in one call of an entry (what full unrolling would replicate).
pub fn hls_work(_program: &Program, entry: &Entry) -> usize {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut total = 0usize;
    let mut todo: Vec<F> = vec![entry.f.clone()];
    while let Some(f) = todo.pop() {
        if !seen.insert(fn_key(&f)) {
            continue;
        }
        for n in f.body.walked() {
            if !n.ty.shape.is_empty() && !matches!(n.k, K::Const(_) | K::GlobalRef { .. } | K::LocalRef { .. } | K::ParamRef { .. } | K::Seq { .. } | K::Assign { .. }) {
                total += size_of(&n.ty.shape);
            }
            match &n.k {
                K::Reduce { a, .. } | K::Scan { a, .. } => total += size_of(&a.ty.shape),
                _ => {}
            }
            if matches!(n.k, K::Map { .. } | K::Map2 { .. } | K::EachPair { .. }) || matches!(&n.k, K::Loop { kind, .. } if kind == "iterate") {
                total += size_of(&n.ty.shape);
            }
            for g in n.fnspecs() {
                todo.push(g.clone());
            }
        }
    }
    total
}

/// A sliding window found by `match_window`.
#[derive(Clone)]
pub struct Window {
    pub taps: usize,
    pub outputs: usize,
    pub offset: usize,
    pub inputs: usize,
    pub reduce: N,
    pub gather: N,
    pub op: String,
}

/// Is the entry a sliding window over its one array argument? (`run::{ f/E }` with E an elementwise expression of shape [A,B] whose
/// only array operand that reads x is `x@idx`, idx a constant table with idx[a][b] = c + a + b, f one of + | &.)
pub fn match_window(program: &Program, entry: &Entry) -> Option<Window> {
    let sig = &entry.decl.signature;
    let f = &entry.f;
    if sig.params.len() != 1 || sig.params[0].shape.len() != 1 || sig.result.shape.len() != 1 {
        return None;
    }
    let stmts = f.stmts();
    if stmts.len() != 1 {
        return None;
    }
    if f.body.walked().iter().any(|n| matches!(n.k, K::ExternCall { .. } | K::Call { .. } | K::Assign { .. } | K::Cond { .. } | K::Map { .. } | K::Map2 { .. } | K::Loop { .. } | K::Scan { .. } | K::EachPair { .. } | K::Amend { .. })) {
        return None;
    }
    let top = &stmts[0];
    let K::Reduce { f: rf, a: e, seed, caps } = &top.k else { return None };
    if seed.is_some() || !caps.is_empty() || rf.kind != "op" {
        return None;
    }
    let K::Binary { op, .. } = &rf.stmts()[0].k else { return None };
    if !matches!(op.as_str(), "add" | "max" | "min") {
        return None;
    }
    if e.ty.shape.len() != 2 || top.ty.shape != vec![e.ty.shape[1]] {
        return None;
    }
    let (a_, b_) = (e.ty.shape[0], e.ty.shape[1]);
    let mut gathers: Vec<&Node> = Vec::new();
    for n in e.walked() {
        if matches!(n.k, K::Gather { .. }) {
            gathers.push(n);
        }
    }
    if gathers.len() != 1 {
        return None;
    }
    let g = gathers[0];
    let K::Gather { a: ga, idx: gidx, .. } = &g.k else { return None };
    if !matches!(&ga.k, K::ParamRef { index: 0, .. }) {
        return None;
    }
    let mut idx = gidx.cval();
    if idx.is_none() {
        if let K::GlobalRef { name, state: false, .. } = &gidx.k {
            idx = program.globals.iter().find(|q| &q.name == name).map(|q| q.value.clone());
        }
    }
    let idx = idx?;
    if gidx.ty.shape != vec![a_, b_] {
        return None;
    }
    let iv = idx.as_i64();
    if iv.len() != a_ * b_ {
        return None;
    }
    let c = iv[0];
    for a in 0..a_ {
        for b in 0..b_ {
            if iv[a * b_ + b] != c + a as i64 + b as i64 {
                return None;
            }
        }
    }
    let n_in = sig.params[0].shape[0] as i64;
    if c < 0 || c + a_ as i64 + b_ as i64 - 2 >= n_in {
        return None;
    }
    for n in e.walked() {
        if matches!(n.k, K::ParamRef { .. }) && !std::ptr::eq(n, &**ga) {
            return None;
        }
        if matches!(&n.k, K::GlobalRef { state: true, .. }) {
            return None;
        }
        if matches!(n.k, K::Index { .. } | K::Take { .. } | K::Drop { .. } | K::Rotate { .. } | K::Reverse { .. } | K::Transpose { .. } | K::Reshape { .. } | K::Join { .. } | K::Iota { .. } | K::MakeList { .. }) {
            return None;
        }
    }
    // the gather node as an Rc: find it among the walked children of the reduce operand
    let gather = find_rc(e, g)?;
    Some(Window { taps: a_, outputs: b_, offset: c as usize, inputs: n_in as usize, reduce: top.clone(), gather, op: op.clone() })
}

fn find_rc(root: &N, target: &Node) -> Option<N> {
    if std::ptr::eq(&**root, target) {
        return Some(root.clone());
    }
    for c in root.children() {
        if let Some(r) = find_rc(c, target) {
            return Some(r);
        }
    }
    None
}

pub fn size_of(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The C type of an element: kg_float, kg_int, or the integer that holds a fixed-point number.
pub fn ctype_of(elem: &str) -> &'static str {
    if elem == "float" {
        "kg_float"
    } else if is_fixed(elem) {
        if fx::spec_of(elem).w <= 32 {
            "int32_t"
        } else {
            "int64_t"
        }
    } else {
        "kg_int"
    }
}

fn fx_conv(src: &str, dst: &str, x: &str) -> String {
    if is_fixed(dst) {
        let d = fx::spec_of(dst);
        let ct = ctype_of(dst);
        if is_fixed(src) {
            let s = fx::spec_of(src);
            return format!("(({})kg_fxshift({}, {}, {}))", ct, x, d.f() as i64 - s.f() as i64, d.w);
        }
        if src == "float" {
            return format!("(({})kg_fxfromf((double)({}), {}, {}))", ct, x, py_float(2f64.powi(d.f() as i32)), d.w);
        }
        return format!("(({})kg_fxfromi((int64_t)({}), {}, {}))", ct, x, d.f(), d.w);
    }
    let s = fx::spec_of(src);
    format!("((kg_float)({}) * (kg_float){})", x, py_float(2f64.powi(-(s.f() as i32))))
}

fn fx_expand(op: &str, elem: &str, a: &str, b: &str, line: usize) -> Option<String> {
    let s = fx::spec_of(elem);
    let ct = ctype_of(elem);
    let mul = if s.w <= 32 { "kg_fxmul" } else { "kg_fxmul128" };
    Some(match op {
        "add" => format!("(({})kg_fxadd({}, {}, {}))", ct, a, b, s.w),
        "sub" => format!("(({})kg_fxsub({}, {}, {}))", ct, a, b, s.w),
        "mul" => format!("(({}){}({}, {}, {}, {}))", ct, mul, a, b, s.f(), s.w),
        "div" => format!("(({})kg_fxdiv({}, {}, {}, {}, {}))", ct, a, b, s.f(), s.w, line),
        "max" => format!("(({}) > ({}) ? ({}) : ({}))", a, b, a, b),
        "min" => format!("(({}) < ({}) ? ({}) : ({}))", a, b, a, b),
        "lt" => format!("((kg_int)(({}) < ({})))", a, b),
        "gt" => format!("((kg_int)(({}) > ({})))", a, b),
        "eq" => format!("((kg_int)(({}) == ({})))", a, b),
        "neg" => format!("(({})kg_fxneg({}, {}))", ct, a, s.w),
        "floor" => format!("kg_fxfloor({}, {})", a, s.f()),
        "not" => format!("((kg_int)(({}) == 0))", a),
        _ => return None,
    })
}

/// The C expansion of an operator row: (int form, float form).
fn c_expansion(op: &str) -> Option<(Option<&'static str>, Option<&'static str>)> {
    Some(match op {
        "add" => (Some("kg_iadd({a}, {b})"), Some("({a} + {b})")),
        "sub" => (Some("kg_isub({a}, {b})"), Some("({a} - {b})")),
        "mul" => (Some("kg_imul({a}, {b})"), Some("({a} * {b})")),
        "div" => (None, Some("kg_fdiv({a}, {b}, {line})")),
        "idiv" => (Some("kg_idiv({a}, {b}, {line})"), None),
        "rem" => (Some("kg_imod({a}, {b}, {line})"), Some("kg_fmod({a}, {b}, {line})")),
        "pow" => (Some("kg_ipow({a}, {b})"), Some("kg_pow({a}, {b})")),
        "max" => (Some("kg_imax({a}, {b})"), Some("kg_fmax({a}, {b})")),
        "min" => (Some("kg_imin({a}, {b})"), Some("kg_fmin({a}, {b})")),
        "lt" => (Some("kg_ilt({a}, {b})"), Some("kg_flt({a}, {b})")),
        "gt" => (Some("kg_igt({a}, {b})"), Some("kg_fgt({a}, {b})")),
        "eq" => (Some("kg_ieq({a}, {b})"), Some("kg_feq({a}, {b})")),
        "match" => (Some("kg_match((kg_float){a}, (kg_float){b})"), Some("kg_match({a}, {b})")),
        "neg" => (Some("kg_ineg({a})"), Some("(-{a})")),
        "floor" => (None, Some("kg_floor({a})")),
        "recip" => (None, Some("kg_fdiv((kg_float)1, {a}, {line})")),
        "not" => (Some("(kg_int)({a} == 0)"), Some("(kg_int)({a} == 0)")),
        _ => return None,
    })
}

/// The operator ids the C column handles (a test compares it with the operator table).
pub const HANDLED: &[&str] = &[
    "add", "sub", "mul", "div", "idiv", "rem", "pow", "max", "min", "lt", "gt", "eq", "match", "index", "take", "drop", "join", "rotate", "reshape", "amend", "define", "neg", "floor", "recip", "not", "size", "shape", "first",
    "reverse", "transpose", "enumerate", "list", "each", "each2", "over", "over-seed", "scan", "scan-seed", "each-pair", "iterate", "while",
];

/// One of the C expansions, filled in. `None` when the row has no form for this element type.
fn expansion(op: &str, elem: &str, a: &str, b: &str, line: usize) -> Option<String> {
    let (i, f) = c_expansion(op)?;
    let form = if elem == "float" { f } else { i }?;
    Some(form.replace("{a}", a).replace("{b}", b).replace("{line}", &line.to_string()))
}

// ---------------------------------------------------------------------------
// Index arithmetic on C expressions, folding literals
// ---------------------------------------------------------------------------

fn lit(e: &str) -> Option<i64> {
    e.trim().parse::<i64>().ok()
}

fn add(a: &str, b: &str) -> String {
    let (la, lb) = (lit(a), lit(b));
    if let (Some(x), Some(y)) = (la, lb) {
        return (x + y).to_string();
    }
    if la == Some(0) {
        return b.to_string();
    }
    if lb == Some(0) {
        return a.to_string();
    }
    if let Some(y) = lb {
        if y < 0 {
            return format!("({} - {})", a, -y);
        }
    }
    format!("({} + {})", a, b)
}

fn sub(a: &str, b: &str) -> String {
    let (la, lb) = (lit(a), lit(b));
    if let (Some(x), Some(y)) = (la, lb) {
        return (x - y).to_string();
    }
    if lb == Some(0) {
        return a.to_string();
    }
    format!("({} - {})", a, b)
}

fn mul(a: &str, b: &str) -> String {
    let (la, lb) = (lit(a), lit(b));
    if let (Some(x), Some(y)) = (la, lb) {
        return (x * y).to_string();
    }
    if la == Some(0) || lb == Some(0) {
        return "0".to_string();
    }
    if la == Some(1) {
        return b.to_string();
    }
    if lb == Some(1) {
        return a.to_string();
    }
    format!("({} * {})", a, b)
}

fn flat(idx: &[String], shape: &[usize]) -> String {
    let mut e = idx[0].clone();
    for k in 1..shape.len() {
        e = add(&mul(&e, &shape[k].to_string()), &idx[k]);
    }
    e
}

fn is_ident(c: &str) -> bool {
    let b = c.as_bytes();
    if !b.is_empty() && (b[0].is_ascii_alphabetic() || b[0] == b'_') && b.iter().all(|x| x.is_ascii_alphanumeric() || *x == b'_') {
        return true;
    }
    // \(?-?\d+\)?\Z
    let mut s = c;
    if let Some(r) = s.strip_prefix('(') {
        s = r;
    }
    if let Some(r) = s.strip_prefix('-') {
        s = r;
    }
    let digits = s.bytes().take_while(|x| x.is_ascii_digit()).count();
    if digits == 0 {
        return false;
    }
    let rest = &s[digits..];
    rest.is_empty() || rest == ")"
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

type At = Rc<dyn Fn(&[String]) -> String>;

#[derive(Clone)]
struct S {
    elem: String,
    c: String,
    reads: BTreeSet<String>,
}

impl S {
    fn new(elem: &str, c: String, reads: BTreeSet<String>) -> S {
        S { elem: elem.to_string(), c, reads }
    }
    fn atomic(&self) -> bool {
        is_ident(&self.c)
    }
}

#[derive(Clone)]
struct A {
    ty: Ty,
    at: At,
    /// {storage key: aligned}: aligned means element i reads only element i of that storage
    reads: BTreeMap<String, bool>,
    ptr: Option<String>,
    key: Option<String>,
}

#[derive(Clone)]
enum Val {
    S(S),
    A(A),
}

impl Val {
    fn elem(&self) -> &str {
        match self {
            Val::S(s) => &s.elem,
            Val::A(a) => &a.ty.elem,
        }
    }
    fn shape(&self) -> &[usize] {
        match self {
            Val::S(_) => &[],
            Val::A(a) => &a.ty.shape,
        }
    }
    fn c(&self) -> String {
        match self {
            Val::S(s) => s.c.clone(),
            Val::A(_) => String::from("0"),
        }
    }
    fn has_read(&self, key: &str) -> bool {
        match self {
            Val::S(s) => s.reads.contains(key),
            Val::A(a) => a.reads.contains_key(key),
        }
    }
    fn read_keys(&self) -> BTreeSet<String> {
        match self {
            Val::S(s) => s.reads.clone(),
            Val::A(a) => a.reads.keys().cloned().collect(),
        }
    }
    fn has_reads(&self) -> bool {
        match self {
            Val::S(s) => !s.reads.is_empty(),
            Val::A(a) => !a.reads.is_empty(),
        }
    }
    fn at(&self, idx: &[String]) -> String {
        match self {
            Val::A(a) => (a.at)(idx),
            Val::S(s) => s.c.clone(),
        }
    }
}

fn keyset(keys: &[&str]) -> BTreeSet<String> {
    keys.iter().map(|k| k.to_string()).collect()
}

/// The view of an array stored at ptr.
fn stored(ty: &Ty, ptr: &str, key: Option<&str>, mutable: bool) -> A {
    let shape = ty.shape.clone();
    let p = ptr.to_string();
    let mut reads = BTreeMap::new();
    if mutable {
        if let Some(k) = key {
            reads.insert(k.to_string(), true);
        }
    }
    A { ty: ty.clone(), at: Rc::new(move |idx: &[String]| format!("{}[{}]", p, flat(idx, &shape))), reads, ptr: Some(ptr.to_string()), key: key.map(String::from) }
}

/// Where an array result may be written: fresh means nothing else can read or alias it while it is filled.
#[derive(Clone)]
struct Dest {
    ptr: String,
    key: Option<String>,
    ty: Ty,
    fresh: bool,
}

fn unaligned(reads: &BTreeMap<String, bool>) -> BTreeMap<String, bool> {
    reads.keys().map(|k| (k.clone(), false)).collect()
}

fn merge(parts: &[&BTreeMap<String, bool>]) -> BTreeMap<String, bool> {
    let mut out: BTreeMap<String, bool> = BTreeMap::new();
    for reads in parts {
        for (k, v) in reads.iter() {
            let cur = out.get(k).copied().unwrap_or(true);
            out.insert(k.clone(), cur && *v);
        }
    }
    out
}

fn reads_of(v: &Val) -> BTreeMap<String, bool> {
    match v {
        Val::A(a) => a.reads.clone(),
        Val::S(_) => BTreeMap::new(),
    }
}

/// The size report of an emitted program.
#[derive(Clone, Debug)]
pub struct SizeReport {
    pub int_bits: u32,
    pub float_bits: u32,
    pub const_bytes: usize,
    pub instance_bytes: usize,
    pub state_bytes: usize,
    pub slot_bytes: usize,
    pub call_depth: usize,
    pub call_chain: Vec<String>,
    pub stack_bytes: usize,
    pub c_lines: usize,
    pub need_math: bool,
    pub entries: Vec<String>,
    pub externs: Vec<String>,
}

/// The result of `emit`: the texts of `<prefix>.c` and `<prefix>.h` and the size report.
pub struct Emitted {
    pub prefix: String,
    pub c: String,
    pub h: String,
    pub report: SizeReport,
    /// the sliding-window functions of the hardware flavour: (entry index, window, prototype, body lines)
    pub windows: Vec<(usize, Window, String, Vec<String>)>,
}

impl Emitted {
    pub fn report_text(&self) -> String {
        let r = &self.report;
        let lines = [
            format!("size report for {} (kg_int {} bits, kg_float {} bits)", self.prefix, r.int_bits, r.float_bits),
            format!("  constants          {} bytes (static const data; flash on a microcontroller)", r.const_bytes),
            format!("  sizeof({}_t)  {} bytes ({} state, {} array locals and temporaries)", self.prefix, r.instance_bytes, r.state_bytes, r.slot_bytes),
            format!("  deepest call chain {} C function(s): {}", r.call_depth, r.call_chain.join(" > ")),
            format!("  stack estimate     {} bytes (scalars, loop counters and call frames along that chain; externs not counted)", r.stack_bytes),
            format!("  C source           {} lines", r.c_lines),
        ];
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// One C function being written
// ---------------------------------------------------------------------------

struct Code {
    lines: Vec<String>,
    depth: usize,
    n: usize,
    scalars: usize,
}

impl Code {
    fn emit(&mut self, text: &str) {
        let l = format!("{}{}", "    ".repeat(self.depth), text);
        self.lines.push(l);
    }
    fn name(&mut self, stem: &str) -> String {
        self.n += 1;
        format!("{}{}", stem, self.n)
    }
}

/// `(\s*)(?:const )?kg_\w+ (t\d+)( = .*)?;\Z`: the name of the temporary a line declares.
fn temp_decl(line: &str) -> Option<(String, String)> {
    let rest = line.trim_start();
    let indent = &line[..line.len() - rest.len()];
    let rest = rest.strip_prefix("const ").unwrap_or(rest);
    let rest = rest.strip_prefix("kg_")?;
    let w = rest.bytes().take_while(|b| b.is_ascii_alphanumeric() || *b == b'_').count();
    if w == 0 {
        return None;
    }
    let rest = rest[w..].strip_prefix(' ')?;
    let rest = rest.strip_prefix('t')?;
    let d = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
    if d == 0 {
        return None;
    }
    let name = format!("t{}", &rest[..d]);
    let tail = &rest[d..];
    if tail == ";" || (tail.starts_with(" = ") && tail.ends_with(';')) {
        Some((indent.to_string(), name))
    } else {
        None
    }
}

fn count_word(text: &str, w: &str) -> usize {
    let b = text.as_bytes();
    let wb = w.as_bytes();
    let is_w = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80;
    let mut n = 0;
    let mut i = 0;
    while i + wb.len() <= b.len() {
        if &b[i..i + wb.len()] == wb && (i == 0 || !is_w(b[i - 1])) && (i + wb.len() == b.len() || !is_w(b[i + wb.len()])) {
            n += 1;
            i += wb.len();
        } else {
            i += 1;
        }
    }
    n
}

/// A temporary that is computed for its effect only gets a `(void)` use, so the function compiles without warnings.
fn mark_unused(lines: Vec<String>) -> Vec<String> {
    let text: Vec<&str> = lines.iter().map(|l| l.as_str()).filter(|l| !l.trim_start().starts_with("/*") && !l.trim_start().starts_with("#line")).collect();
    let text = text.join("\n");
    let mut out = Vec::new();
    for line in &lines {
        out.push(line.clone());
        if let Some((indent, name)) = temp_decl(line) {
            if count_word(&text, &name) == 1 {
                out.push(format!("{}(void){};", indent, name));
            }
        }
    }
    out
}

/// A source text with its line starts (byte offsets).
pub struct SrcText {
    pub name: String,
    pub text: String,
    starts: Vec<usize>,
}

impl SrcText {
    pub fn new(name: &str, text: &str) -> SrcText {
        let mut starts = vec![0];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        SrcText { name: name.to_string(), text: text.to_string(), starts }
    }
    pub fn line(&self, off: usize) -> usize {
        let off = off.min(self.text.len());
        match self.starts.binary_search(&off) {
            Ok(k) => k + 1,
            Err(k) => k,
        }
    }
}

// ---------------------------------------------------------------------------
// The state shared by the functions of a program
// ---------------------------------------------------------------------------

struct Slot {
    field: String,
    elem: String,
    size: usize,
    note: String,
}

struct Em {
    hls: Option<HlsOptions>,
    window_leaves: HashSet<usize>,
    prefix: String,
    int_bits: u32,
    float_bits: u32,
    line_directives: bool,
    source_name: String,
    sources: Vec<Rc<SrcText>>,
    tables: Vec<(String, Ty, V)>,
    table_keys: HashMap<(String, Vec<usize>, Vec<u64>), String>,
    slots: Vec<Slot>,
    need_math: Rc<Cell<bool>>,
    writes_c: HashMap<usize, HashSet<String>>,
    reads_c: HashMap<usize, HashSet<String>>,
    uses_c: HashMap<usize, HashMap<usize, usize>>,
    frames: HashMap<usize, (usize, Vec<F>)>,
    err: Option<String>,
}

impl Em {
    fn isize(&self, elem: &str) -> usize {
        if is_fixed(elem) {
            return if fx::spec_of(elem).w <= 32 { 4 } else { 8 };
        }
        (if elem == "float" { self.float_bits } else { self.int_bits } / 8) as usize
    }

    fn literal(&self, elem: &str, v: &V) -> String {
        if elem == "float" {
            let f = v.as_f64()[0];
            if f.is_nan() {
                return "KG_NAN".to_string();
            }
            if f.is_infinite() {
                return if f > 0.0 { "KG_INF" } else { "(-KG_INF)" }.to_string();
            }
            return format!("((kg_float){})", py_float(f));
        }
        let iv = v.as_i64()[0];
        if is_fixed(elem) {
            let ct = ctype_of(elem);
            if ct == "int64_t" {
                return if iv > i64::MIN { format!("INT64_C({})", iv) } else { "(-INT64_C(9223372036854775807) - 1)".to_string() };
            }
            return if iv >= 0 {
                format!("(({}){})", ct, iv)
            } else if iv > -(1i64 << 31) {
                format!("(({})({}))", ct, iv)
            } else {
                "(-2147483647 - 1)".to_string()
            };
        }
        let lo = -(1i128 << (self.int_bits - 1));
        if iv as i128 == lo {
            return format!("({} - 1)", iv as i128 + 1);
        }
        if iv >= 0 {
            iv.to_string()
        } else {
            format!("({})", iv)
        }
    }

    fn table(&mut self, ty: &Ty, value: &V, name: Option<&str>) -> String {
        let bits: Vec<u64> = if ty.elem == "float" { value.as_f64().iter().map(|x| x.to_bits()).collect() } else { value.as_i64().iter().map(|x| *x as u64).collect() };
        let key = (ty.elem.clone(), ty.shape.clone(), bits);
        if name.is_none() {
            if let Some(n) = self.table_keys.get(&key) {
                return n.clone();
            }
        }
        let nm = match name {
            Some(n) => n.to_string(),
            None => {
                let n = format!("kg_k{}", self.tables.len());
                self.table_keys.insert(key, n.clone());
                n
            }
        };
        let v = if ty.elem == "float" { V::AF(value.as_f64()) } else { V::AI(value.as_i64()) };
        self.tables.push((nm.clone(), ty.clone(), v));
        nm
    }

    fn table_text(&self, name: &str, ty: &Ty, arr: &V) -> Vec<String> {
        let ctype = ctype_of(&ty.elem);
        let n = arr.len();
        let items: Vec<String> = if ty.elem == "float" {
            arr.as_f64()
                .iter()
                .map(|x| {
                    let l = self.literal("float", &V::F(*x));
                    if x.is_finite() {
                        l.replace("((kg_float)", "").trim_end_matches(')').to_string()
                    } else {
                        l
                    }
                })
                .collect()
        } else {
            arr.as_i64().iter().map(|x| self.literal(if is_fixed(&ty.elem) { &ty.elem } else { "int" }, &V::I(*x))).collect()
        };
        let mut lines = Vec::new();
        let mut row = String::from("   ");
        for it in &items {
            if row.chars().count() + it.chars().count() + 2 > 110 {
                lines.push(row);
                row = String::from("   ");
            }
            row.push(' ');
            row.push_str(it);
            row.push(',');
        }
        lines.push(row);
        let dims: Vec<String> = ty.shape.iter().map(|d| d.to_string()).collect();
        let mut out = vec![format!("static const {} {}[{}] = {{ /* {}[{}] */", ctype, name, n, ty.elem, dims.join(","))];
        out.extend(lines);
        out.push("};".to_string());
        out
    }

    fn new_slot(&mut self, tag: &str, elem: &str) -> String {
        let pre = format!("{}_t", tag);
        let k = self.slots.iter().filter(|s| s.field.starts_with(&pre)).count();
        let field = format!("{}_t{}", tag, k);
        self.slots.push(Slot { field: field.clone(), elem: elem.to_string(), size: 0, note: "temporary".to_string() });
        field
    }

    fn grow_slot(&mut self, field: &str, size: usize) {
        let s = self.slots.iter_mut().find(|s| s.field == field).unwrap();
        s.size = s.size.max(size);
    }

    fn new_local(&mut self, tag: &str, slot: usize, cid: &str, ty: &Ty) -> String {
        let field = format!("{}_l{}_{}", tag, slot, cid);
        let s = Slot { field: field.clone(), elem: ty.elem.clone(), size: size_of(&ty.shape), note: format!("local {}: {}", cid, ty) };
        match self.slots.iter_mut().find(|x| x.field == field) {
            Some(x) => *x = s,
            None => self.slots.push(s),
        }
        field
    }

    fn fn_name(&self, f: &FnSpec) -> String {
        let cid = if f.kind != "op" { sanitize(&c_identifier(&f.name)) } else { "op".to_string() };
        format!("kg_{}__{}", cid, f.index.get())
    }

    fn inline(&self, f: &FnSpec) -> bool {
        matches!(f.kind, "op" | "lambda") && f.locals.is_empty() && f.stmts().len() == 1 && !f.effects() && !f.body.walked().iter().any(|n| matches!(n.k, K::ExternCall { .. }))
    }

    fn collect(&mut self, f: &F, writes: bool) -> HashSet<String> {
        let key = fn_key(f);
        if let Some(g) = if writes { self.writes_c.get(&key) } else { self.reads_c.get(&key) } {
            return g.clone();
        }
        if writes {
            self.writes_c.insert(key, HashSet::new());
        } else {
            self.reads_c.insert(key, HashSet::new());
        }
        let mut got: HashSet<String> = HashSet::new();
        let mut subs: Vec<F> = Vec::new();
        for n in f.body.walked() {
            match &n.k {
                K::Assign { target, .. } if writes => {
                    if let K::GlobalRef { name, .. } = &target.k {
                        got.insert(name.clone());
                    }
                }
                K::GlobalRef { name, state: true, .. } if !writes => {
                    got.insert(name.clone());
                }
                _ => {}
            }
            for g in n.fnspecs() {
                subs.push(g.clone());
            }
        }
        for g in subs {
            let s = self.collect(&g, writes);
            got.extend(s);
        }
        if writes {
            self.writes_c.insert(key, got.clone());
        } else {
            self.reads_c.insert(key, got.clone());
        }
        got
    }

    /// Names of the globals f assigns, directly or through what it calls.
    fn writes(&mut self, f: &F) -> HashSet<String> {
        self.collect(f, true)
    }

    /// Names of the state globals f reads or assigns, directly or through what it calls.
    fn touches(&mut self, f: &F) -> HashSet<String> {
        self.collect(f, false)
    }

    fn param_uses(&mut self, f: &F) -> HashMap<usize, usize> {
        let key = fn_key(f);
        if let Some(u) = self.uses_c.get(&key) {
            return u.clone();
        }
        let mut u: HashMap<usize, usize> = HashMap::new();
        for n in f.body.walked() {
            if let K::ParamRef { index, .. } = &n.k {
                *u.entry(*index).or_insert(0) += 1;
            }
        }
        self.uses_c.insert(key, u.clone());
        u
    }
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect()
}

// ---------------------------------------------------------------------------
// Compiling one function
// ---------------------------------------------------------------------------

/// The expansion of an operator as a closure over the shared flags (it records the use of `pow`).
#[derive(Clone)]
struct Expander {
    op: String,
    elem: String,
    line: usize,
    math: Rc<Cell<bool>>,
}

impl Expander {
    fn expand(&self, a: &str, b: &str) -> String {
        if is_fixed(&self.elem) {
            return fx_expand(&self.op, &self.elem, a, b, self.line).unwrap_or_else(|| "0".to_string());
        }
        if self.op == "pow" && self.elem == "float" {
            self.math.set(true);
        }
        expansion(&self.op, &self.elem, a, b, self.line).unwrap_or_else(|| "0".to_string())
    }
}

struct Gen<'a> {
    em: &'a mut Em,
    spec: F,
    code: Code,
    params: Vec<Vec<Val>>,
    locals: HashMap<usize, (bool, String)>,
    free: Vec<(String, String)>,
    live: Vec<(String, String)>,
    calls: Vec<F>,
    tag: String,
    line: usize,
}

type OVal = Option<Val>;

impl<'a> Gen<'a> {
    fn new(em: &'a mut Em, spec: F) -> Gen<'a> {
        let tag = format!("f{}", spec.index.get());
        Gen { em, spec, code: Code { lines: Vec::new(), depth: 1, n: 0, scalars: 0 }, params: Vec::new(), locals: HashMap::new(), free: Vec::new(), live: Vec::new(), calls: Vec::new(), tag, line: 0 }
    }

    fn fail(&mut self, _node: &Node, message: String) {
        if self.em.err.is_none() {
            self.em.err = Some(format!("the C emitter cannot compile this: {} [unsupported-op]", message));
        }
    }

    fn line_of(&self, node: &Node) -> usize {
        match (node.span, self.em.sources.get(self.spec.source)) {
            (Some(sp), Some(src)) => src.line(sp.0),
            _ => self.line,
        }
    }

    fn temp(&mut self, elem: &str, expr: &str, konst: bool) -> String {
        let name = self.code.name("t");
        self.code.scalars += 1;
        self.code.emit(&format!("{}{} {} = {};", if konst { "const " } else { "" }, ctype_of(elem), name, expr));
        name
    }

    /// A temp slot for an array of this type: a fresh Dest.
    fn slot(&mut self, ty: &Ty) -> Dest {
        let need = size_of(&ty.shape);
        let field = match self.free.iter().position(|(e, _)| *e == ty.elem) {
            Some(i) => self.free.remove(i).1,
            None => self.em.new_slot(&self.tag, &ty.elem),
        };
        self.em.grow_slot(&field, need);
        self.live.push((ty.elem.clone(), field.clone()));
        Dest { ptr: format!("p->{}", field), key: Some(format!("t:{}", field)), ty: ty.clone(), fresh: true }
    }

    fn release(&mut self) {
        let l = std::mem::take(&mut self.live);
        self.free.extend(l);
    }

    /// The HLS pragma of the loop just opened (stage B only).
    fn pragma(&mut self, count: Option<i64>, innermost: bool) {
        let Some(h) = self.em.hls.clone() else { return };
        if h.mode == "pipeline" || matches!(count, Some(c) if c <= h.unroll_max as i64) {
            self.code.emit("#pragma HLS unroll");
        } else if innermost {
            self.code.emit("#pragma HLS pipeline II=1");
        }
    }

    fn open_loops(&mut self, shape: &[usize]) -> Vec<String> {
        let mut idx = Vec::new();
        for (k, d) in shape.iter().enumerate() {
            let i = self.code.name("i");
            self.code.scalars += 1;
            self.code.emit(&format!("for (kg_int {} = 0; {} < {}; {}++) {{", i, i, d, i));
            self.code.depth += 1;
            self.pragma(Some(*d as i64), k == shape.len() - 1);
            idx.push(i);
        }
        idx
    }

    fn close(&mut self, n: usize) {
        for _ in 0..n {
            self.code.depth -= 1;
            self.code.emit("}");
        }
    }

    // -- values -----------------------------------------------------------------------
    fn conv(&mut self, v: Val, elem: &str) -> Val {
        if v.elem() == elem || (elem != "float" && !is_fixed(elem)) {
            return v;
        }
        let src = v.elem().to_string();
        if is_fixed(elem) || is_fixed(&src) {
            return match v {
                Val::S(s) => Val::S(S::new(elem, fx_conv(&src, elem, &s.c), s.reads)),
                Val::A(a) => {
                    let (e2, at) = (elem.to_string(), a.at.clone());
                    Val::A(A { ty: Ty::new(elem, a.ty.shape.clone()), at: Rc::new(move |idx: &[String]| fx_conv(&src, &e2, &at(idx))), reads: a.reads, ptr: None, key: None })
                }
            };
        }
        match v {
            Val::S(s) => {
                let c = if !s.atomic() { format!("(kg_float)({})", s.c) } else { format!("(kg_float){}", s.c) };
                Val::S(S::new("float", c, s.reads))
            }
            Val::A(a) => {
                let at = a.at.clone();
                Val::A(A { ty: Ty::new("float", a.ty.shape.clone()), at: Rc::new(move |idx: &[String]| format!("(kg_float)({})", at(idx))), reads: a.reads, ptr: None, key: None })
            }
        }
    }

    /// v as a value that later statements cannot change.
    fn pin(&mut self, v: OVal) -> OVal {
        let v = v?;
        if !v.has_reads() {
            return Some(v);
        }
        Some(match v {
            Val::S(s) => {
                let t = self.temp(&s.elem, &s.c, true);
                Val::S(S::new(&s.elem, t, BTreeSet::new()))
            }
            Val::A(a) => {
                let d = self.slot(&a.ty);
                Val::A(self.store(Val::A(a), &d))
            }
        })
    }

    fn pin1(&mut self, v: Val) -> Val {
        self.pin(Some(v)).unwrap()
    }

    /// A scalar that is cheap to repeat inside a loop.
    fn hoist(&mut self, v: Val) -> Val {
        match &v {
            Val::S(s) if !s.atomic() && s.elem != "str" => {
                let t = self.temp(&s.elem, &s.c, true);
                Val::S(S::new(&s.elem, t, BTreeSet::new()))
            }
            _ => v,
        }
    }

    /// An array view as a stored array (a pointer).
    fn materialize(&mut self, v: Val) -> A {
        match v {
            Val::A(a) if a.ptr.is_some() => a,
            Val::A(a) => {
                let d = self.slot(&a.ty);
                self.store(Val::A(a), &d)
            }
            Val::S(s) => stored(&Ty::new(&s.elem, vec![]), "0", None, false),
        }
    }

    /// Fill dest with the elements of v; returns the view of dest.
    fn store(&mut self, v: Val, dest: &Dest) -> A {
        let out = stored(&dest.ty, &dest.ptr, dest.key.as_deref(), !dest.fresh);
        let mut v = v;
        if let Val::A(a) = &v {
            if a.ptr.as_deref() == Some(dest.ptr.as_str()) {
                return out;
            }
            if let Some(k) = &dest.key {
                if a.reads.get(k) == Some(&false) {
                    let d = self.slot(&a.ty);
                    v = Val::A(self.store(v.clone(), &d));
                }
            }
        }
        v = self.conv(v, &dest.ty.elem);
        let shape = dest.ty.shape.clone();
        if matches!(v, Val::S(_)) {
            v = self.hoist(v);
        }
        let idx = self.open_loops(&shape);
        let rhs = match &v {
            Val::S(s) => s.c.clone(),
            Val::A(a) => (a.at)(&idx),
        };
        self.code.emit(&format!("{}[{}] = {};", dest.ptr, flat(&idx, &shape), rhs));
        self.close(shape.len());
        out
    }

    /// Compile nodes in this (evaluation) order, pinning a value when a later one has a side effect.
    fn operands(&mut self, nodes: &[&N]) -> Vec<OVal> {
        let mut vals = Vec::new();
        for (k, n) in nodes.iter().enumerate() {
            let mut v = self.compile(n, None);
            if nodes[k + 1..].iter().any(|m| m.effects()) {
                v = self.pin(v);
            }
            vals.push(v);
        }
        vals
    }

    fn operands1(&mut self, nodes: &[&N]) -> Vec<Val> {
        self.operands(nodes).into_iter().map(|v| v.expect("a value")).collect()
    }

    /// Item j (a C index expression) of an array value: a scalar or a view.
    fn row(&mut self, v: &Val, j: &str) -> Val {
        let rest: Vec<usize> = v.shape()[1..].to_vec();
        if rest.is_empty() {
            return Val::S(S::new(v.elem(), v.at(&[j.to_string()]), v.read_keys()));
        }
        let Val::A(a) = v else { unreachable!() };
        let mut ptr = None;
        if let Some(p) = &a.ptr {
            let off = mul(j, &size_of(&rest).to_string());
            ptr = Some(if off == "0" { p.clone() } else { format!("({} + {})", p, off) });
        }
        let (at, jj) = (a.at.clone(), j.to_string());
        let key = if ptr.is_some() { a.key.clone() } else { None };
        Val::A(A {
            ty: Ty::new(v.elem(), rest),
            at: Rc::new(move |idx: &[String]| {
                let mut full = vec![jj.clone()];
                full.extend(idx.iter().cloned());
                at(&full)
            }),
            reads: unaligned(&a.reads),
            ptr,
            key,
        })
    }

    fn row_dest(&self, dest: &Dest, j: &str) -> Dest {
        let rest: Vec<usize> = dest.ty.shape[1..].to_vec();
        let off = mul(j, &size_of(&rest).to_string());
        Dest { ptr: if off == "0" { dest.ptr.clone() } else { format!("({} + {})", dest.ptr, off) }, key: dest.key.clone(), ty: Ty::new(&dest.ty.elem, rest), fresh: dest.fresh }
    }

    /// dest if it may be filled while vals are still being read and fns are being called, else a new slot.
    fn target(&mut self, dest: Option<&Dest>, ty: &Ty, vals: &[&OVal], fns: &[F]) -> Dest {
        if let Some(d) = dest {
            if d.ty.shape == ty.shape && d.ty.elem == ty.elem {
                if d.fresh {
                    return d.clone();
                }
                let key = d.key.clone().unwrap_or_default();
                let mut ok = vals.iter().all(|v| v.as_ref().map(|v| !v.has_read(&key)).unwrap_or(true));
                if ok && key.starts_with("g:") {
                    let name = &key[2..];
                    for f in fns {
                        if self.em.touches(f).contains(name) {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    return d.clone();
                }
            }
        }
        self.slot(ty)
    }

    // -- dispatch --------------------------------------------------------------------------
    fn compile(&mut self, node: &N, dest: Option<&Dest>) -> OVal {
        match &node.k {
            K::Const(_) => Some(self.c_const(node)),
            K::GlobalRef { .. } => Some(self.c_global(node)),
            K::LocalRef { slot, .. } => Some(self.local(*slot)),
            K::ParamRef { index, .. } => Some(self.params.last().unwrap()[*index].clone()),
            K::Convert { a } => {
                let v = self.compile(a, None).unwrap();
                Some(self.conv(v, &node.ty.elem))
            }
            K::Unary { .. } => Some(self.c_unary(node)),
            K::Binary { .. } => Some(self.c_binary(node)),
            K::Match { .. } => Some(self.c_match(node)),
            K::Index { .. } => Some(self.c_index(node)),
            K::Gather { .. } => Some(self.c_gather(node)),
            K::Take { .. } => Some(self.c_take(node)),
            K::Drop { .. } => Some(self.c_drop(node)),
            K::Rotate { .. } => Some(self.c_rotate(node)),
            K::Reverse { .. } => Some(self.c_reverse(node)),
            K::Transpose { .. } => Some(self.c_transpose(node)),
            K::Reshape { .. } => Some(self.c_reshape(node)),
            K::Join { .. } => Some(self.c_join(node)),
            K::Iota { .. } => Some(Val::A(A { ty: node.ty.clone(), at: Rc::new(|idx: &[String]| idx[0].clone()), reads: BTreeMap::new(), ptr: None, key: None })),
            K::MakeList { .. } => Some(self.c_makelist(node, dest)),
            K::Amend { .. } => Some(self.c_amend(node, dest)),
            K::Cond { .. } => self.c_cond(node, dest),
            K::Call { .. } => self.c_call(node, dest),
            K::ExternCall { .. } => self.c_extern(node, dest),
            K::Map { .. } | K::Map2 { .. } | K::EachPair { .. } => self.c_each(node, dest),
            K::Reduce { .. } => self.fold(node, dest, false),
            K::Scan { .. } => self.fold(node, dest, true),
            K::Loop { .. } => self.c_loop(node, dest),
            K::Assign { .. } => self.c_assign(node),
            K::Seq { stmts } => {
                let mut v = None;
                for (i, s) in stmts.iter().enumerate() {
                    v = self.compile(s, if i == stmts.len() - 1 { dest } else { None });
                }
                v
            }
        }
    }

    fn c_const(&mut self, n: &N) -> Val {
        let K::Const(v) = &n.k else { unreachable!() };
        if n.ty.elem == "str" {
            let V::S(s) = v else { unreachable!() };
            return Val::S(S::new("str", format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")), BTreeSet::new()));
        }
        if n.ty.shape.is_empty() {
            return Val::S(S::new(&n.ty.elem, self.em.literal(&n.ty.elem, v), BTreeSet::new()));
        }
        let name = self.em.table(&n.ty, v, None);
        Val::A(stored(&n.ty, &name, None, false))
    }

    fn c_global(&mut self, n: &N) -> Val {
        let K::GlobalRef { name, state, .. } = &n.k else { unreachable!() };
        let cid = c_identifier(name);
        if !*state {
            return Val::A(stored(&n.ty, &format!("kg_c_{}", cid), None, false));
        }
        let key = format!("g:{}", name);
        if n.ty.shape.is_empty() {
            return Val::S(S::new(&n.ty.elem, format!("p->g_{}", cid), keyset(&[&key])));
        }
        Val::A(stored(&n.ty, &format!("p->g_{}", cid), Some(&key), true))
    }

    fn local(&mut self, slot: usize) -> Val {
        let ty = self.spec.locals[slot].clone();
        let (scalar, c) = self.locals[&slot].clone();
        if scalar {
            return Val::S(S::new(&ty.elem, c.clone(), keyset(&[&format!("v:{}", c)])));
        }
        Val::A(stored(&ty, &c, Some(&format!("l:{}", slot)), true))
    }

    fn expander(&self, op: &str, elem: &str, node: &Node) -> Expander {
        Expander { op: op.to_string(), elem: elem.to_string(), line: self.line_of(node), math: self.em.need_math.clone() }
    }

    fn c_unary(&mut self, n: &N) -> Val {
        let K::Unary { op, a, .. } = &n.k else { unreachable!() };
        let a = self.compile(a, None).unwrap();
        let ex = self.expander(op, a.elem(), n);
        match a {
            Val::S(s) => Val::S(S::new(&n.ty.elem, ex.expand(&s.c, ""), s.reads)),
            Val::A(a) => {
                let at = a.at.clone();
                Val::A(A { ty: n.ty.clone(), at: Rc::new(move |idx: &[String]| ex.expand(&at(idx), "")), reads: a.reads, ptr: None, key: None })
            }
        }
    }

    fn c_binary(&mut self, n: &N) -> Val {
        let K::Binary { op, a: na, b: nb } = &n.k else { unreachable!() };
        let mut vs = self.operands1(&[nb, na]);
        let (a, b) = (vs.pop().unwrap(), vs.pop().unwrap());
        let work = a.elem().to_string();
        let ex = self.expander(op, &work, n);
        if let (Val::S(x), Val::S(y)) = (&a, &b) {
            let reads: BTreeSet<String> = x.reads.union(&y.reads).cloned().collect();
            return Val::S(S::new(&n.ty.elem, ex.expand(&x.c, &y.c), reads));
        }
        let rank = n.ty.shape.len();
        let mut parts: Vec<At> = Vec::new();
        let mut reads: BTreeMap<String, bool> = BTreeMap::new();
        for v in [a, b] {
            match v {
                Val::S(_) => {
                    let h = self.hoist(v);
                    let Val::S(s) = self.pin1(h) else { unreachable!() };
                    let c = s.c.clone();
                    parts.push(Rc::new(move |_idx: &[String]| c.clone()));
                }
                Val::A(av) => {
                    let k = av.ty.shape.len();
                    let at = av.at.clone();
                    parts.push(Rc::new(move |idx: &[String]| at(&idx[..k])));
                    let r = if k == rank { av.reads.clone() } else { unaligned(&av.reads) };
                    reads = merge(&[&reads, &r]);
                }
            }
        }
        let (p0, p1) = (parts[0].clone(), parts[1].clone());
        Val::A(A { ty: n.ty.clone(), at: Rc::new(move |idx: &[String]| ex.expand(&p0(idx), &p1(idx))), reads, ptr: None, key: None })
    }

    fn c_match(&mut self, n: &N) -> Val {
        let K::Match { a: na, b: nb } = &n.k else { unreachable!() };
        let mut vs = self.operands1(&[nb, na]);
        let (a, b) = (vs.pop().unwrap(), vs.pop().unwrap());
        let (ie, fe) = c_expansion("match").unwrap();
        let mut form = if a.elem() == "float" { fe.unwrap() } else { ie.unwrap() }.to_string();
        if is_fixed(a.elem()) {
            form = "((kg_int)(({a}) == ({b})))".to_string();
        }
        if let (Val::S(x), Val::S(y)) = (&a, &b) {
            let reads: BTreeSet<String> = x.reads.union(&y.reads).cloned().collect();
            return Val::S(S::new("int", form.replace("{a}", &format!("({})", x.c)).replace("{b}", &format!("({})", y.c)), reads));
        }
        let t = self.temp("int", "1", false);
        let shape = a.shape().to_vec();
        let idx = self.open_loops(&shape);
        let eq = if a.elem() == "float" { "kg_feq" } else { "kg_ieq" };
        let test = if is_fixed(a.elem()) { format!("({}) == ({})", a.at(&idx), b.at(&idx)) } else { format!("{}({}, {})", eq, a.at(&idx), b.at(&idx)) };
        self.code.emit(&format!("if (!({})) {{", test));
        self.code.emit(&format!("    {} = 0;", t));
        self.code.emit("}");
        self.close(shape.len());
        Val::S(S::new("int", t, BTreeSet::new()))
    }

    fn index_expr(&self, i: &str, n: usize, checked: bool, node: &Node) -> String {
        if checked {
            i.to_string()
        } else {
            format!("kg_ix({}, {}, {})", i, n, self.line_of(node))
        }
    }

    fn c_index(&mut self, n: &N) -> Val {
        let K::Index { a: na, i: ni, checked } = &n.k else { unreachable!() };
        let mut vs = self.operands1(&[ni, na]);
        let (a, i) = (vs.pop().unwrap(), vs.pop().unwrap());
        let j = if *checked {
            i.c()
        } else {
            let e = self.index_expr(&i.c(), na.ty.shape[0], false, n);
            self.temp("int", &e, true)
        };
        self.row(&a, &j)
    }

    fn c_gather(&mut self, n: &N) -> Val {
        let K::Gather { a: na, idx: nidx, checked } = &n.k else { unreachable!() };
        if self.em.window_leaves.contains(&(Rc::as_ptr(n) as usize)) {
            return Val::A(A { ty: n.ty.clone(), at: Rc::new(|i: &[String]| format!("win[{}]", i[0])), reads: BTreeMap::new(), ptr: None, key: None });
        }
        let mut vs = self.operands1(&[nidx, na]);
        let (a, idx) = (vs.pop().unwrap(), vs.pop().unwrap());
        let r = idx.shape().len();
        let size = na.ty.shape[0];
        let line = self.line_of(n);
        let checked = *checked;
        let (aat, iat) = (match &a { Val::A(x) => x.at.clone(), Val::S(s) => { let c = s.c.clone(); Rc::new(move |_: &[String]| c.clone()) as At } }, match &idx { Val::A(x) => x.at.clone(), Val::S(s) => { let c = s.c.clone(); Rc::new(move |_: &[String]| c.clone()) as At } });
        let reads = unaligned(&merge(&[&reads_of(&idx), &reads_of(&a)]));
        Val::A(A {
            ty: n.ty.clone(),
            at: Rc::new(move |i: &[String]| {
                let e = iat(&i[..r]);
                let ix = if checked { e } else { format!("kg_ix({}, {}, {})", e, size, line) };
                let mut full = vec![ix];
                full.extend(i[r..].iter().cloned());
                aat(&full)
            }),
            reads,
            ptr: None,
            key: None,
        })
    }

    fn view_at(v: &Val) -> At {
        match v {
            Val::A(x) => x.at.clone(),
            Val::S(s) => {
                let c = s.c.clone();
                Rc::new(move |_: &[String]| c.clone())
            }
        }
    }

    fn c_take(&mut self, n: &N) -> Val {
        let K::Take { a: na, n: cnt } = &n.k else { unreachable!() };
        let a = self.compile(na, None).unwrap();
        let size = na.ty.shape[0] as i64;
        let count = cnt.abs();
        let start = if *cnt > 0 { 0 } else { (-count).rem_euclid(size) };
        let at = Self::view_at(&a);
        let st = start.to_string();
        let reads = unaligned(&reads_of(&a));
        if start + count <= size {
            return Val::A(A { ty: n.ty.clone(), at: Rc::new(move |i: &[String]| at(&[add(&i[0], &st)])), reads, ptr: None, key: None });
        }
        Val::A(A { ty: n.ty.clone(), at: Rc::new(move |i: &[String]| at(&[format!("({} % {})", add(&i[0], &st), size)])), reads, ptr: None, key: None })
    }

    fn c_drop(&mut self, n: &N) -> Val {
        let K::Drop { a: na, n: cnt } = &n.k else { unreachable!() };
        let a = self.compile(na, None).unwrap();
        let off = if *cnt > 0 { *cnt } else { 0 }.to_string();
        let at = Self::view_at(&a);
        let reads = unaligned(&reads_of(&a));
        Val::A(A {
            ty: n.ty.clone(),
            at: Rc::new(move |i: &[String]| {
                let mut full = vec![add(&i[0], &off)];
                full.extend(i[1..].iter().cloned());
                at(&full)
            }),
            reads,
            ptr: None,
            key: None,
        })
    }

    fn c_rotate(&mut self, n: &N) -> Val {
        let K::Rotate { a: na, n: nk } = &n.k else { unreachable!() };
        let mut vs = self.operands1(&[na, nk]);
        let (k, a) = (vs.pop().unwrap(), vs.pop().unwrap());
        let size = n.ty.shape[0] as i64;
        let back: String = match lit(&k.c()) {
            Some(l) => ((size - l.rem_euclid(size)).rem_euclid(size)).to_string(),
            None => self.temp("int", &format!("{} - kg_rot({}, {})", size, k.c(), size), true),
        };
        let at = Self::view_at(&a);
        let reads = unaligned(&reads_of(&a));
        Val::A(A { ty: n.ty.clone(), at: Rc::new(move |i: &[String]| at(&[format!("({} % {})", add(&i[0], &back), size)])), reads, ptr: None, key: None })
    }

    fn c_reverse(&mut self, n: &N) -> Val {
        let K::Reverse { a: na } = &n.k else { unreachable!() };
        let a = self.compile(na, None).unwrap();
        let last = (n.ty.shape[0] - 1).to_string();
        let at = Self::view_at(&a);
        let reads = unaligned(&reads_of(&a));
        Val::A(A {
            ty: n.ty.clone(),
            at: Rc::new(move |i: &[String]| {
                let mut full = vec![sub(&last, &i[0])];
                full.extend(i[1..].iter().cloned());
                at(&full)
            }),
            reads,
            ptr: None,
            key: None,
        })
    }

    fn c_transpose(&mut self, n: &N) -> Val {
        let K::Transpose { a: na } = &n.k else { unreachable!() };
        let a = self.compile(na, None).unwrap();
        let at = Self::view_at(&a);
        let reads = unaligned(&reads_of(&a));
        Val::A(A { ty: n.ty.clone(), at: Rc::new(move |i: &[String]| at(&[i[1].clone(), i[0].clone()])), reads, ptr: None, key: None })
    }

    fn c_reshape(&mut self, n: &N) -> Val {
        let K::Reshape { a: na } = &n.k else { unreachable!() };
        let a = self.compile(na, None).unwrap();
        let shape = n.ty.shape.clone();
        if let Val::S(_) = a {
            let h = self.hoist(a);
            let Val::S(s) = self.pin1(h) else { unreachable!() };
            let c = s.c;
            return Val::A(A { ty: n.ty.clone(), at: Rc::new(move |_: &[String]| c.clone()), reads: BTreeMap::new(), ptr: None, key: None });
        }
        let mut want = vec![1usize];
        want.extend(a.shape().iter());
        if shape == want {
            let at = Self::view_at(&a);
            let reads = unaligned(&reads_of(&a));
            return Val::A(A { ty: n.ty.clone(), at: Rc::new(move |i: &[String]| at(&i[1..])), reads, ptr: None, key: None });
        }
        let m = self.materialize(a);
        let have = size_of(&m.ty.shape);
        let ptr = m.ptr.clone().unwrap();
        let reads = unaligned(&m.reads);
        if size_of(&shape) <= have {
            return Val::A(A { ty: n.ty.clone(), at: Rc::new(move |i: &[String]| format!("{}[{}]", ptr, flat(i, &shape))), reads, ptr: None, key: None });
        }
        Val::A(A { ty: n.ty.clone(), at: Rc::new(move |i: &[String]| format!("{}[{} % {}]", ptr, flat(i, &shape), have)), reads, ptr: None, key: None })
    }

    fn c_join(&mut self, n: &N) -> Val {
        let K::Join { a: na, b: nb } = &n.k else { unreachable!() };
        let mut vs = self.operands1(&[nb, na]);
        let (mut a, mut b) = (vs.pop().unwrap(), vs.pop().unwrap());
        let nlen = if a.shape().is_empty() { 1 } else { a.shape()[0] };
        if let Val::S(_) = a {
            let h = self.hoist(a);
            a = self.pin1(h);
        }
        if let Val::S(_) = b {
            let h = self.hoist(b);
            b = self.pin1(h);
        }
        let reads = unaligned(&merge(&[&reads_of(&a), &reads_of(&b)]));
        let (aat, bat) = (Self::view_at(&a), Self::view_at(&b));
        let (a_s, b_s) = (matches!(a, Val::S(_)), matches!(b, Val::S(_)));
        let _ = (a_s, b_s);
        Val::A(A {
            ty: n.ty.clone(),
            at: Rc::new(move |i: &[String]| {
                let (j, rest) = (&i[0], &i[1..]);
                let part = |at: &At, jj: &str| {
                    let mut full = vec![jj.to_string()];
                    full.extend(rest.iter().cloned());
                    at(&full)
                };
                match lit(j) {
                    Some(l) => {
                        if l < nlen as i64 {
                            part(&aat, j)
                        } else {
                            part(&bat, &(l - nlen as i64).to_string())
                        }
                    }
                    None => format!("({} < {} ? {} : {})", j, nlen, part(&aat, j), part(&bat, &sub(j, &nlen.to_string()))),
                }
            }),
            reads,
            ptr: None,
            key: None,
        })
    }

    fn c_makelist(&mut self, n: &N, dest: Option<&Dest>) -> Val {
        let K::MakeList { items } = &n.k else { unreachable!() };
        let t = match dest {
            Some(d) if d.fresh && d.ty == n.ty => d.clone(),
            _ => self.slot(&n.ty),
        };
        for (k, item) in items.iter().enumerate() {
            let v = self.compile(item, None).unwrap();
            match v {
                Val::S(s) => self.code.emit(&format!("{}[{}] = {};", t.ptr, k, s.c)),
                v => {
                    let rd = self.row_dest(&t, &k.to_string());
                    self.store(v, &rd);
                }
            }
        }
        Val::A(stored(&n.ty, &t.ptr, t.key.as_deref(), false))
    }

    fn c_amend(&mut self, n: &N, dest: Option<&Dest>) -> Val {
        let K::Amend { a: na, value: nv, idx: ni, checked } = &n.k else { unreachable!() };
        let mut vs = self.operands1(&[ni, nv, na]);
        let (a, value, mut idx) = (vs.pop().unwrap(), vs.pop().unwrap(), vs.pop().unwrap());
        let value = {
            let h = self.hoist(value);
            self.pin1(h)
        };
        if let (Some(d), Val::A(ia)) = (dest, &idx) {
            if let Some(k) = &d.key {
                if ia.reads.contains_key(k) {
                    if !d.fresh {
                        idx = self.pin1(idx.clone());
                    }
                    if idx.has_read(k) {
                        let Val::A(ia) = &idx else { unreachable!() };
                        let sl = self.slot(&ia.ty);
                        idx = Val::A(self.store(idx.clone(), &sl));
                    }
                }
            }
        }
        if let Val::S(_) = idx {
            let h = self.hoist(idx);
            idx = self.pin1(h);
        }
        let t = self.target(dest, &n.ty, &[&Some(idx.clone())], &[]);
        self.store(a, &t);
        let size = n.ty.shape[0];
        if let Val::S(s) = &idx {
            let ie = self.index_expr(&s.c, size, *checked, n);
            self.code.emit(&format!("{}[{}] = {};", t.ptr, ie, value.c()));
        } else {
            let shape = idx.shape().to_vec();
            let lp = self.open_loops(&shape);
            let ie = self.index_expr(&idx.at(&lp), size, *checked, n);
            self.code.emit(&format!("{}[{}] = {};", t.ptr, ie, value.c()));
            self.close(lp.len());
        }
        Val::A(stored(&n.ty, &t.ptr, t.key.as_deref(), !t.fresh))
    }

    /// Compile node into a separate list of lines (one block deeper).
    fn branch(&mut self, node: &N) -> (Vec<String>, OVal) {
        let saved = std::mem::take(&mut self.code.lines);
        self.code.depth += 1;
        let v = self.compile(node, None);
        self.code.depth -= 1;
        let lines = std::mem::replace(&mut self.code.lines, saved);
        (lines, v)
    }

    fn c_cond(&mut self, n: &N, dest: Option<&Dest>) -> OVal {
        let K::Cond { c: nc, t: nt, e: ne } = &n.k else { unreachable!() };
        let c = self.compile(nc, None).unwrap();
        let cond = format!("{} != 0", c.c());
        let (lt, vt) = self.branch(nt);
        let (le, ve) = self.branch(ne);
        if n.ty.elem == "void" {
            self.code.emit(&format!("if ({}) {{", cond));
            self.code.lines.extend(lt);
            self.code.emit("} else {");
            self.code.lines.extend(le);
            self.code.emit("}");
            return None;
        }
        if n.ty.shape.is_empty() {
            let (vt, ve) = (vt.unwrap(), ve.unwrap());
            if lt.is_empty() && le.is_empty() {
                let mut reads = c.read_keys();
                reads.extend(vt.read_keys());
                reads.extend(ve.read_keys());
                return Some(Val::S(S::new(&n.ty.elem, format!("({} ? {} : {})", cond, vt.c(), ve.c()), reads)));
            }
            let r = self.code.name("t");
            self.code.scalars += 1;
            self.code.emit(&format!("{} {};", ctype_of(&n.ty.elem), r));
            self.code.emit(&format!("if ({}) {{", cond));
            self.code.lines.extend(lt);
            self.code.emit(&format!("    {} = {};", r, vt.c()));
            self.code.emit("} else {");
            self.code.lines.extend(le);
            self.code.emit(&format!("    {} = {};", r, ve.c()));
            self.code.emit("}");
            return Some(Val::S(S::new(&n.ty.elem, r, BTreeSet::new())));
        }
        let t = if dest.is_some() { self.target(dest, &n.ty, &[&vt, &ve], &[]) } else { self.slot(&n.ty) };
        self.code.emit(&format!("if ({}) {{", cond));
        self.code.lines.extend(lt);
        self.code.depth += 1;
        self.store(vt.unwrap(), &t);
        self.code.depth -= 1;
        self.code.emit("} else {");
        self.code.lines.extend(le);
        self.code.depth += 1;
        self.store(ve.unwrap(), &t);
        self.code.depth -= 1;
        self.code.emit("}");
        Some(Val::A(stored(&n.ty, &t.ptr, t.key.as_deref(), !t.fresh)))
    }

    // -- calls --------------------------------------------------------------------------------
    fn arg(&mut self, v: Val, f: Option<&F>) -> String {
        if let Val::S(s) = v {
            return s.c;
        }
        let mut m = self.materialize(v);
        if let (Some(f), Some(k)) = (f, m.key.clone()) {
            if k.starts_with("g:") && self.em.writes(f).contains(&k[2..]) {
                let d = self.slot(&m.ty);
                m = self.store(Val::A(m), &d);
            }
        }
        m.ptr.unwrap_or_default()
    }

    fn call_c(&mut self, name: &str, with_p: bool, result: &Ty, vals: Vec<Val>, dest: Option<&Dest>, f: Option<&F>) -> OVal {
        let mut args: Vec<String> = if with_p { vec!["p".to_string()] } else { Vec::new() };
        for v in vals {
            let a = self.arg(v, f);
            args.push(a);
        }
        if result.elem == "void" {
            self.code.emit(&format!("{}({});", name, args.join(", ")));
            return None;
        }
        if result.shape.is_empty() {
            let t = self.temp(&result.elem, &format!("{}({})", name, args.join(", ")), true);
            return Some(Val::S(S::new(&result.elem, t, BTreeSet::new())));
        }
        let t = match dest {
            Some(d) if d.fresh && d.ty == *result => d.clone(),
            _ => self.slot(result),
        };
        args.push(t.ptr.clone());
        self.code.emit(&format!("{}({});", name, args.join(", ")));
        Some(Val::A(stored(result, &t.ptr, t.key.as_deref(), false)))
    }

    fn c_call(&mut self, n: &N, dest: Option<&Dest>) -> OVal {
        let K::Call { f, args } = &n.k else { unreachable!() };
        let refs: Vec<&N> = args.iter().collect();
        let vals = self.operands1(&refs);
        self.calls.push(f.clone());
        let name = self.em.fn_name(f);
        self.call_c(&name, true, &f.result_ty, vals, dest, Some(f))
    }

    fn c_extern(&mut self, n: &N, dest: Option<&Dest>) -> OVal {
        let K::ExternCall { ext, args } = &n.k else { unreachable!() };
        let refs: Vec<&N> = args.iter().collect();
        let vals = self.operands1(&refs);
        self.call_c(&c_name_of(ext), false, &n.ty, vals, dest, None)
    }

    /// The value of fn applied to these argument values: inlined when fn is a small pure verb, else a C call.
    fn apply(&mut self, f: &F, vals: Vec<Val>, dest: Option<&Dest>) -> OVal {
        if self.em.inline(f) {
            let uses = self.em.param_uses(f);
            let mut bound = Vec::new();
            for (i, v) in vals.into_iter().enumerate() {
                let v = if matches!(v, Val::S(_)) && uses.get(&i).copied().unwrap_or(0) > 1 { self.hoist(v) } else { v };
                bound.push(v);
            }
            self.params.push(bound);
            let r = self.compile(&f.stmts()[0].clone(), dest);
            self.params.pop();
            return r;
        }
        self.calls.push(f.clone());
        let name = self.em.fn_name(f);
        self.call_c(&name, true, &f.result_ty, vals, dest, Some(f))
    }

    fn captured(&mut self, nodes: &[N]) -> Vec<Val> {
        let mut out = Vec::new();
        for c in nodes {
            let v = self.compile(c, None).unwrap();
            out.push(if c.ty.shape.is_empty() {
                let h = self.hoist(v);
                self.pin1(h)
            } else {
                v
            });
        }
        out
    }

    /// Operands of an adverb whose verb has side effects are evaluated in full before the first call.
    fn settle(&mut self, vals: Vec<Val>, fns: &[&F]) -> Vec<Val> {
        if fns.iter().any(|f| f.effects()) {
            return vals.into_iter().map(|v| self.pin1(v)).collect();
        }
        vals
    }

    fn c_each(&mut self, n: &N, dest: Option<&Dest>) -> OVal {
        let (f, operands, caps_n, count): (&F, Vec<&N>, &Vec<N>, usize) = match &n.k {
            K::Map { f, a, caps } => (f, vec![a], caps, a.ty.shape[0]),
            K::Map2 { f, a, b, caps } => (f, vec![a, b], caps, a.ty.shape[0]),
            K::EachPair { f, a, caps } => (f, vec![a], caps, a.ty.shape[0] - 1),
            _ => unreachable!(),
        };
        let vals = self.operands1(&operands);
        let caps = self.captured(caps_n);
        let vals = self.settle(vals, &[f]);
        let r = f.result_ty.clone();
        let t = if r.elem == "void" {
            None
        } else {
            let all: Vec<OVal> = vals.iter().chain(caps.iter()).cloned().map(Some).collect();
            let refs: Vec<&OVal> = all.iter().collect();
            Some(self.target(dest, &n.ty, &refs, &[f.clone()]))
        };
        let j = self.open_loops(&[count]).remove(0);
        let mut items: Vec<Val> = match &n.k {
            K::Map { .. } => vec![self.row(&vals[0], &j)],
            K::Map2 { .. } => vec![self.row(&vals[0], &j), self.row(&vals[1], &j)],
            _ => vec![self.row(&vals[0], &j), self.row(&vals[0], &add(&j, "1"))],
        };
        items.extend(caps.iter().cloned());
        let rd = match &t {
            Some(t) if !r.shape.is_empty() => Some(self.row_dest(t, &j)),
            _ => None,
        };
        let res = self.apply(f, items, rd.as_ref());
        if let Some(t) = &t {
            if !r.shape.is_empty() {
                let rd = self.row_dest(t, &j);
                self.store(res.unwrap(), &rd);
            } else {
                let c = self.conv(res.unwrap(), &n.ty.elem).c();
                self.code.emit(&format!("{}[{}] = {};", t.ptr, j, c));
            }
        }
        self.close(1);
        t.map(|t| Val::A(stored(&n.ty, &t.ptr, t.key.as_deref(), !t.fresh)))
    }

    /// Over and Scan.
    fn fold(&mut self, n: &N, dest: Option<&Dest>, scan: bool) -> OVal {
        let (f, na, seed, caps_n, acc_ty) = match &n.k {
            K::Reduce { f, a, seed, caps } => (f, a, seed, caps, n.ty.clone()),
            K::Scan { f, a, seed, caps, acc_ty } => (f, a, seed, caps, acc_ty.clone()),
            _ => unreachable!(),
        };
        let mut nodes: Vec<&N> = Vec::new();
        if let Some(s) = seed {
            nodes.push(s);
        }
        nodes.push(na);
        let vals = self.operands1(&nodes);
        let caps = self.captured(caps_n);
        let vals = self.settle(vals, &[f]);
        let a = vals.last().unwrap().clone();
        let count = na.ty.shape[0];
        let init = if seed.is_some() { vals[0].clone() } else { self.row(&a, "0") };
        let first = if seed.is_some() { 0 } else { 1 };
        let all: Vec<OVal> = vals.iter().chain(caps.iter()).cloned().map(Some).collect();
        let refs: Vec<&OVal> = all.iter().collect();
        let out = if scan { Some(self.target(dest, &n.ty, &refs, &[f.clone()])) } else { None };
        let k0: i64 = if scan { 1 } else { 0 };
        if acc_ty.shape.is_empty() && !scan && self.em.hls.is_some() && caps.is_empty() {
            if let Some(tree) = self.tree_fold(n, f, &init, &a, first, count, &acc_ty) {
                return Some(tree);
            }
        }
        if acc_ty.shape.is_empty() {
            let ic = self.conv(init, &acc_ty.elem).c();
            let acc = self.temp(&acc_ty.elem, &ic, false);
            if scan {
                self.code.emit(&format!("{}[0] = {};", out.as_ref().unwrap().ptr, acc));
            }
            if count > first {
                let j = self.code.name("i");
                self.code.scalars += 1;
                self.code.emit(&format!("for (kg_int {} = {}; {} < {}; {}++) {{", j, first, j, count, j));
                self.code.depth += 1;
                self.pragma(Some((count - first) as i64), true);
                let mut args = vec![Val::S(S::new(&acc_ty.elem, acc.clone(), BTreeSet::new())), self.row(&a, &j)];
                args.extend(caps.iter().cloned());
                let res = self.apply(f, args, None).unwrap();
                let c = self.conv(res, &acc_ty.elem).c();
                self.code.emit(&format!("{} = {};", acc, c));
                if scan {
                    let off = (k0 - first as i64).to_string();
                    self.code.emit(&format!("{}[{}] = {};", out.as_ref().unwrap().ptr, add(&j, &off), acc));
                }
                self.close(1);
            }
            return Some(if scan {
                let o = out.unwrap();
                Val::A(stored(&n.ty, &o.ptr, o.key.as_deref(), !o.fresh))
            } else {
                Val::S(S::new(&acc_ty.elem, acc, BTreeSet::new()))
            });
        }
        if scan {
            let o = out.unwrap();
            let rd = self.row_dest(&o, "0");
            self.store(init, &rd);
            if count > first {
                let j = self.code.name("i");
                self.code.scalars += 1;
                self.code.emit(&format!("for (kg_int {} = {}; {} < {}; {}++) {{", j, first, j, count, j));
                self.code.depth += 1;
                self.pragma(Some((count - first) as i64), true);
                let prev = self.row_dest(&o, &add(&j, &(k0 - first as i64 - 1).to_string()));
                let cur = self.row_dest(&o, &add(&j, &(k0 - first as i64).to_string()));
                let mut args = vec![Val::A(stored(&acc_ty, &prev.ptr, None, false)), self.row(&a, &j)];
                args.extend(caps.iter().cloned());
                let res = self.apply(f, args, None).unwrap();
                self.store(res, &Dest { ptr: cur.ptr.clone(), key: None, ty: acc_ty.clone(), fresh: true });
                self.close(1);
            }
            return Some(Val::A(stored(&n.ty, &o.ptr, o.key.as_deref(), !o.fresh)));
        }
        let t = self.target(dest, &acc_ty, &refs, &[f.clone()]);
        self.store(init, &t);
        if count > first {
            let j = self.code.name("i");
            self.code.scalars += 1;
            self.code.emit(&format!("for (kg_int {} = {}; {} < {}; {}++) {{", j, first, j, count, j));
            self.code.depth += 1;
            self.pragma(Some((count - first) as i64), true);
            let st = stored(&acc_ty, &t.ptr, None, false);
            let mut reads = BTreeMap::new();
            if let Some(k) = &t.key {
                reads.insert(k.clone(), true);
            }
            let accv = Val::A(A { ty: acc_ty.clone(), at: st.at.clone(), reads, ptr: Some(t.ptr.clone()), key: t.key.clone() });
            let mut args = vec![accv, self.row(&a, &j)];
            args.extend(caps.iter().cloned());
            let res = self.apply(f, args, None).unwrap();
            self.store(res, &Dest { ptr: t.ptr.clone(), key: t.key.clone(), ty: acc_ty.clone(), fresh: false });
            self.close(1);
        }
        Some(Val::A(stored(&acc_ty, &t.ptr, t.key.as_deref(), !t.fresh)))
    }

    /// A reduction by + | & of a short array as a balanced tree (hardware); None when the form does not apply.
    #[allow(clippy::too_many_arguments)]
    fn tree_fold(&mut self, n: &N, f: &F, init: &Val, a: &Val, first: usize, count: usize, acc_ty: &Ty) -> Option<Val> {
        let _ = first;
        let h = self.em.hls.clone()?;
        let stmts = f.stmts();
        let body = if f.kind == "op" && stmts.len() == 1 { Some(&stmts[0]) } else { None }?;
        let K::Binary { op, a: ba, b: bb } = &body.k else { return None };
        if !matches!(op.as_str(), "add" | "max" | "min") {
            return None;
        }
        if !matches!((&ba.k, &bb.k), (K::ParamRef { index: 0, .. }, K::ParamRef { index: 1, .. })) {
            return None;
        }
        if acc_ty.elem == "float" && !h.reassociate {
            return None;
        }
        if acc_ty.elem != "float" && acc_ty.elem != "int" && !is_fixed(&acc_ty.elem) {
            return None;
        }
        let na = match &n.k {
            K::Reduce { a, .. } | K::Scan { a, .. } => a,
            _ => return None,
        };
        if count < 2 || count > h.tree_max || na.ty.elem != acc_ty.elem {
            return None;
        }
        let mut items: Vec<String> = (0..count).map(|j| self.row(a, &j.to_string()).c()).collect();
        let ex = self.expander(op, &acc_ty.elem, n);
        while items.len() > 1 {
            let mut nxt: Vec<String> = Vec::new();
            let mut k = 0;
            while k + 1 < items.len() {
                nxt.push(ex.expand(&items[k], &items[k + 1]));
                k += 2;
            }
            if items.len() % 2 == 1 {
                nxt.push(items.last().unwrap().clone());
            }
            items = nxt;
        }
        let mut expr = items.remove(0);
        let seeded = matches!(&n.k, K::Reduce { seed: Some(_), .. } | K::Scan { seed: Some(_), .. });
        if seeded {
            let ic = self.conv(init.clone(), &acc_ty.elem).c();
            expr = ex.expand(&ic, &expr);
        }
        let t = self.temp(&acc_ty.elem, &expr, true);
        Some(Val::S(S::new(&acc_ty.elem, t, BTreeSet::new())))
    }

    fn c_loop(&mut self, n: &N, dest: Option<&Dest>) -> OVal {
        let K::Loop { kind, f, init: ninit, count: ncount, pred, caps: caps_n, pcaps: pcaps_n } = &n.k else { unreachable!() };
        let (mut init, count): (Val, Option<Val>);
        if kind == "iterate" {
            let mut vs = self.operands1(&[ncount.as_ref().unwrap(), ninit]);
            init = vs.pop().unwrap();
            let c = vs.pop().unwrap();
            let h = self.hoist(c);
            count = Some(self.pin1(h));
        } else {
            init = self.operands1(&[ninit]).pop().unwrap();
            count = None;
        }
        let caps = self.captured(caps_n);
        let pcaps = self.captured(pcaps_n);
        let mut fns: Vec<&F> = vec![f];
        if let Some(p) = pred {
            fns.push(p);
        }
        init = self.settle(vec![init], &fns).remove(0);
        let ty = n.ty.clone();
        let fvec: Vec<F> = fns.iter().map(|x| (*x).clone()).collect();
        let mut t: Option<Dest> = None;
        let mut var = String::new();
        if !ty.shape.is_empty() {
            let all: Vec<OVal> = std::iter::once(&init).chain(caps.iter()).chain(pcaps.iter()).cloned().map(Some).collect();
            let refs: Vec<&OVal> = all.iter().collect();
            let td = self.target(dest, &ty, &refs, &fvec);
            self.store(init.clone(), &td);
            t = Some(td);
        } else {
            var = self.temp(&ty.elem, &init.c(), false);
        }
        let state = |t: &Option<Dest>, var: &str| -> Val {
            if let Some(t) = t {
                let st = stored(&ty, &t.ptr, None, false);
                let mut reads = BTreeMap::new();
                if let Some(k) = &t.key {
                    reads.insert(k.clone(), true);
                }
                Val::A(A { ty: ty.clone(), at: st.at.clone(), reads, ptr: Some(t.ptr.clone()), key: t.key.clone() })
            } else {
                Val::S(S::new(&ty.elem, var.to_string(), BTreeSet::new()))
            }
        };
        if kind == "iterate" {
            let k = self.code.name("i");
            self.code.scalars += 1;
            let cc = count.as_ref().unwrap().c();
            self.code.emit(&format!("for (kg_int {} = 0; {} < {}; {}++) {{", k, k, cc, k));
            self.code.depth += 1;
            self.pragma(lit(&cc), true);
        } else {
            self.code.emit("for (;;) {");
            self.code.depth += 1;
            let mut args = vec![state(&t, &var)];
            args.extend(pcaps.iter().cloned());
            let c = self.apply(pred.as_ref().unwrap(), args, None).unwrap();
            self.code.emit(&format!("if (!({} != 0)) {{", c.c()));
            self.code.emit("    break;");
            self.code.emit("}");
        }
        let mut args = vec![state(&t, &var)];
        args.extend(caps.iter().cloned());
        let res = self.apply(f, args, None).unwrap();
        if let Some(t) = &t {
            self.store(res, &Dest { ptr: t.ptr.clone(), key: t.key.clone(), ty: ty.clone(), fresh: false });
        } else {
            self.code.emit(&format!("{} = {};", var, res.c()));
        }
        self.close(1);
        Some(if let Some(t) = &t { Val::A(stored(&ty, &t.ptr, t.key.as_deref(), !t.fresh)) } else { Val::S(S::new(&ty.elem, var, BTreeSet::new())) })
    }

    // -- statements -------------------------------------------------------------------------------
    fn c_assign(&mut self, n: &N) -> OVal {
        let K::Assign { target, value } = &n.k else { unreachable!() };
        let (c, key): (String, String) = match &target.k {
            K::LocalRef { slot, .. } => {
                let (scalar, c) = self.locals[slot].clone();
                let key = if scalar { format!("v:{}", c) } else { format!("l:{}", slot) };
                (c, key)
            }
            K::GlobalRef { name, .. } => (format!("p->g_{}", c_identifier(name)), format!("g:{}", name)),
            _ => unreachable!(),
        };
        if n.ty.shape.is_empty() {
            let v = self.compile(value, None).unwrap();
            self.code.emit(&format!("{} = {};", c, v.c()));
            return Some(Val::S(S::new(&n.ty.elem, c, keyset(&[&key]))));
        }
        let d = Dest { ptr: c, key: Some(key), ty: n.ty.clone(), fresh: false };
        let v = self.compile(value, Some(&d)).unwrap();
        Some(Val::A(self.store(v, &d)))
    }

    // -- the function --------------------------------------------------------------------------------
    fn build(&mut self) -> (String, Vec<String>) {
        let spec = self.spec.clone();
        let prefix = self.em.prefix.clone();
        let mut own: Vec<Val> = Vec::new();
        let mut decl = vec![format!("{}_t *p", prefix)];
        for (i, (name, ty)) in spec.param_names.iter().zip(&spec.param_tys).enumerate() {
            let c = format!("a{}_{}", i, c_identifier(name));
            if !ty.shape.is_empty() {
                decl.push(format!("const {} *{}", ctype_of(&ty.elem), c));
                own.push(Val::A(stored(ty, &c, None, false)));
            } else {
                decl.push(format!("{} {}", ctype_of(&ty.elem), c));
                own.push(Val::S(S::new(&ty.elem, c, BTreeSet::new())));
            }
        }
        self.params.push(own);
        let r = spec.result_ty.clone();
        let mut out: Option<Dest> = None;
        if !r.shape.is_empty() {
            decl.push(format!("{} *out", ctype_of(&r.elem)));
            out = Some(Dest { ptr: "out".to_string(), key: Some("out".to_string()), ty: r.clone(), fresh: true });
        }
        let mut head: Vec<String> = Vec::new();
        for (slot, (name, ty)) in spec.local_names.iter().zip(&spec.locals).enumerate() {
            let cid = sanitize(&c_identifier(name));
            if !ty.shape.is_empty() {
                let field = self.em.new_local(&self.tag.clone(), slot, &cid, ty);
                self.locals.insert(slot, (false, format!("p->{}", field)));
            } else {
                let c = format!("v{}_{}", slot, cid);
                self.locals.insert(slot, (true, c.clone()));
                head.push(format!("    {} {} = 0;", ctype_of(&ty.elem), c));
                head.push(format!("    (void){};", c));
                self.code.scalars += 1;
            }
        }
        let stmts: Vec<N> = spec.stmts().to_vec();
        for (i, s) in stmts.iter().enumerate() {
            let last = i == stmts.len() - 1;
            let sp = spec.stmt_spans.get(i).copied().flatten().or(s.span);
            self.comment(sp);
            let v = self.compile(s, if last { out.as_ref() } else { None });
            if last {
                if !r.shape.is_empty() {
                    self.store(v.unwrap(), out.as_ref().unwrap());
                } else if r.elem != "void" {
                    let c = v.unwrap().c();
                    self.code.emit(&format!("return {};", c));
                }
            }
            self.release();
        }
        let rtype = if !r.shape.is_empty() || r.elem == "void" { "void" } else { ctype_of(&r.elem) };
        let sig = format!("static {} {}({})", rtype, self.em.fn_name(&spec), decl.join(", "));
        let mut body: Vec<String> = vec!["    (void)p;".to_string()];
        for (i, name) in spec.param_names.iter().enumerate() {
            body.push(format!("    (void)a{}_{};", i, c_identifier(name)));
        }
        body.extend(head);
        body.extend(std::mem::take(&mut self.code.lines));
        if self.em.hls.is_some() {
            body.insert(0, "#pragma HLS inline".to_string());
        }
        (sig, mark_unused(body))
    }

    fn comment(&mut self, sp: Option<crate::ast::Sp>) {
        let Some(sp) = sp else { return };
        let Some(src) = self.em.sources.get(self.spec.source).cloned() else { return };
        let line = src.line(sp.0);
        self.line = line;
        let seg = src.text.get(sp.0..sp.1).unwrap_or("");
        let text: String = seg.split_whitespace().collect::<Vec<_>>().join(" ");
        let chars: Vec<char> = text.chars().collect();
        let text: String = if chars.len() > 72 { format!("{}...", chars[..69].iter().collect::<String>()) } else { text };
        let text = text.replace("*/", "* /").replace("/*", "/ *").replace('\\', "\\\\");
        let name = self.em.source_name.clone();
        if self.em.line_directives {
            self.code.lines.push(format!("#line {} \"{}\"", line, name));
        }
        self.code.emit(&format!("/* {}:{}: {} */", name, line, text));
    }
}

// ---------------------------------------------------------------------------
// The whole program
// ---------------------------------------------------------------------------

/// Why `emit` refused a program.
#[derive(Debug)]
pub enum EmitError {
    /// A `ValueError` of the Python emitter: a bad prefix, an entry that clashes with a setter.
    Value(String),
    /// A public C name that an extern would take: (message, hint, span, source)
    Static(String, String, (usize, usize), usize),
    Unsupported(String),
}

impl Em {
    fn extern_proto(&self, x: &crate::sig::ExternDecl) -> String {
        let sig = &x.signature;
        let ctype = |t: &Ty| ctype_of(&t.elem);
        let mut decl: Vec<String> = Vec::new();
        for (i, ty) in sig.params.iter().enumerate() {
            if ty.elem == "str" {
                decl.push(format!("const char *a{}", i));
            } else if !ty.shape.is_empty() {
                let d: Vec<String> = ty.shape.iter().map(|d| d.to_string()).collect();
                decl.push(format!("const {} *a{} /* [{}] */", ctype(ty), i, d.join(",")));
            } else {
                decl.push(format!("{} a{}", ctype(ty), i));
            }
        }
        let r = &sig.result;
        if !r.shape.is_empty() {
            let d: Vec<String> = r.shape.iter().map(|d| d.to_string()).collect();
            decl.push(format!("{} *out /* [{}] */", ctype(r), d.join(",")));
        }
        let rtype = if !r.shape.is_empty() || r.elem == "void" { "void" } else { ctype(r) };
        let args = if decl.is_empty() { "void".to_string() } else { decl.join(", ") };
        format!("{} {}({})", rtype, c_identifier(&x.name), args)
    }

    /// The deepest chain of C calls from an entry and an estimate of its stack use.
    fn stack(&self, prog: &Program) -> (Vec<String>, usize) {
        let word = (self.int_bits.max(self.float_bits).max(32) / 8) as usize;
        let by_index: HashMap<usize, &F> = prog.fns.iter().map(|f| (f.index.get(), f)).collect();
        let mut memo: HashMap<usize, (Vec<String>, usize)> = HashMap::new();
        fn depth(me: &Em, f: &F, word: usize, by_index: &HashMap<usize, &F>, memo: &mut HashMap<usize, (Vec<String>, usize)>) -> (Vec<String>, usize) {
            if let Some(m) = memo.get(&f.index.get()) {
                return m.clone();
            }
            let (scalars, calls) = me.frames.get(&f.index.get()).cloned().unwrap_or((0, Vec::new()));
            let own = 16 + word * (f.param_tys.len() + 2 + scalars);
            let mut best: (Vec<String>, usize) = (Vec::new(), 0);
            for g in &calls {
                let cand = depth(me, by_index[&g.index.get()], word, by_index, memo);
                if cand.1 > best.1 || (cand.1 == best.1 && cand.0.len() > best.0.len()) {
                    best = cand;
                }
            }
            let mut chain = vec![me.fn_name(f)];
            chain.extend(best.0);
            let r = (chain, own + best.1);
            memo.insert(f.index.get(), r.clone());
            r
        }
        let mut best: (Vec<String>, usize) = (Vec::new(), 0);
        for e in &prog.entries {
            let (chain, size) = depth(self, &e.f, word, &by_index, &mut memo);
            let own = 16 + word * (e.f.param_tys.len() + 2);
            let mut c = vec![format!("{}_{}", self.prefix, e.c_name)];
            c.extend(chain);
            let cand = (c, size + own);
            if cand.1 > best.1 {
                best = cand;
            }
        }
        best
    }
}

/// `(prototype, body lines)` of `T <P>_<entry>_window(const T *win)`: the reduction of one window.
fn window_function(em: &mut Em, e: &Entry, m: &Window) -> (String, Vec<String>) {
    let K::Reduce { a: red_a, .. } = &m.reduce.k else { unreachable!() };
    let elem = m.reduce.ty.elem.clone();
    let mut ctx = Gen::new(em, e.f.clone());
    ctx.em.window_leaves.insert(Rc::as_ptr(&m.gather) as usize);
    ctx.params.push(Vec::new());
    let ev = ctx.compile(red_a, None).unwrap();
    let items: Vec<String> = (0..m.taps).map(|j| ev.at(&[j.to_string(), "0".to_string()])).collect();
    let op = m.op.clone();
    let ex = ctx.expander(&op, &elem, &m.reduce);
    let hls = ctx.em.hls.clone();
    let result;
    if (elem == "float" && !hls.as_ref().map(|h| h.reassociate).unwrap_or(false)) || hls.is_none() {
        let acc = ctx.temp(&elem, &items[0], false);
        for it in &items[1..] {
            ctx.code.emit(&format!("{} = {};", acc, ex.expand(&acc, it)));
        }
        result = acc;
    } else {
        let mut items = items;
        while items.len() > 1 {
            let mut nxt: Vec<String> = Vec::new();
            let mut k = 0;
            while k + 1 < items.len() {
                nxt.push(ex.expand(&items[k], &items[k + 1]));
                k += 2;
            }
            if items.len() % 2 == 1 {
                nxt.push(items.last().unwrap().clone());
            }
            items = nxt;
        }
        result = ctx.temp(&elem, &items[0], true);
    }
    let ctype = ctype_of(&elem);
    let proto = format!("{} {}_{}_window(const {} *win)", ctype, ctx.em.prefix, e.c_name, ctype_of(&e.decl.signature.params[0].elem));
    let mut body = vec!["#pragma HLS inline".to_string()];
    body.extend(mark_unused(std::mem::take(&mut ctx.code.lines)));
    body.push(format!("    return {};", result));
    (proto, body)
}

/// Emit C for a lowered program. `prefix` is the C prefix of every public name (default: the program's name): `<prefix>_t`,
/// `<prefix>_init`, `<prefix>_<entry>`.
pub fn emit(program: &Program, report: &crate::check::Report, prefix: Option<&str>, float_bits: u32, line_directives: bool, hls: Option<HlsOptions>) -> Result<Emitted, EmitError> {
    if float_bits != 32 && float_bits != 64 {
        return Err(EmitError::Value("float_bits must be 32 or 64".to_string()));
    }
    let prefix = prefix.filter(|p| !p.is_empty()).map(String::from).unwrap_or_else(|| program.name.clone());
    let valid = prefix.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false) && prefix.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid || prefix == "kg" || prefix.starts_with("kg_") || C_KEYWORDS.contains(&prefix.as_str()) {
        return Err(EmitError::Value(format!(
            "'{}' cannot be the C prefix of a program: it must be a C identifier that is not a C keyword and does not start with kg_",
            prefix
        )));
    }
    let mut public: HashSet<String> = program.entries.iter().map(|e| format!("{}_{}", prefix, e.c_name)).collect();
    for s in ["init", "size", "t", "storage_t"] {
        public.insert(format!("{}_{}", prefix, s));
    }
    for g in program.globals.iter().filter(|g| g.param) {
        public.insert(format!("{}_set_{}", prefix, c_identifier(&g.name)));
    }
    for e in &program.entries {
        if program.globals.iter().any(|g| g.param && e.c_name == format!("set_{}", c_identifier(&g.name))) {
            return Err(EmitError::Value(format!("the entry {} has the C name of the setter of a parameter", e.c_name)));
        }
    }
    for x in &program.externs {
        let cn = c_name_of(x);
        if public.contains(&cn) || cn == prefix {
            return Err(EmitError::Static(
                format!("the C name of extern {} is '{}', which this program generates itself", x.name, cn),
                "rename the extern, or compile with another --prefix".to_string(),
                x.span,
                x.src,
            ));
        }
    }
    let sources: Vec<Rc<SrcText>> = report.sources.iter().map(|s| Rc::new(SrcText::new(&s.name, &s.source.text))).collect();
    let src_name = report.sources.get(program.source).map(|s| s.name.clone()).filter(|n| !n.is_empty()).unwrap_or_else(|| program.name.clone());
    let source_name = src_name.replace('\\', "/").rsplit('/').next().unwrap_or("").to_string();
    let mut em = Em {
        hls,
        window_leaves: HashSet::new(),
        prefix: prefix.clone(),
        int_bits: program.int_bits,
        float_bits,
        line_directives,
        source_name,
        sources,
        tables: Vec::new(),
        table_keys: HashMap::new(),
        slots: Vec::new(),
        need_math: Rc::new(Cell::new(false)),
        writes_c: HashMap::new(),
        reads_c: HashMap::new(),
        uses_c: HashMap::new(),
        frames: HashMap::new(),
        err: None,
    };
    run(&mut em, program).map_err(EmitError::Unsupported)
}

fn run(em: &mut Em, prog: &Program) -> Result<Emitted, String> {
    let p = em.prefix.clone();
    let mut fns: Vec<(F, String, Vec<String>)> = Vec::new();
    for f in &prog.fns {
        if em.inline(f) {
            continue;
        }
        let (sig, body, scalars, calls) = {
            let mut ctx = Gen::new(em, f.clone());
            let (sig, body) = ctx.build();
            (sig, body, ctx.code.scalars, ctx.calls.clone())
        };
        fns.push((f.clone(), sig, body));
        em.frames.insert(f.index.get(), (scalars, calls));
    }
    if let Some(e) = em.err.take() {
        return Err(e);
    }
    let mut windows: Vec<(usize, Window, String, Vec<String>)> = Vec::new();
    if em.hls.is_some() {
        for (ei, e) in prog.entries.iter().enumerate() {
            if let Some(m) = match_window(prog, e) {
                let (proto, body) = window_function(em, e, &m);
                windows.push((ei, m, proto, body));
            }
        }
    }
    let state: Vec<&GlobalDef> = prog.globals.iter().filter(|g| g.state).collect();
    let consts: Vec<&GlobalDef> = prog.globals.iter().filter(|g| !g.state && !g.ty.shape.is_empty()).collect();
    for g in &consts {
        em.table(&g.ty, &g.value, Some(&format!("kg_c_{}", c_identifier(&g.name))));
    }
    for g in &state {
        if !g.ty.shape.is_empty() {
            em.table(&g.ty, &g.value, Some(&format!("kg_i_{}", c_identifier(&g.name))));
        }
    }

    // ---- the entries
    let mut wrappers: Vec<(String, Vec<String>)> = Vec::new();
    let mut protos: Vec<(String, String)> = Vec::new();
    let mut entry_slots = 0;
    for e in &prog.entries {
        let sig = &e.decl.signature;
        let f = &e.f;
        let want = &sig.result;
        let ctype = |t: &Ty| ctype_of(&t.elem);
        let mut decl = vec![format!("{}_t *p", p)];
        let mut args = vec!["p".to_string()];
        let mut names: Vec<String> = Vec::new();
        for (i, (name, ty)) in f.param_names.iter().zip(&sig.params).enumerate() {
            let c = sanitize(&c_identifier(name));
            let c = if !["p", "out", "i"].contains(&c.as_str()) && !names.contains(&c) && !C_KEYWORDS.contains(&c.as_str()) { c } else { format!("a{}_{}", i, c) };
            names.push(c.clone());
            let dims: Vec<String> = ty.shape.iter().map(|d| d.to_string()).collect();
            if !ty.shape.is_empty() {
                decl.push(format!("const {} *{} /* [{}] */", ctype(ty), c, dims.join(",")));
            } else {
                decl.push(format!("{} {}", ctype(ty), c));
            }
            args.push(c);
        }
        if !want.shape.is_empty() {
            let dims: Vec<String> = want.shape.iter().map(|d| d.to_string()).collect();
            decl.push(format!("{} *out /* [{}] */", ctype(want), dims.join(",")));
        }
        let rtype = if !want.shape.is_empty() || want.elem == "void" { "void" } else { ctype(want) };
        let proto = format!("{} {}_{}({})", rtype, p, e.c_name, decl.join(", "));
        let fname = em.fn_name(f);
        let mut a2 = args.clone();
        if !f.result_ty.shape.is_empty() && f.result_ty.elem == want.elem {
            a2.push("out".to_string());
        }
        let mut call = format!("{}({})", fname, a2.join(", "));
        let mut body: Vec<String> = Vec::new();
        if want.elem == "void" {
            if !f.result_ty.shape.is_empty() {
                let field = format!("e{}", entry_slots);
                entry_slots += 1;
                set_slot(em, &field, &f.result_ty.elem, size_of(&f.result_ty.shape), format!("discarded result of {}", e.c_name));
                let mut a3 = args.clone();
                a3.push(format!("p->{}", field));
                call = format!("{}({})", fname, a3.join(", "));
            }
            let v = if f.result_ty.elem != "void" && f.result_ty.shape.is_empty() { "(void)" } else { "" };
            body.push(format!("    {}{};", v, call));
        } else if want.shape.is_empty() {
            body.push(format!("    return {}{};", if want.elem != f.result_ty.elem { "(kg_float)" } else { "" }, call));
        } else if f.result_ty.elem == want.elem {
            body.push(format!("    {};", call));
        } else {
            let field = format!("e{}", entry_slots);
            entry_slots += 1;
            let size = size_of(&want.shape);
            set_slot(em, &field, &f.result_ty.elem, size, format!("int result of {} before conversion", e.c_name));
            let mut a3 = args.clone();
            a3.push(format!("p->{}", field));
            body.push(format!("    {}({});", fname, a3.join(", ")));
            body.push(format!("    for (kg_int i = 0; i < {}; i++) {{", size));
            body.push(format!("        out[i] = (kg_float)p->{}[i];", field));
            body.push("    }".to_string());
        }
        wrappers.push((proto.clone(), body));
        protos.push((proto, format!("entry {}: {}", e.decl.name, sig)));
    }

    // ---- the instance struct: 8-byte fields first, so there is no padding between fields
    struct Field {
        elem: String,
        name: String,
        size: usize,
        note: String,
        state: bool,
    }
    let mut fields: Vec<Field> = Vec::new();
    for g in &state {
        let cid = c_identifier(&g.name);
        fields.push(Field { elem: g.ty.elem.clone(), name: format!("g_{}", cid), size: if g.ty.shape.is_empty() { 0 } else { size_of(&g.ty.shape) }, note: format!("state {}: {}", g.name, g.ty), state: true });
    }
    for s in &em.slots {
        fields.push(Field { elem: s.elem.clone(), name: s.field.clone(), size: s.size.max(0), note: s.note.clone(), state: false });
    }
    // (a slot field is always an array of at least one element)
    for f in fields.iter_mut() {
        if !f.state && f.size == 0 {
            f.size = 1;
        }
    }
    fields.sort_by_key(|f| std::cmp::Reverse(em.isize(&f.elem)));
    let bytes = |f: &Field| em.isize(&f.elem) * f.size.max(1);
    let state_bytes: usize = fields.iter().filter(|f| f.state).map(bytes).sum();
    let slot_bytes: usize = fields.iter().filter(|f| !f.state).map(bytes).sum();
    let mut total = state_bytes + slot_bytes;
    if fields.is_empty() {
        total = (em.int_bits / 8) as usize;
    }
    let widest = fields.iter().map(|f| em.isize(&f.elem)).max().unwrap_or((em.int_bits / 8) as usize);
    total = total.div_ceil(widest) * widest;
    let const_bytes: usize = em.tables.iter().map(|(_, ty, arr)| em.isize(&ty.elem) * arr.len()).sum();

    let guard = format!("{}_H", p.to_uppercase());
    let sname = em.source_name.clone();
    let mut h: Vec<String> = vec![format!("/* {}.h - generated by KlongPy (kgpy compile c) from {}. Do not edit. */", p, sname), format!("#ifndef {}", guard), format!("#define {}", guard), String::new()];
    h.push("/* The element types this program was compiled for. */".to_string());
    if em.int_bits == 64 {
        h.extend(["#ifndef KG_INT64", "#define KG_INT64 1", "#endif"].map(String::from));
    } else {
        h.extend(["#ifdef KG_INT64".to_string(), format!("#error \"{} was compiled for a 32-bit kg_int\"", p), "#endif".to_string()]);
    }
    if em.float_bits == 64 {
        h.extend(["#ifndef KG_FLOAT64", "#define KG_FLOAT64 1", "#endif"].map(String::from));
    } else {
        h.extend(["#ifdef KG_FLOAT64".to_string(), format!("#error \"{} was compiled for a 32-bit kg_float\"", p), "#endif".to_string()]);
    }
    let need_math = em.need_math.get();
    if need_math {
        h.extend(
            [
                "/* The program uses ^ with a float result: pow from <math.h>, or kg_pow of klong_rt.h under KG_FREESTANDING. */",
                "#ifndef KG_NEED_MATH",
                "#define KG_NEED_MATH 1",
                "#endif",
            ]
            .map(String::from),
        );
    }
    h.push("#include \"klong_rt.h\"".to_string());
    h.push(String::new());
    h.extend(
        [
            "/*".to_string(),
            " * One instance of the program: its state and the storage of its array temporaries.".to_string(),
            " * The caller chooses where it lives (a static, a heap block, external RAM) and may".to_string(),
            " * keep several. One instance must not be entered from two threads at once. The fields".to_string(),
            " * are private to the generated code.".to_string(),
            " */".to_string(),
            format!("struct {} {{", p),
        ],
    );
    if fields.is_empty() {
        h.push("    kg_int kg_unused; /* the program has no state and needs no array storage */".to_string());
    }
    for f in &fields {
        let ctype = ctype_of(&f.elem);
        let dim = if f.size > 0 { format!("[{}]", f.size) } else { String::new() };
        h.push(format!("    {} {}{}; /* {} */", ctype, f.name, dim, f.note));
    }
    h.extend(["};".to_string(), format!("typedef struct {} {}_t;", p, p), format!("typedef struct {} {}_storage_t; /* for static allocation */", p, p), format!("#define {}_SIZE (sizeof(struct {}))", p.to_uppercase(), p), String::new()]);
    let pdefs: Vec<&GlobalDef> = prog.globals.iter().filter(|g| g.param).collect();
    h.push("/* Sets the state to its initial values. Call it once before the first entry. */".to_string());
    h.push(format!("void {}_init({}_t *p);", p, p));
    h.push(format!("/* sizeof({}_t), for a host that allocates the instance itself. */", p));
    h.push(format!("size_t {}_size(void);", p));
    h.push(String::new());
    if !pdefs.is_empty() {
        h.push("/* Parameters (.param): input registers. Set one between runs; init puts back its default. */".to_string());
    }
    for g in &pdefs {
        h.push(format!("void {}_set_{}({}_t *p, {} value); /* {}: {} */", p, c_identifier(&g.name), p, ctype_of(&g.ty.elem), g.name, g.ty));
    }
    for (ei, m, sig_text, _) in &windows {
        let e = &prog.entries[*ei];
        h.push(format!("/* hardware: one output of entry {} from the last {} samples (win[0] the oldest). */", e.decl.name, m.taps));
        h.push(format!("{};", sig_text));
    }
    if !protos.is_empty() {
        h.push("/* Entries. An array argument and the out array must not overlap. */".to_string());
    }
    for (proto, note) in &protos {
        h.push(format!("{}; /* {} */", proto, note));
    }
    if !prog.externs.is_empty() {
        h.push(String::new());
        h.push("/* Externs: functions the target must define. */".to_string());
    }
    for x in &prog.externs {
        h.push(format!("{}; /* Klong name {}: {} (module {}) */", em.extern_proto(x), x.name, x.signature, x.module));
    }
    h.push(String::new());
    h.push(format!("#endif /* {} */", guard));
    h.push(String::new());

    let mut c: Vec<String> = vec![
        format!("/* {}.c - generated by KlongPy ({}) from {}. Do not edit. */", p, if em.hls.is_some() { "kgpy fpga source, the streamable core" } else { "kgpy compile c" }, sname),
        "#define KG_RT_IMPLEMENTATION /* the weak kg_trap (and freestanding memcpy, memset) are defined in this file */".to_string(),
        format!("#include \"{}.h\"", p),
        String::new(),
    ];
    for (name, ty, arr) in &em.tables {
        c.extend(em.table_text(name, ty, arr));
        c.push(String::new());
    }
    for (_, sig, _) in &fns {
        c.push(format!("{};", sig));
    }
    if !fns.is_empty() {
        c.push(String::new());
    }
    for (f, sig, body) in &fns {
        let params: Vec<String> = f.param_tys.iter().map(|t| t.to_string()).collect();
        c.push(format!("/* {}({}) -> {} */", f.name, params.join("; "), f.result_ty));
        c.push(format!("{} {{", sig));
        c.extend(body.iter().cloned());
        c.push("}".to_string());
        c.push(String::new());
    }
    c.push(format!("void {}_init({}_t *p) {{", p, p));
    c.push("    (void)p;".to_string());
    for g in &state {
        let cid = c_identifier(&g.name);
        if !g.ty.shape.is_empty() {
            c.push(format!("    for (kg_int i = 0; i < {}; i++) {{", size_of(&g.ty.shape)));
            c.push(format!("        p->g_{}[i] = kg_i_{}[i];", cid, cid));
            c.push("    }".to_string());
        } else {
            c.push(format!("    p->g_{} = {};", cid, em.literal(&g.ty.elem, &g.value)));
        }
    }
    c.extend(["}".to_string(), String::new(), format!("size_t {}_size(void) {{", p), format!("    return sizeof({}_t);", p), "}".to_string(), String::new()]);
    for (ei, m, sig_text, body) in &windows {
        let e = &prog.entries[*ei];
        c.push(format!("/* the window of entry {}: the result of one output from the last {} samples */", e.decl.name, m.taps));
        c.push(format!("{} {{", sig_text));
        c.extend(body.iter().cloned());
        c.push("}".to_string());
        c.push(String::new());
    }
    for g in &pdefs {
        let cid = c_identifier(&g.name);
        c.push(format!("void {}_set_{}({}_t *p, {} value) {{", p, cid, p, ctype_of(&g.ty.elem)));
        c.push(format!("    p->g_{} = value;", cid));
        c.push("}".to_string());
        c.push(String::new());
    }
    for (proto, body) in &wrappers {
        c.push(format!("{} {{", proto));
        c.extend(body.iter().cloned());
        c.push("}".to_string());
        c.push(String::new());
    }
    let ctext = c.join("\n");
    let (chain, stack) = em.stack(prog);
    let report = SizeReport {
        int_bits: em.int_bits,
        float_bits: em.float_bits,
        const_bytes,
        instance_bytes: total,
        state_bytes,
        slot_bytes,
        call_depth: chain.len(),
        call_chain: chain,
        stack_bytes: stack,
        c_lines: ctext.matches('\n').count() + 1,
        need_math,
        entries: prog.entries.iter().map(|e| format!("{}_{}", p, e.c_name)).collect(),
        externs: prog.externs.iter().map(|x| c_name_of(x)).collect(),
    };
    Ok(Emitted { prefix: p, c: ctext, h: h.join("\n"), report, windows })
}

fn set_slot(em: &mut Em, field: &str, elem: &str, size: usize, note: String) {
    let s = Slot { field: field.to_string(), elem: elem.to_string(), size, note };
    match em.slots.iter_mut().find(|x| x.field == field) {
        Some(x) => *x = s,
        None => em.slots.push(s),
    }
}
