//! The typed IR of the static compiler (`klongpy/static/ir.py`): lowering (`infer.rs`) builds one `FnSpec` per function and tuple of
//! argument types, the emitters walk it. Evaluation order is part of its meaning: `children` lists the operands in the order they are
//! evaluated (a dyad evaluates its right operand first).
use crate::ast::Sp;
use crate::sig::{EntryDecl, ExternDecl, Ty};
use crate::util::py_float;
use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

/// A compile-time value: an int or float scalar, a table (the shape is in the node's type) or a string literal.
#[derive(Clone, Debug, PartialEq)]
pub enum V {
    I(i64),
    F(f64),
    AI(Vec<i64>),
    AF(Vec<f64>),
    S(String),
}

impl V {
    pub fn is_scalar(&self) -> bool {
        matches!(self, V::I(_) | V::F(_))
    }
    pub fn len(&self) -> usize {
        match self {
            V::AI(a) => a.len(),
            V::AF(a) => a.len(),
            _ => 1,
        }
    }
    pub fn as_f64(&self) -> Vec<f64> {
        match self {
            V::I(i) => vec![*i as f64],
            V::F(f) => vec![*f],
            V::AI(a) => a.iter().map(|x| *x as f64).collect(),
            V::AF(a) => a.clone(),
            V::S(_) => Vec::new(),
        }
    }
    pub fn as_i64(&self) -> Vec<i64> {
        match self {
            V::I(i) => vec![*i],
            V::F(f) => vec![*f as i64],
            V::AI(a) => a.clone(),
            V::AF(a) => a.iter().map(|x| *x as i64).collect(),
            V::S(_) => Vec::new(),
        }
    }
    pub fn int(&self) -> Option<i64> {
        if let V::I(i) = self {
            Some(*i)
        } else {
            None
        }
    }
}

pub type N = Rc<Node>;
pub type F = Rc<FnSpec>;

#[derive(Debug)]
pub struct Node {
    pub ty: Ty,
    pub span: Option<Sp>,
    pub k: K,
}

#[derive(Debug)]
pub enum K {
    Const(V),
    GlobalRef { name: String, state: bool, known: Option<V> },
    LocalRef { slot: usize, name: String },
    ParamRef { index: usize, name: String },
    Unary { op: String, a: N, known: Option<V> },
    Binary { op: String, a: N, b: N },
    Convert { a: N },
    Match { a: N, b: N },
    Index { a: N, i: N, checked: bool },
    Gather { a: N, idx: N, checked: bool },
    Take { a: N, n: i64 },
    Drop { a: N, n: i64 },
    Rotate { a: N, n: N },
    Reverse { a: N },
    Transpose { a: N },
    Reshape { a: N },
    Join { a: N, b: N },
    Iota { n: i64 },
    MakeList { items: Vec<N> },
    Amend { a: N, value: N, idx: N, checked: bool },
    Cond { c: N, t: N, e: N },
    Call { f: F, args: Vec<N> },
    ExternCall { ext: Rc<ExternDecl>, args: Vec<N> },
    Map { f: F, a: N, caps: Vec<N> },
    Map2 { f: F, a: N, b: N, caps: Vec<N> },
    Reduce { f: F, a: N, seed: Option<N>, caps: Vec<N> },
    Scan { f: F, a: N, seed: Option<N>, caps: Vec<N>, acc_ty: Ty },
    EachPair { f: F, a: N, caps: Vec<N> },
    /// kind "iterate": `count f:*init`; "while": `pred f:~init` (`count` is None)
    Loop { kind: String, f: F, init: N, count: Option<N>, pred: Option<F>, caps: Vec<N>, pcaps: Vec<N> },
    Assign { target: N, value: N },
    Seq { stmts: Vec<N> },
}

impl Node {
    pub fn new(ty: Ty, span: Option<Sp>, k: K) -> N {
        Rc::new(Node { ty, span, k })
    }

    /// The compile-time value when lowering knows it.
    pub fn cval(&self) -> Option<V> {
        match &self.k {
            K::Const(v) => {
                if self.ty.elem == "str" {
                    None
                } else {
                    Some(v.clone())
                }
            }
            K::GlobalRef { known, .. } => known.clone(),
            K::Unary { known, .. } => known.clone(),
            K::Iota { n } => {
                if *n <= 4096 {
                    Some(V::AI((0..*n).collect()))
                } else {
                    None
                }
            }
            K::Seq { stmts } => stmts.last().and_then(|s| s.cval()),
            _ => None,
        }
    }

    pub fn is_const(&self) -> bool {
        matches!(self.k, K::Const(_))
    }

    /// The class name Python gives the node (`type(n).__name__`).
    pub fn class(&self) -> &'static str {
        match &self.k {
            K::Const(_) => "Const",
            K::GlobalRef { .. } => "GlobalRef",
            K::LocalRef { .. } => "LocalRef",
            K::ParamRef { .. } => "ParamRef",
            K::Unary { .. } => "Unary",
            K::Binary { .. } => "Binary",
            K::Convert { .. } => "Convert",
            K::Match { .. } => "Match",
            K::Index { .. } => "Index",
            K::Gather { .. } => "Gather",
            K::Take { .. } => "Take",
            K::Drop { .. } => "Drop",
            K::Rotate { .. } => "Rotate",
            K::Reverse { .. } => "Reverse",
            K::Transpose { .. } => "Transpose",
            K::Reshape { .. } => "Reshape",
            K::Join { .. } => "Join",
            K::Iota { .. } => "Iota",
            K::MakeList { .. } => "MakeList",
            K::Amend { .. } => "Amend",
            K::Cond { .. } => "Cond",
            K::Call { .. } => "Call",
            K::ExternCall { .. } => "ExternCall",
            K::Map { .. } => "Map",
            K::Map2 { .. } => "Map2",
            K::Reduce { .. } => "Reduce",
            K::Scan { .. } => "Scan",
            K::EachPair { .. } => "EachPair",
            K::Loop { .. } => "Loop",
            K::Assign { .. } => "Assign",
            K::Seq { .. } => "Seq",
        }
    }

    /// The operand nodes in evaluation order (`children` of ir.py).
    pub fn children(&self) -> Vec<&N> {
        let mut o: Vec<&N> = Vec::new();
        match &self.k {
            K::Const(_) | K::GlobalRef { .. } | K::LocalRef { .. } | K::ParamRef { .. } | K::Iota { .. } => {}
            K::Unary { a, .. } | K::Convert { a } | K::Take { a, .. } | K::Drop { a, .. } | K::Reverse { a } | K::Transpose { a } | K::Reshape { a } => o.push(a),
            K::Binary { a, b, .. } | K::Match { a, b } | K::Join { a, b } => {
                o.push(b);
                o.push(a);
            }
            K::Index { a, i, .. } => {
                o.push(i);
                o.push(a);
            }
            K::Gather { a, idx, .. } => {
                o.push(idx);
                o.push(a);
            }
            K::Rotate { a, n } => {
                o.push(a);
                o.push(n);
            }
            K::MakeList { items } => o.extend(items.iter()),
            K::Amend { a, value, idx, .. } => {
                o.push(idx);
                o.push(value);
                o.push(a);
            }
            K::Cond { c, t, e } => {
                o.push(c);
                o.push(t);
                o.push(e);
            }
            K::Call { args, .. } | K::ExternCall { args, .. } => o.extend(args.iter()),
            K::Map { a, caps, .. } | K::EachPair { a, caps, .. } => {
                o.push(a);
                o.extend(caps.iter());
            }
            K::Map2 { a, b, caps, .. } => {
                o.push(a);
                o.push(b);
                o.extend(caps.iter());
            }
            K::Reduce { a, seed, caps, .. } | K::Scan { a, seed, caps, .. } => {
                if let Some(s) = seed {
                    o.push(s);
                }
                o.push(a);
                o.extend(caps.iter());
            }
            K::Loop { init, count, caps, pcaps, .. } => {
                if let Some(c) = count {
                    o.push(c);
                }
                o.push(init);
                o.extend(caps.iter());
                o.extend(pcaps.iter());
            }
            K::Assign { value, .. } => o.push(value),
            K::Seq { stmts } => o.extend(stmts.iter()),
        }
        o
    }

    /// n and every node below it (not into the FnSpecs it names).
    pub fn walk<'a>(&'a self, out: &mut Vec<&'a Node>) {
        out.push(self);
        for c in self.children() {
            c.walk(out);
        }
    }

    pub fn walked(&self) -> Vec<&Node> {
        let mut v = Vec::new();
        self.walk(&mut v);
        v
    }

    /// The FnSpecs a node names directly (`fn`, then `pred`).
    pub fn fnspecs(&self) -> Vec<&F> {
        match &self.k {
            K::Map { f, .. } | K::Map2 { f, .. } | K::Reduce { f, .. } | K::Scan { f, .. } | K::EachPair { f, .. } => vec![f],
            K::Call { f, .. } => vec![f],
            K::Loop { f, pred, .. } => {
                let mut v = vec![f];
                if let Some(p) = pred {
                    v.push(p);
                }
                v
            }
            _ => Vec::new(),
        }
    }

    /// Does evaluating the node assign a variable or call an extern (directly or through a function)?
    pub fn effects(&self) -> bool {
        if matches!(self.k, K::Assign { .. } | K::ExternCall { .. }) {
            return true;
        }
        if self.fnspecs().iter().any(|f| f.effects()) {
            return true;
        }
        self.children().iter().any(|c| c.effects())
    }
}

#[derive(Debug)]
pub struct FnSpec {
    pub name: String,
    pub index: Cell<usize>,
    /// "fn", "lambda" or "op"
    pub kind: &'static str,
    pub param_tys: Vec<Ty>,
    pub n_own: usize,
    pub param_names: Vec<String>,
    pub locals: Vec<Ty>,
    pub local_names: Vec<String>,
    pub body: N,
    pub result_ty: Ty,
    pub span: Option<Sp>,
    pub source: usize,
    pub stmt_spans: Vec<Option<Sp>>,
    effects: Cell<u8>,
}

impl FnSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        kind: &'static str,
        param_tys: Vec<Ty>,
        n_own: usize,
        param_names: Vec<String>,
        locals: Vec<Ty>,
        local_names: Vec<String>,
        body: N,
        result_ty: Ty,
        span: Option<Sp>,
        source: usize,
        stmt_spans: Vec<Option<Sp>>,
    ) -> FnSpec {
        FnSpec { name, index: Cell::new(usize::MAX), kind, param_tys, n_own, param_names, locals, local_names, body, result_ty, span, source, stmt_spans, effects: Cell::new(0) }
    }
    pub fn label(&self) -> String {
        format!("{}#{}", self.name, self.index.get() as isize)
    }
    pub fn effects(&self) -> bool {
        match self.effects.get() {
            1 => false,
            2 => true,
            _ => {
                let r = self.body.effects();
                self.effects.set(if r { 2 } else { 1 });
                r
            }
        }
    }
    pub fn stmts(&self) -> &[N] {
        match &self.body.k {
            K::Seq { stmts } => stmts,
            _ => std::slice::from_ref(&self.body),
        }
    }
}

#[derive(Debug)]
pub struct GlobalDef {
    pub name: String,
    pub ty: Ty,
    pub value: V,
    pub state: bool,
    pub param: bool,
}

#[derive(Debug)]
pub struct Entry {
    pub decl: EntryDecl,
    pub f: F,
    /// without the program prefix: "step", or "run__f256" when the function has several entries
    pub c_name: String,
}

pub struct Program {
    pub name: String,
    pub globals: Vec<GlobalDef>,
    /// callees before callers
    pub fns: Vec<F>,
    pub entries: Vec<Entry>,
    pub externs: Vec<Rc<ExternDecl>>,
    pub int_bits: u32,
    pub source: usize,
}

pub fn c_name_of(x: &ExternDecl) -> String {
    crate::sig::c_identifier(&x.name)
}

// ---------------------------------------------------------------------------
// The text form
// ---------------------------------------------------------------------------

pub fn num(v: &V, as_float: bool) -> String {
    match v {
        V::F(f) => py_float(*f),
        V::I(i) => i.to_string(),
        V::AI(a) => a.first().map(|x| x.to_string()).unwrap_or_default(),
        V::AF(a) => a.first().map(|x| py_float(*x)).unwrap_or_default(),
        V::S(s) => {
            let _ = as_float;
            s.clone()
        }
    }
}

fn table(v: &V) -> String {
    let (items, n): (Vec<String>, usize) = match v {
        V::AI(a) => (a.iter().take(8).map(|x| x.to_string()).collect(), a.len()),
        V::AF(a) => (a.iter().take(8).map(|x| py_float(*x)).collect(), a.len()),
        V::I(i) => (vec![i.to_string()], 1),
        V::F(f) => (vec![py_float(*f)], 1),
        V::S(_) => (Vec::new(), 0),
    };
    format!("[{}{}]", items.join(" "), if n > 8 { " ..." } else { "" })
}

fn fnl(f: &FnSpec) -> String {
    format!("<{} {}>", f.kind, f.label())
}

fn caps(c: &[N]) -> String {
    if c.is_empty() {
        String::new()
    } else {
        format!(" captures{}", c.iter().map(|q| format!(" {}", expr(q))).collect::<String>())
    }
}

pub fn expr(n: &Node) -> String {
    let t = format!(":{}", n.ty);
    match &n.k {
        K::Const(v) => {
            if n.ty.elem == "str" {
                format!("(str \"{}\")", num(v, false))
            } else if n.ty.shape.is_empty() {
                format!("({}{})", num(v, false), t)
            } else {
                format!("(table {}{})", table(v), t)
            }
        }
        K::GlobalRef { name, state, .. } => format!("({} {}{})", if *state { "state" } else { "const" }, name, t),
        K::LocalRef { slot, name } => format!("(local {} {}{})", slot, name, t),
        K::ParamRef { index, name } => format!("(param {} {}{})", index, name, t),
        K::Unary { op, a, .. } => format!("({}{} {})", op, t, expr(a)),
        K::Binary { op, a, b } => format!("({}{} {} {})", op, t, expr(a), expr(b)),
        K::Convert { a } => {
            if n.ty.elem != "float" || a.ty.elem != "int" {
                format!("(convert{} {})", t, expr(a))
            } else {
                format!("(float{} {})", t, expr(a))
            }
        }
        K::Match { a, b } => format!("(match{} {} {})", t, expr(a), expr(b)),
        K::Index { a, i, checked } => format!("(index{}{} {} {})", t, if *checked { " checked" } else { "" }, expr(a), expr(i)),
        K::Gather { a, idx, checked } => format!("(gather{}{} {} {})", t, if *checked { " checked" } else { "" }, expr(a), expr(idx)),
        K::Take { a, n: c } => format!("(take{} {} {})", t, c, expr(a)),
        K::Drop { a, n: c } => format!("(drop{} {} {})", t, c, expr(a)),
        K::Rotate { a, n: c } => format!("(rotate{} {} {})", t, expr(c), expr(a)),
        K::Reverse { a } => format!("(reverse{} {})", t, expr(a)),
        K::Transpose { a } => format!("(transpose{} {})", t, expr(a)),
        K::Reshape { a } => format!("(reshape{} {})", t, expr(a)),
        K::Join { a, b } => format!("(join{} {} {})", t, expr(a), expr(b)),
        K::Iota { .. } => format!("(iota{})", t),
        K::MakeList { items } => format!("(list{} {})", t, items.iter().map(|q| expr(q)).collect::<Vec<_>>().join(" ")),
        K::Amend { a, value, idx, checked } => format!("(amend{}{} {} {} {})", t, if *checked { " checked" } else { "" }, expr(a), expr(value), expr(idx)),
        K::Cond { c, t: th, e } => format!("(cond{} {} {} {})", t, expr(c), expr(th), expr(e)),
        K::Call { f, args } => format!("(call{} {}{})", t, fnl(f), args.iter().map(|q| format!(" {}", expr(q))).collect::<String>()),
        K::ExternCall { ext, args } => format!("(extern{} {}{})", t, ext.name, args.iter().map(|q| format!(" {}", expr(q))).collect::<String>()),
        K::Map { f, a, caps: c } => format!("(each{} {} {}{})", t, fnl(f), expr(a), caps(c)),
        K::Map2 { f, a, b, caps: c } => format!("(each2{} {} {} {}{})", t, fnl(f), expr(a), expr(b), caps(c)),
        K::EachPair { f, a, caps: c } => format!("(each-pair{} {} {}{})", t, fnl(f), expr(a), caps(c)),
        K::Reduce { f, a, seed, caps: c } | K::Scan { f, a, seed, caps: c, .. } => {
            let s = seed.as_ref().map(|s| format!(" seed {}", expr(s))).unwrap_or_default();
            let w = if matches!(n.k, K::Reduce { .. }) { "over" } else { "scan" };
            format!("({}{} {}{} {}{})", w, t, fnl(f), s, expr(a), caps(c))
        }
        K::Loop { kind, f, init, count, pred, caps: c, pcaps } => {
            let head = if kind == "iterate" { format!("count {}", expr(count.as_ref().unwrap())) } else { format!("pred {}", fnl(pred.as_ref().unwrap())) };
            format!("({}{} {} {} {}{}{})", kind, t, head, fnl(f), expr(init), caps(c), caps(pcaps))
        }
        K::Assign { target, value } => format!("(set{} {} {})", t, expr(target), expr(value)),
        K::Seq { stmts } => format!("(seq{} {})", t, stmts.iter().map(|q| expr(q)).collect::<Vec<_>>().join(" ")),
    }
}

/// The program as text, stable across runs (`dump` of ir.py).
pub fn dump(p: &Program) -> String {
    let mut out: Vec<String> = vec![format!("program {} (kg_int: {} bits)", p.name, p.int_bits)];
    for g in &p.globals {
        let kind = if g.param {
            "param"
        } else if g.state {
            "state"
        } else {
            "const"
        };
        let shown = if g.ty.shape.is_empty() { num(&g.value, false) } else { table(&g.value) };
        out.push(format!("global {} {}: {} = {}", kind, g.name, g.ty, shown));
    }
    for x in &p.externs {
        out.push(format!("extern {}: {} (C {}, module {})", x.name, x.signature, c_name_of(x), x.module));
    }
    for e in &p.entries {
        out.push(format!("entry {}: {} -> {}", e.c_name, e.decl.signature, e.f.label()));
    }
    for f in &p.fns {
        let params: Vec<String> = f.param_names.iter().zip(&f.param_tys).map(|(n, t)| format!("{}: {}", n, t)).collect();
        out.push(String::new());
        let cap = if f.param_tys.len() > f.n_own { format!("  captures {}", f.param_tys.len() - f.n_own) } else { String::new() };
        out.push(format!("fn {} [{}] ({}) -> {}{}", f.label(), f.kind, params.join("; "), f.result_ty, cap));
        for (i, (n, t)) in f.local_names.iter().zip(&f.locals).enumerate() {
            out.push(format!("  local {} {}: {}", i, n, t));
        }
        for s in f.stmts() {
            out.push(format!("  {}", expr(s)));
        }
        let (arrays, calls) = storage_inputs(f);
        if !arrays.is_empty() {
            out.push(format!("  arrays: {}", arrays.join(", ")));
        }
        if !calls.is_empty() {
            out.push(format!("  calls: {}", calls.join(", ")));
        }
        out.push(format!("  effects: {}", if f.effects() { "yes" } else { "no" }));
    }
    out.join("\n") + "\n"
}

fn is_leaf(n: &Node) -> bool {
    matches!(n.k, K::Const(_) | K::GlobalRef { .. } | K::LocalRef { .. } | K::ParamRef { .. })
}

pub fn is_fusible(n: &Node) -> bool {
    matches!(
        n.k,
        K::Unary { .. } | K::Binary { .. } | K::Convert { .. } | K::Index { .. } | K::Gather { .. } | K::Take { .. } | K::Drop { .. } | K::Rotate { .. } | K::Reverse { .. } | K::Transpose { .. } | K::Reshape { .. } | K::Join { .. } | K::Iota { .. }
    )
}

/// What the storage planner needs to know about one FnSpec: the array-valued intermediates ("kind:type fused|slot") in evaluation
/// order, and the labels of the FnSpecs called, in order of first mention.
pub fn storage_inputs(f: &FnSpec) -> (Vec<String>, Vec<String>) {
    let (mut arrays, mut calls) = (Vec::new(), Vec::new());
    for n in f.body.walked() {
        for g in n.fnspecs() {
            let l = g.label();
            if !calls.contains(&l) {
                calls.push(l);
            }
        }
        if !n.ty.shape.is_empty() && !is_leaf(n) && !matches!(n.k, K::Assign { .. } | K::Seq { .. }) {
            arrays.push(format!("{}:{} {}", n.class().to_lowercase(), n.ty, if is_fusible(n) { "fused" } else { "slot" }));
        }
    }
    (arrays, calls)
}

pub fn fn_key(f: &F) -> usize {
    Rc::as_ptr(f) as usize
}

pub type FnMap<T> = HashMap<usize, T>;
